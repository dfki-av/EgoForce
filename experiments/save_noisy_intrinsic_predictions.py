import argparse
import collections
import copy
import gc
import json
import multiprocessing as mp
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from camera_models import OVR624CameraModel
from datasets import Arm3DDataset, HOT3DLoader
from experiments.save_predictions import infer
from models import HALOAblations, LimbModel
from settings import config as cfg
from anycalib import AnyCalib
from utils.plot_utils import (
    _enable_crisp_rendering,
    _stroke_all_text,
    _save_png_supersampled,
    _blend_axes_to_paper,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_DATA_DIR = os.path.join(ROOT_DIR, '_DATA')
DEFAULT_NOISY_PREDICTIONS_DIR = os.path.join(_DATA_DIR, 'noisy_predictions')
ANYCALIB_CAM_ID = 'simple_kb:4'
ANYCALIB_INTRINSIC_ORDER = ['f', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4']


def _normalize_anycalib_intrinsics(intrinsics):
    values = np.asarray(intrinsics, dtype=np.float64).reshape(-1)
    if values.size < 7:
        raise ValueError(
            'AnyCalib intrinsics must contain at least 7 values '
            '[f, cx, cy, k1, k2, k3, k4].'
        )
    return values[:7]


def infer_anycalib_intrinsics_first_frame(config, target_dataset, device):
    """Infer AnyCalib intrinsics once from the first dataset frame."""
    frame_dataset = Arm3DDataset(config, target_dataset, return_complete_image=True)
    if len(frame_dataset) == 0:
        raise RuntimeError('Cannot infer AnyCalib intrinsics from an empty dataset.')

    _, meta = frame_dataset[0]
    image = meta['image']
    image = image if torch.is_tensor(image) else torch.tensor(image)
    image = image.to(device=device, dtype=torch.float32)

    if image.ndim != 3:
        raise ValueError(f'Expected first-frame image to be 3D, got shape {tuple(image.shape)}.')
    if image.shape[0] != 3 and image.shape[-1] == 3:
        image = image.permute(2, 0, 1)
    if image.shape[0] != 3:
        raise ValueError(f'Expected CHW image with 3 channels, got shape {tuple(image.shape)}.')
    if torch.max(image).item() > 1.0:
        image = image / 255.0

    anycalib_model = AnyCalib(model_id='anycalib_gen').to(device)
    with torch.no_grad():
        prediction = anycalib_model.predict(image, cam_id=ANYCALIB_CAM_ID)
    intrinsics = prediction['intrinsics']
    intrinsics = intrinsics.detach().cpu().numpy() if torch.is_tensor(intrinsics) else intrinsics

    del anycalib_model

    return _normalize_anycalib_intrinsics(intrinsics)


def save_anycalib_intrinsics(output_dir, dataset_name, intrinsics):
    intrinsics = _normalize_anycalib_intrinsics(intrinsics)
    payload = {
        'dataset_name': dataset_name,
        'source': 'AnyCalib(anycalib_gen)',
        'cam_id': ANYCALIB_CAM_ID,
        'frame_index': 0,
        'intrinsic_order': ANYCALIB_INTRINSIC_ORDER,
        'intrinsics': intrinsics.tolist(),
    }

    base = os.path.join(output_dir, f'{dataset_name}_anycalib_first_frame_intrinsics')
    json_path = f'{base}.json'

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    return payload, json_path


class InterpolatedNoisyAnyCalib624Dataset(Dataset):
    """
    Single-number noise model in OVR624 space:
    0%   -> GT dataset camera
    100% -> software(anycalib) camera
    >100% -> extrapolation beyond software(anycalib) camera
    """

    def __init__(
        self,
        target_dataset,
        software_intrinsics,
        noise_percent=100.0,
        ray_grid_size=21,
        radial_bins=10,
        depth_values_m=None,
        camera_error_store=None,
    ):
        self.target_dataset = target_dataset
        self.split = target_dataset.split
        self.length = len(self.target_dataset)
        self.get_camera = target_dataset.get_camera

        self.noise_percent = max(0.0, float(noise_percent))
        self.alpha = self.noise_percent / 100.0
        self.ray_grid_size = max(5, int(ray_grid_size))
        self.radial_bins = max(2, int(radial_bins))
        if depth_values_m is None:
            depth_values_m = [0.2, 0.5, 1.0, 1.5, 2.0]
        self.depth_values_m = np.asarray(depth_values_m, dtype=np.float64)

        # Can be a regular dict (single-process) or multiprocessing.Manager().dict proxy.
        self._camera_error_by_samplekey = camera_error_store if camera_error_store is not None else {}

        self.min_focal = 1e-6
        self.software_intrinsics = _normalize_anycalib_intrinsics(software_intrinsics)

    def __len__(self):
        return self.length

    def _software_theta16(self):
        sw_f = float(self.software_intrinsics[0])
        sw_cx = float(self.software_intrinsics[1])
        sw_cy = float(self.software_intrinsics[2])
        sw_k = self.software_intrinsics[3:7]

        distortion = np.zeros(12, dtype=np.float64)
        distortion[:4] = sw_k
        return np.concatenate(
            [
                np.array([sw_f, sw_f, sw_cx, sw_cy], dtype=np.float64),
                distortion,
            ],
            axis=0,
        )

    def _rational8_to_ovr12(self, distortion8):
        d8 = np.asarray(distortion8, dtype=np.float64).reshape(-1)
        out = np.zeros(12, dtype=np.float64)

        # Rational8 order: [k1, k2, p1, p2, k3, k4, k5, k6]
        # OVR624 order:    [k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4]
        if d8.size >= 8:
            out[0] = d8[0]
            out[1] = d8[1]
            out[2] = d8[4]
            out[3] = d8[5]
            out[4] = d8[6]
            out[5] = d8[7]
            out[6] = d8[2]
            out[7] = d8[3]
            return out

        n = min(d8.size, 8)
        out[:n] = d8[:n]
        return out

    def _camera_to_theta16(self, camera_params):
        focal = np.asarray(camera_params.get('focal_length', [1.0, 1.0]), dtype=np.float64).reshape(-1)
        principal = np.asarray(camera_params.get('principal_point', [0.0, 0.0]), dtype=np.float64).reshape(-1)
        projection = np.asarray(camera_params.get('projection_params', []), dtype=np.float64).reshape(-1)
        camera_type = int(camera_params.get('camera_type', 3))

        focal_xy = np.array([focal[0], focal[1] if focal.size > 1 else focal[0]], dtype=np.float64)
        principal_xy = np.array(
            [principal[0], principal[1] if principal.size > 1 else principal[0]], dtype=np.float64
        )

        if camera_type == 3:
            if projection.size >= 15:
                distortion = projection[3:15].copy()
            elif projection.size >= 12:
                distortion = projection[:12].copy()
            else:
                distortion = np.zeros(12, dtype=np.float64)
                distortion[: projection.size] = projection
        elif camera_type == 2:
            if projection.size >= 8:
                distortion = self._rational8_to_ovr12(projection[:8])
            else:
                distortion = np.zeros(12, dtype=np.float64)
        else:
            distortion = np.zeros(12, dtype=np.float64)

        return np.concatenate([focal_xy, principal_xy, distortion], axis=0)

    def _theta16_to_camera_params(self, theta):
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        focal = np.clip(theta[:2], self.min_focal, None).astype(np.float32)
        principal = theta[2:4].astype(np.float32)
        distortion = theta[4:16].astype(np.float32)
        projection = np.concatenate([np.zeros(3, dtype=np.float32), distortion], axis=0)
        return focal, principal, projection

    def _interpolate(self, theta_hw, theta_sw):
        theta = (1.0 - self.alpha) * theta_hw + self.alpha * theta_sw
        theta[:2] = np.exp(
            (1.0 - self.alpha) * np.log(np.clip(theta_hw[:2], self.min_focal, None))
            + self.alpha * np.log(np.clip(theta_sw[:2], self.min_focal, None))
        )
        return theta

    def _build_ray_grid(self, width, height):
        us = np.linspace(0.0, float(width - 1), self.ray_grid_size, dtype=np.float64)
        vs = np.linspace(0.0, float(height - 1), self.ray_grid_size, dtype=np.float64)
        uu, vv = np.meshgrid(us, vs, indexing='xy')
        return np.stack([uu.reshape(-1), vv.reshape(-1)], axis=-1)

    @staticmethod
    def _normalize(vecs):
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return vecs / norms

    def _camera_to_rays(self, cam, uv):
        uvz = np.concatenate([uv.astype(np.float32), np.ones((uv.shape[0], 1), dtype=np.float32)], axis=-1)
        xyz = cam.uvz_to_camera(uvz)
        xyz = np.asarray(xyz, dtype=np.float64)
        return self._normalize(xyz)

    @staticmethod
    def _nanmean(x):
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return np.nan
        return float(np.nanmean(x))

    def _get_samplekey(self, data):
        dataset = data['extras']['dataset']
        index = data['extras']['index']
        annotation_key = data['extras']['annotation_key']
        return f'{dataset}@{index}@{annotation_key}'

    def _compute_ray_error_stats(self, samplekey, gt_cam, noisy_cam, width, height):
        uv = self._build_ray_grid(width, height)
        gt_rays = self._camera_to_rays(gt_cam, uv)
        noisy_rays = self._camera_to_rays(noisy_cam, uv)

        cos = np.sum(gt_rays * noisy_rays, axis=-1)
        cos = np.clip(cos, -1.0, 1.0)
        angle_rad = np.arccos(cos)
        angle_deg = np.degrees(angle_rad)

        gt_center = np.asarray(gt_cam.c, dtype=np.float64).reshape(1, 2)
        radial_px = np.linalg.norm(uv - gt_center, axis=-1)
        max_r = np.linalg.norm(
            np.array(
                [
                    max(gt_center[0, 0], width - 1 - gt_center[0, 0]),
                    max(gt_center[0, 1], height - 1 - gt_center[0, 1]),
                ],
                dtype=np.float64,
            )
        )
        max_r = max(max_r, 1e-9)
        radial_norm = np.clip(radial_px / max_r, 0.0, 1.0)

        depth_errors = np.tan(angle_rad[:, None]) * self.depth_values_m[None, :]

        radial_edges = np.linspace(0.0, 1.0, self.radial_bins + 1, dtype=np.float64)
        radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
        angle_mean_by_radial = np.full(self.radial_bins, np.nan, dtype=np.float64)
        angle_p95_by_radial = np.full(self.radial_bins, np.nan, dtype=np.float64)
        depth_mean_by_radial = np.full((self.radial_bins, len(self.depth_values_m)), np.nan, dtype=np.float64)
        depth_p95_by_radial = np.full((self.radial_bins, len(self.depth_values_m)), np.nan, dtype=np.float64)

        for bdx in range(self.radial_bins):
            left = radial_edges[bdx]
            right = radial_edges[bdx + 1]
            mask = (radial_norm >= left) & (
                radial_norm < right if bdx < self.radial_bins - 1 else radial_norm <= right
            )
            if not np.any(mask):
                continue
            angle_mean_by_radial[bdx] = np.mean(angle_deg[mask])
            angle_p95_by_radial[bdx] = np.percentile(angle_deg[mask], 95)
            depth_mean_by_radial[bdx] = np.mean(depth_errors[mask], axis=0)
            depth_p95_by_radial[bdx] = np.percentile(depth_errors[mask], 95, axis=0)

        sample_stats = {
            'samplekey': samplekey,
            'noise_percent': self.noise_percent,
            'alpha': self.alpha,
            'ray_grid_size': self.ray_grid_size,
            'radial_bins': self.radial_bins,
            'depth_values_m': self.depth_values_m.tolist(),
            'angle_error_deg': {
                'mean': float(np.mean(angle_deg)),
                'median': float(np.median(angle_deg)),
                'p95': float(np.percentile(angle_deg, 95)),
                'max': float(np.max(angle_deg)),
            },
            'depth_error_m': {
                'mean': np.mean(depth_errors, axis=0).tolist(),
                'p95': np.percentile(depth_errors, 95, axis=0).tolist(),
                'max': np.max(depth_errors, axis=0).tolist(),
            },
            'radial_profile': {
                'bin_edges': radial_edges.tolist(),
                'bin_centers': radial_centers.tolist(),
                'angle_mean_deg': angle_mean_by_radial.tolist(),
                'angle_p95_deg': angle_p95_by_radial.tolist(),
                'depth_mean_m': depth_mean_by_radial.tolist(),
                'depth_p95_m': depth_p95_by_radial.tolist(),
            },
        }
        self._camera_error_by_samplekey[samplekey] = sample_stats

    def camera_error_payload(self):
        by_sample = dict(self._camera_error_by_samplekey)
        samples = list(by_sample.values())
        if not samples:
            return {
                'noise_percent': self.noise_percent,
                'alpha': self.alpha,
                'depth_values_m': self.depth_values_m.tolist(),
                'by_sample': by_sample,
                'aggregate': {},
            }

        angle_mean = np.array([s['angle_error_deg']['mean'] for s in samples], dtype=np.float64)
        angle_p95 = np.array([s['angle_error_deg']['p95'] for s in samples], dtype=np.float64)
        angle_max = np.array([s['angle_error_deg']['max'] for s in samples], dtype=np.float64)
        depth_mean = np.array([s['depth_error_m']['mean'] for s in samples], dtype=np.float64)
        depth_p95 = np.array([s['depth_error_m']['p95'] for s in samples], dtype=np.float64)
        radial_angle_mean = np.array([s['radial_profile']['angle_mean_deg'] for s in samples], dtype=np.float64)
        radial_angle_p95 = np.array([s['radial_profile']['angle_p95_deg'] for s in samples], dtype=np.float64)
        radial_depth_mean = np.array([s['radial_profile']['depth_mean_m'] for s in samples], dtype=np.float64)
        radial_depth_p95 = np.array([s['radial_profile']['depth_p95_m'] for s in samples], dtype=np.float64)

        aggregate = {
            'num_samples': len(samples),
            'angle_error_deg': {
                'mean_of_means': self._nanmean(angle_mean),
                'mean_of_p95': self._nanmean(angle_p95),
                'max': self._nanmean(angle_max),
            },
            'depth_error_m': {
                'mean': np.nanmean(depth_mean, axis=0).tolist(),
                'p95': np.nanmean(depth_p95, axis=0).tolist(),
            },
            'radial_profile': {
                'bin_edges': samples[0]['radial_profile']['bin_edges'],
                'bin_centers': samples[0]['radial_profile']['bin_centers'],
                'angle_mean_deg': np.nanmean(radial_angle_mean, axis=0).tolist(),
                'angle_p95_deg': np.nanmean(radial_angle_p95, axis=0).tolist(),
                'depth_mean_m': np.nanmean(radial_depth_mean, axis=0).tolist(),
                'depth_p95_m': np.nanmean(radial_depth_p95, axis=0).tolist(),
            },
        }

        return {
            'noise_percent': self.noise_percent,
            'alpha': self.alpha,
            'depth_values_m': self.depth_values_m.tolist(),
            'by_sample': by_sample,
            'aggregate': aggregate,
        }

    def __getitem__(self, index):
        data = self.target_dataset[index]

        if data.get('extras', {}).get('dataset', None) != 'HOT3D':
            return data

        camera_params_gt = copy.deepcopy(data['camera_params'])
        theta_hw = self._camera_to_theta16(camera_params_gt)
        theta_sw = self._software_theta16()
        theta_final = self._interpolate(theta_hw, theta_sw)

        focal_length, principal_point, projection_params = self._theta16_to_camera_params(theta_final)
        camera_params_noisy = copy.deepcopy(camera_params_gt)
        camera_params_noisy['focal_length'] = focal_length
        camera_params_noisy['principal_point'] = principal_point
        camera_params_noisy['projection_params'] = projection_params
        camera_params_noisy['camera_type'] = 3
        data['camera_params'] = camera_params_noisy

        if self.get_camera:
            image = data['extras']['rgb_np']
            height, width = image.shape[:2]

            gt_f, gt_c, gt_proj = self._theta16_to_camera_params(theta_hw)
            gt_cam = OVR624CameraModel(gt_f, gt_c, gt_proj[3:], width, height)
            camera_model = OVR624CameraModel(
                focal_length,
                principal_point,
                projection_params[3:],
                width,
                height,
            )
            data['extras']['camera_model'] = camera_model
            samplekey = self._get_samplekey(data)
            if samplekey not in self._camera_error_by_samplekey:
                self._compute_ray_error_stats(samplekey, gt_cam, camera_model, width, height)

        return data


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Save noisy intrinsic predictions and camera-noise analysis for HOT3D.'
    )
    parser.add_argument('--no-cit', action='store_true', help='Disable CIT module (suffix: _no_cit).')
    parser.add_argument(
        '--ray-grid-size',
        type=int,
        default=21,
        help='Ray-grid side length for camera error analysis.',
    )
    parser.add_argument(
        '--radial-bins',
        type=int,
        default=10,
        help='Number of radial bins for camera error analysis.',
    )
    parser.add_argument(
        '--force-recompute',
        action='store_true',
        help='Force recomputation even if noisy prediction cache exists.',
    )
    parser.add_argument(
        '--noisy-predictions-dir',
        default=DEFAULT_NOISY_PREDICTIONS_DIR,
        help='Directory where noisy predictions and analysis artifacts are saved.',
    )
    return parser.parse_args(argv)


def percent_to_suffix_token(percent):
    p = float(percent)
    if p.is_integer():
        return str(int(p))
    text = f'{p:g}'
    return text.replace('.', '_')


def get_noise_run_cache_paths(output_dir, dataset_name, suffix):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{dataset_name}_{suffix}_predictions.pkl')
    camera_analysis_path = os.path.join(output_dir, f'{dataset_name}_{suffix}_camera_noise_analysis.pkl')
    return {
        'save_path': save_path,
        'camera_analysis_path': camera_analysis_path,
    }


def load_noise_run_cache(output_dir, dataset_name, suffix, require_camera_analysis=False):
    paths = get_noise_run_cache_paths(output_dir, dataset_name, suffix)
    save_path = paths['save_path']
    camera_analysis_path = paths['camera_analysis_path']
    if not os.path.exists(save_path):
        return None
    if require_camera_analysis and not os.path.exists(camera_analysis_path):
        return None

    analysis_payload = None
    if os.path.exists(camera_analysis_path):
        with open(camera_analysis_path, 'rb') as f:
            analysis_payload = pickle.load(f)

    return {
        'save_path': save_path,
        'camera_analysis_path': camera_analysis_path if analysis_payload is not None else None,
        'camera_analysis': analysis_payload,
        'from_cache': True,
    }


def _save_styled_png(fig, ax, output_path_no_ext):
    _blend_axes_to_paper(ax)
    _stroke_all_text(fig)
    _save_png_supersampled(fig, output_path_no_ext, scale=2)
    plt.close(fig)


def save_camera_noise_plots(analysis_payload, output_prefix):
    aggregate = analysis_payload.get('aggregate', {})
    if not aggregate:
        print('No camera-noise aggregate stats available for plotting.')
        return

    _enable_crisp_rendering()

    radial = aggregate['radial_profile']
    depth_values = np.asarray(analysis_payload['depth_values_m'], dtype=np.float64)

    centers = np.asarray(radial['bin_centers'], dtype=np.float64)
    angle_mean = np.asarray(radial['angle_mean_deg'], dtype=np.float64)
    angle_p95 = np.asarray(radial['angle_p95_deg'], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(centers, angle_mean, label='Mean angular error')
    ax.plot(centers, angle_p95, label='P95 angular error')
    ax.set_xlabel('Normalized radial distance from GT camera center')
    ax.set_ylabel('Angular error (deg)')
    ax.set_title(f"Ray Angular Error vs Radius ({analysis_payload['noise_percent']}%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_styled_png(fig, ax, f'{output_prefix}_angle_vs_radius')

    depth_mean = np.asarray(aggregate['depth_error_m']['mean'], dtype=np.float64)
    depth_p95 = np.asarray(aggregate['depth_error_m']['p95'], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depth_values, depth_mean, marker='o', label='Mean lateral error')
    ax.plot(depth_values, depth_p95, marker='o', label='P95 lateral error')
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel('Possible positional error (m)')
    ax.set_title(f"Depth-wise Possible Error ({analysis_payload['noise_percent']}%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_styled_png(fig, ax, f'{output_prefix}_depth_error')


def save_camera_noise_plots_all_runs(all_run_summaries, output_prefix):
    if not all_run_summaries:
        return

    valid = [s for s in all_run_summaries if s.get('aggregate')]
    if not valid:
        print('No combined camera-noise aggregate stats available for plotting.')
        return

    _enable_crisp_rendering()

    valid = sorted(valid, key=lambda x: x['noise_percent'])
    percents = np.array([s['noise_percent'] for s in valid], dtype=np.float64)
    mean_angles = np.array([s['aggregate']['angle_error_deg']['mean_of_means'] for s in valid], dtype=np.float64)
    p95_angles = np.array([s['aggregate']['angle_error_deg']['mean_of_p95'] for s in valid], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(percents, mean_angles, marker='o', label='Mean angular error')
    ax.plot(percents, p95_angles, marker='o', label='P95 angular error')
    ax.set_xlabel('Noise level (%)')
    ax.set_ylabel('Angular error (deg)')
    ax.set_title('Angular Error vs Noise Level')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_styled_png(fig, ax, f'{output_prefix}_angular_vs_noise')

    depth_values = np.asarray(valid[0]['depth_values_m'], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 5))
    for summary in valid:
        depth_mean = np.asarray(summary['aggregate']['depth_error_m']['mean'], dtype=np.float64)
        ax.plot(depth_values, depth_mean, marker='o', label=f"{summary['noise_percent']:g}%")
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel('Mean possible positional error (m)')
    ax.set_title('Depth Error vs Noise Level')
    ax.grid(True, alpha=0.3)
    ax.legend(title='Noise')
    fig.tight_layout()
    _save_styled_png(fig, ax, f'{output_prefix}_depth_error_vs_noise')

    radial_centers = np.asarray(valid[0]['aggregate']['radial_profile']['bin_centers'], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 5))
    for summary in valid:
        radial_mean = np.asarray(summary['aggregate']['radial_profile']['angle_mean_deg'], dtype=np.float64)
        ax.plot(radial_centers, radial_mean, marker='o', label=f"{summary['noise_percent']:g}%")
    ax.set_xlabel('Normalized radial distance from GT camera center')
    ax.set_ylabel('Mean angular error (deg)')
    ax.set_title('Angular Error vs Radius (All Noise Levels)')
    ax.grid(True, alpha=0.3)
    ax.legend(title='Noise')
    fig.tight_layout()
    _save_styled_png(fig, ax, f'{output_prefix}_radius_angle_all_noise')


def predict_two_hands(
    config,
    dataset_name,
    left_loader,
    right_loader,
    model,
    limb_model,
    device,
    output_dir,
    suffix='',
    camera_dataset=None,
    skip_if_exists=True,
):
    if skip_if_exists:
        cached = load_noise_run_cache(output_dir, dataset_name, suffix, require_camera_analysis=False)
        if cached is not None:
            print(f"Cache hit for {suffix}: {cached['save_path']}")
            return cached

    paths = get_noise_run_cache_paths(output_dir, dataset_name, suffix)
    save_path = paths['save_path']
    camera_analysis_path = paths['camera_analysis_path']

    model.eval()

    left_iterator = iter(left_loader)
    right_iterator = iter(right_loader)

    n_batches = len(left_iterator)

    output_data = collections.defaultdict(dict)
    progress_bar = tqdm(total=n_batches, desc='Validation', unit='frame')
    while True:
        with torch.no_grad():
            try:
                left_batch = next(left_iterator)
                right_batch = next(right_iterator)
            except StopIteration:
                break

            left_outs = infer(config, model, limb_model, left_batch, device)
            right_outs = infer(config, model, limb_model, right_batch, device)

            n_samples = len(left_outs['samplekeys'])

            for idx in range(n_samples):
                for htype, out in [['left', left_outs], ['right', right_outs]]:
                    samplekey = out['samplekeys'][idx]

                    output_data[samplekey][htype] = {
                        'pred_transl': out['pred_transl'][idx],
                        'hand': {
                            'visible': out['visible_hand'][idx],
                            'gt_vertices': out['gt_hand_vertices'][idx],
                            'gt_j3d': out['gt_hand_j3d'][idx],
                            'gt_j2d': out['gt_hand_j2d'][idx],
                            'pred_vertices': out['pred_hand_vertices'][idx],
                            'pred_j3d': out['pred_hand_j3d'][idx],
                            'pred_j2d': out['pred_hand_j2d'][idx],
                        },
                        'arm': {
                            'visible': out['visible_arm'][idx],
                            'gt_vertices': out['gt_arm_vertices'][idx],
                            'gt_j3d': out['gt_arm_j3d'][idx],
                            'gt_j2d': out['gt_arm_j2d'][idx],
                            'pred_vertices': out['pred_arm_vertices'][idx],
                            'pred_j3d': out['pred_arm_j3d'][idx],
                            'pred_j2d': out['pred_arm_j2d'][idx],
                        },
                    }

            progress_bar.update(1)

    progress_bar.close()

    print(f'Saving predictions to {save_path}')
    with open(save_path, 'wb') as f:
        pickle.dump(output_data, f)

    analysis_payload = None
    if camera_dataset is not None:
        analysis_payload = camera_dataset.camera_error_payload()
        with open(camera_analysis_path, 'wb') as f:
            pickle.dump(analysis_payload, f)
        print(f'Saving camera noise analysis to {camera_analysis_path}')

    return {
        'save_path': save_path,
        'camera_analysis_path': camera_analysis_path if analysis_payload is not None else None,
        'camera_analysis': analysis_payload,
    }


def main(argv=None):
    args = _parse_args(argv)

    output_dir = os.path.abspath(args.noisy_predictions_dir)

    os.makedirs(output_dir, exist_ok=True)

    dataset_name = 'HOT3D'
    cfg.DATASET.NAME = dataset_name

    suffix_base = 'undistort_inp_true'
    if args.no_cit:
        suffix_base += '_no_cit'
    undistort_inp = True
    anycalib = True
    noise_percentages = [0.5, 1, 2.5, 5, 7.5, 10, 25, 50, 75, 100, 150, 200]
    ray_grid_size = int(args.ray_grid_size)
    radial_bins = int(args.radial_bins)
    depth_values_m = [0.2, 0.5, 1.0, 1.5, 2.0]

    print('Base suffix', suffix_base)
    print('Noise percentages', noise_percentages)
    print(
        'Ray error analysis params',
        {
            'ray_grid_size': ray_grid_size,
            'radial_bins': radial_bins,
            'depth_values_m': depth_values_m,
            'force_recompute': bool(args.force_recompute),
            'noisy_predictions_dir': output_dir,
            'no_cit': bool(args.no_cit),
        },
    )

    dataset = HOT3DLoader(cfg.DATASET.HOT3D_ROOT, get_camera=True, split='val', config=cfg)

    software_intrinsics = None
    if anycalib:
        print(f'Inferring AnyCalib intrinsics from first frame (cam_id={ANYCALIB_CAM_ID}).')
        software_intrinsics = infer_anycalib_intrinsics_first_frame(cfg, dataset, device)
        payload, anycalib_json_path = save_anycalib_intrinsics(
            output_dir,
            dataset_name,
            software_intrinsics,
        )
        print(f"Using AnyCalib first-frame intrinsics: {payload['intrinsics']}")
        print(f'Saving AnyCalib intrinsics JSON to {anycalib_json_path}')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model = HALOAblations(cfg, use_cit=not args.no_cit)

    print('Loading ', cfg.POSE_3D.CHECKPOINT_PATH)
    model.load_state_dict(torch.load(cfg.POSE_3D.CHECKPOINT_PATH, map_location=device), strict=True)
    model.eval()
    model.cuda()

    batch_size = 32
    n_workers = 16
    prefetch_factor = 2
    persistent_workers = n_workers > 0
    pin_memory = True

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    if anycalib and n_workers > 0:
        print('Camera-noise analysis will be merged across workers via shared manager dict.')

    all_run_summaries = []
    force_recompute = bool(args.force_recompute)

    for noise_percent in noise_percentages:
        suffix = f"{suffix_base}_noise_{percent_to_suffix_token(noise_percent)}_percent"
        print(f'Running noise level: {noise_percent}% | suffix: {suffix}')

        run_out = None
        if not force_recompute:
            run_out = load_noise_run_cache(
                output_dir,
                dataset_name,
                suffix,
                require_camera_analysis=anycalib,
            )
            if run_out is not None:
                print(f'Cache hit for noise {noise_percent}%: skipping dataloader/model inference.')

        if run_out is None:
            run_dataset = dataset
            camera_error_manager = None
            camera_error_store = None
            if anycalib:
                if n_workers > 0:
                    camera_error_manager = mp.Manager()
                    camera_error_store = camera_error_manager.dict()
                run_dataset = InterpolatedNoisyAnyCalib624Dataset(
                    dataset,
                    software_intrinsics=software_intrinsics,
                    noise_percent=noise_percent,
                    ray_grid_size=ray_grid_size,
                    radial_bins=radial_bins,
                    depth_values_m=depth_values_m,
                    camera_error_store=camera_error_store,
                )

            left_loader = torch.utils.data.DataLoader(
                Arm3DDataset(
                    cfg,
                    run_dataset,
                    undistort_inp=undistort_inp,
                    return_complete_image=True,
                    hand_type='left',
                ),
                batch_size=batch_size,
                shuffle=False,
                num_workers=n_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
                drop_last=False,
            )
            right_loader = torch.utils.data.DataLoader(
                Arm3DDataset(
                    cfg,
                    run_dataset,
                    undistort_inp=undistort_inp,
                    return_complete_image=False,
                    hand_type='right',
                ),
                batch_size=batch_size,
                shuffle=False,
                num_workers=n_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
                drop_last=False,
            )
            limb_model = LimbModel(cfg, device=device, use_pose_pca=False, n_components=5)

            model.eval()
            run_out = predict_two_hands(
                cfg,
                dataset_name,
                left_loader,
                right_loader,
                model,
                limb_model,
                device,
                output_dir=output_dir,
                suffix=suffix,
                camera_dataset=run_dataset if anycalib else None,
                skip_if_exists=False,
            )

            if camera_error_manager is not None:
                camera_error_manager.shutdown()

            del left_loader
            del right_loader
            del limb_model

            torch.cuda.empty_cache()
            gc.collect()

        if run_out is not None and run_out.get('camera_analysis') is not None:
            all_run_summaries.append(run_out['camera_analysis'])
            plot_prefix = run_out['camera_analysis_path'].rsplit('.pkl', 1)[0]
            save_camera_noise_plots(run_out['camera_analysis'], plot_prefix)

    combined_prefix = os.path.join(output_dir, f'{dataset_name}_{suffix_base}_all_noise_levels')
    combined_pkl = f'{combined_prefix}_camera_noise_analysis.pkl'
    with open(combined_pkl, 'wb') as f:
        pickle.dump(all_run_summaries, f)
    print(f'Saving combined camera noise analysis to {combined_pkl}')
    save_camera_noise_plots_all_runs(all_run_summaries, combined_prefix)

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    main()
