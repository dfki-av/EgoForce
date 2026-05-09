import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import traceback  
import argparse
import pickle
import numpy as np  
from PIL import Image, ImageDraw
from tqdm import tqdm
from core import KalmanFilterCV3D
from settings import config as cfg
from models import LimbModel

from datasets import Arm3DDataset, HOT3DLoader, ArcticLoader, H2OLoader, HO3DV2Loader
from utils.evaluation_protocols import (
    evaluate_batch_hand_arm as evaluate_batch,
    evaluate_batch_hand as evaluate_batch_hand_only,
    evaluate_batch_two_hand,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


HOT3D_DATASET_NAMES = {
    'HOT3D',
    'HOT3D_PER',
    'HOT3D_PINHOLE',
    'HOT3D_EQUISOLID',
    'HOT3D_EQUIRECTANGULAR',
    'HOT3D_STEREOGRAPHIC',
}

HOT3D_MASK_DATASET_NAMES = {
    'HOT3D',
    'HOT3D_PER',
    'HOT3D_PINHOLE',
}

LEGACY_HOT3D_MODE_BY_DATASET = {
    'HOT3D': 'none',
    'HOT3D_PER': 'pinhole',
    'HOT3D_PINHOLE': 'pinhole',
    'HOT3D_EQUISOLID': 'equisolid',
    'HOT3D_EQUIRECTANGULAR': 'equirectangular',
    'HOT3D_STEREOGRAPHIC': 'stereographic',
}

HOT3D_SUFFIX_BY_MODE = {
    'none': 'undistort_inp_true',
    'pinhole': 'undistort_inp_false_pinhole',
    'equisolid': 'undistort_inp_false_equisolid',
    'equirectangular': 'undistort_inp_false_equirectangular',
    'stereographic': 'undistort_inp_false_stereographic',
}

PREDICTION_SUFFIX_FALLBACKS = tuple(
    dict.fromkeys(
        ['undistort_inp_true', 'undistort_inp_false'] + list(HOT3D_SUFFIX_BY_MODE.values())
    )
)

EVAL_LIMBS = ('hand', 'arm')

REQUIRED_LIMB_METRIC_KEYS = (
    'ACC_ERROR',
    'CS_MPJPE',
    'RR_MPJPE',
    'PA_MPJPE',
)

OPTIONAL_LIMB_METRIC_KEYS = (
    'INVISIBLE_CS_MPJPE',
    'INVISIBLE_RR_MPJPE',
    'INVISIBLE_PA_MPJPE',
    'INVISIBLE_ACC_ERROR',
    'OCC_CS_MPJPE',
    'OCC_RR_MPJPE',
    'OCC_PA_MPJPE',
    'VIS_CS_MPJPE',
    'VIS_RR_MPJPE',
    'VIS_PA_MPJPE',
    'OCC_CNT',
    'VIS_CNT',
    'FAILURE_RATE',
    'CS_MPJPE_WHEN_ARM_VISIBLE_GIVEN_HAND_VISIBLE',
    'RR_MPJPE_WHEN_ARM_VISIBLE_GIVEN_HAND_VISIBLE',
    'PA_MPJPE_WHEN_ARM_VISIBLE_GIVEN_HAND_VISIBLE',
    'ACC_ERROR_WHEN_ARM_VISIBLE_GIVEN_HAND_VISIBLE',
    'CS_MPJPE_WHEN_ARM_INVISIBLE_GIVEN_HAND_VISIBLE',
    'RR_MPJPE_WHEN_ARM_INVISIBLE_GIVEN_HAND_VISIBLE',
    'PA_MPJPE_WHEN_ARM_INVISIBLE_GIVEN_HAND_VISIBLE',
    'ACC_ERROR_WHEN_ARM_INVISIBLE_GIVEN_HAND_VISIBLE',
)

ALL_LIMB_METRIC_KEYS = REQUIRED_LIMB_METRIC_KEYS + OPTIONAL_LIMB_METRIC_KEYS

TWO_HAND_METRIC_KEYS = (
    'ACC_ERROR',
    'CS_MPJPE',
    'RR_MPJPE',
    'PA_MPJPE',
    'INVISIBLE_CS_MPJPE',
    'INVISIBLE_RR_MPJPE',
    'INVISIBLE_PA_MPJPE',
    'INVISIBLE_ACC_ERROR',
)


def _cfg_get(path, default):
    cur = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate EgoForce predictions from saved .pkl files.')
    default_predictions_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '_DATA', 'predictions')
    parser.add_argument(
        '--test-dataset-name',
        default=_cfg_get(['DATASET', 'TEST_NAME'], 'ARCTIC'),
        choices=[
            'ARCTIC',
            'H2O',
            'HO3D',
            'HOT3D',
            'HOT3D_PER',
            'HOT3D_PINHOLE',
            'HOT3D_EQUISOLID',
            'HOT3D_EQUIRECTANGULAR',
            'HOT3D_STEREOGRAPHIC',
        ],
        help='Dataset alias to evaluate.',
    )
    parser.add_argument('--log-path', default='', help='Optional subdirectory under --log-dir that contains prediction files.')
    parser.add_argument('--log-dir', default=default_predictions_dir, help='Directory that contains prediction files (default: _DATA).')
    parser.add_argument('--eval-split', default='val', help='Split for ARCTIC/HOT3D loaders.')

    parser.add_argument(
        '--hot3d-conversion',
        default='auto',
        choices=['auto', 'none', 'pinhole', 'equisolid', 'equirectangular', 'stereographic'],
        help='HOT3D conversion mode. "auto" infers from HOT3D_* dataset aliases.',
    )

    parser.add_argument('--no-undistort-inp', action='store_true', help='Use undistort_inp=False in Arm3DDataset.')
    parser.add_argument('--no-cit', action='store_true', help='Disable CIT module (suffix: _no_cit).')
    parser.add_argument('--no-arm-prior', action='store_true', help='Disable arm prior (suffix: _no_arm_prior).')
    parser.add_argument('--no-arm-input', action='store_true', help='Disable arm input (suffix: _no_arm_input).')
    parser.add_argument('--anycalib-624', action='store_true', help='Evaluate AnyCalib 624 suffix variant.')
    parser.add_argument('--anycalib-pin', action='store_true', help='Evaluate AnyCalib pinhole suffix variant.')
    parser.add_argument('--depth-model', action='store_true', help='Evaluate depth-model suffix variant.')
    parser.add_argument('--dgp-model', action='store_true', help='Evaluate DGP-model suffix variant.')
    parser.add_argument(
        '--disable-kalman-filter',
        action='store_true',
        help='Disable Kalman filtering for translation smoothing (enabled by default).',
    )

    parser.add_argument(
        '--ignore-failure-solves',
        dest='ignore_failure_solves',
        action='store_true',
        default=True,
        help='Ignore failed solves (HAND_CS_MPJPE > 1000mm) when averaging hand CS/ACC metrics. Enabled by default.',
    )
    parser.add_argument(
        '--no-ignore-failure-solves',
        dest='ignore_failure_solves',
        action='store_false',
        help='Disable filtering of failed solves in hand CS/ACC metric averaging.',
    )
    default_results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    parser.add_argument('--results-root', default=default_results_dir, help='Root directory for evaluation output artifacts.')

    return parser.parse_args(argv)


def _resolve_hot3d_mode(test_dataset_name, requested_mode):
    inferred_mode = LEGACY_HOT3D_MODE_BY_DATASET.get(test_dataset_name, 'none')
    mode = inferred_mode if requested_mode == 'auto' else requested_mode

    if test_dataset_name not in HOT3D_DATASET_NAMES and mode != 'none':
        raise ValueError(
            '--hot3d-conversion only applies to HOT3D datasets. '
            f'Got test dataset {test_dataset_name} with conversion mode {mode}.',
        )
    return mode


def _should_evaluate_arm(dataset_name):
    return dataset_name not in HOT3D_DATASET_NAMES and dataset_name != 'HO3D'


def _resolve_hot3d_image_mask_path(test_dataset_name):
    if test_dataset_name not in HOT3D_MASK_DATASET_NAMES:
        return None

    return os.path.abspath(_default_hot3d_mask_path())


def _resolve_hot3d_suffix(mode):
    if mode not in HOT3D_SUFFIX_BY_MODE:
        raise ValueError(f'Unknown HOT3D conversion mode: {mode}')

    suffix = HOT3D_SUFFIX_BY_MODE[mode]
    undistort_inp = mode == 'none'
    return suffix, undistort_inp


def _resolve_suffix_and_output_variant(args, hot3d_mode):
    suffix, undistort_inp = _resolve_hot3d_suffix(hot3d_mode)
    output_variant = 'OURS'

    if args.no_undistort_inp:
        suffix = 'undistort_inp_false'
        undistort_inp = False

    if args.no_cit:
        suffix += '_no_cit'

    if args.no_arm_prior:
        suffix += '_no_arm_prior'

    if args.no_arm_input:
        suffix += '_no_arm_input'

    if args.anycalib_624:
        suffix += '_anycalib_624'

    if args.anycalib_pin:
        suffix += '_anycalib_pin'

    if args.depth_model:
        suffix += '_depth_model'

    if args.dgp_model:
        suffix += '_DGP_model'

    return suffix, undistort_inp, output_variant


def _resolve_result_suffix(prediction_suffix, disable_kalman_filter, ignore_failure_solves):
    suffix = prediction_suffix
    if disable_kalman_filter:
        suffix += '_no_kalman'
    if not ignore_failure_solves:
        suffix += '_with_failure_solves'
    return suffix


def _iter_settings(prefix, value):
    if isinstance(value, dict):
        for key in sorted(value.keys()):
            child_prefix = f'{prefix}.{key}' if prefix else str(key)
            yield from _iter_settings(child_prefix, value[key])
        return

    yield prefix, value


def _print_runtime_settings(args, config, resolved_settings):
    all_settings = {
        'args': vars(args),
        'resolved': resolved_settings,
        'cfg': config,
    }

    print('=== Runtime Settings ===')
    for key, value in _iter_settings('', all_settings):
        print(f'{key}: {value}')
    print('=== End Runtime Settings ===')


def _resolve_side_result(sample_result, side):
    if isinstance(sample_result, dict) and side in sample_result and isinstance(sample_result[side], dict):
        return sample_result[side]
    return sample_result


def _get_prediction_field(sample_result, limb, field, default=None):
    if isinstance(sample_result, dict):
        limb_data = sample_result.get(limb)
        if isinstance(limb_data, dict) and field in limb_data:
            return limb_data[field]

        if field == 'visible':
            flat_key = f'visible_{limb}'
        elif field.startswith('gt_'):
            flat_key = f'gt_{limb}_{field[3:]}'
        elif field.startswith('pred_'):
            flat_key = f'pred_{limb}_{field[5:]}'
        else:
            flat_key = f'{limb}_{field}'

        if flat_key in sample_result:
            return sample_result[flat_key]

    return default


def _resolve_predictions_path(log_dir, log_path, dataset_name, suffix):
    base = log_dir if not log_path else os.path.join(log_dir, log_path)

    candidate_suffixes = [suffix]
    # Backward-compatible fallback names for prediction files.
    for alt_suffix in PREDICTION_SUFFIX_FALLBACKS:
        if alt_suffix not in candidate_suffixes:
            candidate_suffixes.append(alt_suffix)

    _data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_DATA')
    search_bases = []
    for path in [_data_dir, base]:
        if not path:
            continue
        abs_path = os.path.abspath(path)
        if abs_path not in search_bases:
            search_bases.append(abs_path)

    candidates = [
        os.path.join(search_base, f'{dataset_name}_{candidate_suffix}_predictions.pkl')
        for search_base in search_bases
        for candidate_suffix in candidate_suffixes
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        'Could not find saved predictions. Tried:\n' + '\n'.join(candidates)
    )


def get_outs(config, limb_model, side, batch, device, results, filter=None):
    data, meta = batch
    
    batch_samplekeys = meta['samplekey']

    outs = {
        'pred_hand_j3d': [],
        'pred_hand_j2d': [],
        'gt_hand_j3d': [],
        'gt_hand_j2d': [],
        'org_img_size': [],
        'gt_hand_type': [],
        'pred_hand_type': [],
        'pred_hand_vertices': [],
        'gt_hand_vertices': [],
        'pred_hand_mesh': [],
        'gt_hand_mesh': [],
        'visible_hand': [],
        'pred_arm_j3d': [],
        'pred_arm_j2d': [],
        'gt_arm_j3d': [],
        'gt_arm_j2d': [],
        'gt_arm_type': [],
        'pred_arm_type': [],
        'pred_arm_vertices': [],
        'gt_arm_vertices': [],
        'pred_arm_mesh': [],
        'gt_arm_mesh': [],
        'visible_arm': [],
        'valid_arm_j3d': [],
        'occluded_hand_jnt': [],
        'valid_hand_j3d': [],
        'pred_transl': [],
        'gt_transl': [],
    }

    N = data['hand_crop'].shape[0]
    for i in range(N):
        samplekey = ''.join(map(chr, batch_samplekeys[i].cpu().numpy())).rstrip("\x00")

        sample_result = results[samplekey]
        result = _resolve_side_result(sample_result, side)

        default_gt_hand_j3d = (
            data['hand_j3d'][i].cpu().numpy()
            if 'hand_j3d' in data
            else np.zeros((21, 3), dtype=np.float32)
        )
        default_gt_arm_j3d = (
            data['arm_j3d'][i].cpu().numpy()
            if 'arm_j3d' in data
            else np.zeros((3, 3), dtype=np.float32)
        )
        default_gt_hand_j2d = (
            data['hand_j2d'][i].cpu().numpy()
            if 'hand_j2d' in data
            else np.zeros((default_gt_hand_j3d.shape[0], 2), dtype=np.float32)
        )
        default_gt_arm_j2d = (
            data['arm_j2d'][i].cpu().numpy()
            if 'arm_j2d' in data
            else np.zeros((default_gt_arm_j3d.shape[0], 2), dtype=np.float32)
        )

        gt_hand_j3d_result = np.asarray(
            _get_prediction_field(result, 'hand', 'gt_j3d', default_gt_hand_j3d)
        )
        gt_arm_j3d = np.asarray(
            _get_prediction_field(result, 'arm', 'gt_j3d', default_gt_arm_j3d)
        )

        if config.DATASET.NAME == 'HO3D':
            gt_hand_j3d = default_gt_hand_j3d
        else:
            gt_hand_j3d = gt_hand_j3d_result

        pred_hand_j3d = np.asarray(
            _get_prediction_field(result, 'hand', 'pred_j3d', gt_hand_j3d)
        )
        pred_arm_j3d = np.asarray(
            _get_prediction_field(result, 'arm', 'pred_j3d', gt_arm_j3d)
        )

        gt_hand_vertices = np.asarray(
            _get_prediction_field(result, 'hand', 'gt_vertices', gt_hand_j3d)
        )
        pred_hand_vertices = np.asarray(
            _get_prediction_field(result, 'hand', 'pred_vertices', pred_hand_j3d)
        )
        gt_arm_vertices = np.asarray(
            _get_prediction_field(result, 'arm', 'gt_vertices', gt_arm_j3d)
        )
        pred_arm_vertices = np.asarray(
            _get_prediction_field(result, 'arm', 'pred_vertices', pred_arm_j3d)
        )

        gt_hand_j2d = np.asarray(
            _get_prediction_field(result, 'hand', 'gt_j2d', default_gt_hand_j2d)
        )
        pred_hand_j2d = np.asarray(
            _get_prediction_field(result, 'hand', 'pred_j2d', gt_hand_j2d)
        )
        gt_arm_j2d = np.asarray(
            _get_prediction_field(result, 'arm', 'gt_j2d', default_gt_arm_j2d)
        )
        pred_arm_j2d = np.asarray(
            _get_prediction_field(result, 'arm', 'pred_j2d', gt_arm_j2d)
        )

        if isinstance(result, dict) and 'pred_transl' in result:
            pred_transl = np.asarray(result['pred_transl'])
        else:
            pred_transl = np.asarray(pred_hand_j3d[0])

        outs['gt_hand_j3d'].append(gt_hand_j3d)

        outs['gt_transl'].append(gt_hand_j3d[0])
        outs['pred_transl'].append(pred_hand_j3d[0])

        visible_hand = data['visible_hand'][i].item()

        if filter is not None:
            if False:#not visible_hand:
                filter.reset_state()
            else:
                inp_pred_transl = torch.tensor(pred_transl, device=device)
                ft_pred_transl = filter.step(inp_pred_transl.squeeze(0), visible_hand).unsqueeze(0).cpu().numpy()
            
                pred_hand_vertices = (pred_hand_vertices - pred_transl) + ft_pred_transl 
                pred_hand_j3d = (pred_hand_j3d - pred_transl) + ft_pred_transl 
                pred_arm_vertices = (pred_arm_vertices - pred_transl) + ft_pred_transl 
                pred_arm_j3d = (pred_arm_j3d - pred_transl) + ft_pred_transl 

        outs['pred_hand_j3d'].append(np.nan_to_num(pred_hand_j3d, nan=0.0))
        outs['pred_hand_j2d'].append(np.nan_to_num(pred_hand_j2d, nan=0.0))
        outs['pred_hand_vertices'].append(pred_hand_vertices)
        outs['gt_hand_j2d'].append(gt_hand_j2d)
        outs['gt_hand_vertices'].append(gt_hand_vertices)
        outs['org_img_size'].append(meta['org_img_size'][i].cpu().numpy())

        outs['pred_arm_j3d'].append(pred_arm_j3d)
        outs['pred_arm_j2d'].append(pred_arm_j2d)
        outs['pred_arm_vertices'].append(pred_arm_vertices)
        outs['gt_arm_j3d'].append(gt_arm_j3d)
        outs['gt_arm_j2d'].append(gt_arm_j2d)
        outs['gt_arm_vertices'].append(gt_arm_vertices)

        hand_type = data['hand_type'][i].item()

        outs['visible_hand'].append(data['visible_hand'][i].item())
        outs['visible_arm'].append(data['visible_arm'][i].item())

        outs['valid_hand_j3d'].append(data['valid_hand_j3d'][i].item()) 
        outs['valid_arm_j3d'].append(data['valid_arm_j3d'][i].item())

        outs['occluded_hand_jnt'].append(data['occluded_hand_jnt'][i].cpu().numpy())

        outs['gt_hand_type'].append(hand_type)
        outs['pred_hand_type'].append(hand_type)

    for k, v in outs.items():
        outs[k] = np.array(v)

    return outs


HAND_RADIAL_BINS_PERCENT = [
    ('0_25', 0.0, 25.0),
    ('25_50', 25.0, 50.0),
    ('50_75', 50.0, 75.0),
    ('75_100', 75.0, 100.0),
]


def _safe_mean(values):
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))


_HOT3D_MASK_CIRCLE_CACHE = {}


def _default_hot3d_mask_path():
    return os.path.join('.', 'datasets', 'hot3d_image_mask.png')


def _load_mask_gray(mask_path):
    mask = np.asarray(Image.open(mask_path))

    if mask.ndim == 3:
        mask = mask[..., :3].mean(axis=2)
    return mask.astype(np.float32), None


def _get_hot3d_mask_circle(mask_path=None, valid_threshold=100):
    if mask_path is None:
        mask_path = _default_hot3d_mask_path()
    mask_path = os.path.abspath(mask_path)
    cache_key = (mask_path, float(valid_threshold))
    if cache_key in _HOT3D_MASK_CIRCLE_CACHE:
        return _HOT3D_MASK_CIRCLE_CACHE[cache_key]

    circle = {
        'mask_path': mask_path,
        'valid_threshold': float(valid_threshold),
        'error': None,
    }

    if not os.path.exists(mask_path):
        circle['error'] = f'Mask not found: {mask_path}'
        _HOT3D_MASK_CIRCLE_CACHE[cache_key] = circle
        return circle

    mask_gray, load_err = _load_mask_gray(mask_path)
    if mask_gray is None:
        circle['error'] = f'Mask load failed: {load_err}'
        _HOT3D_MASK_CIRCLE_CACHE[cache_key] = circle
        return circle

    valid_mask = mask_gray > float(valid_threshold)
    if not np.any(valid_mask):
        circle['error'] = (
            f'No valid pixels found in mask (threshold={valid_threshold}) for {mask_path}'
        )
        _HOT3D_MASK_CIRCLE_CACHE[cache_key] = circle
        return circle

    ys, xs = np.nonzero(valid_mask)
    center_x = float(np.mean(xs))
    center_y = float(np.mean(ys))
    # Treat mask as circular and infer radius from area.
    radius_px = float(np.sqrt(np.sum(valid_mask) / np.pi))
    if not np.isfinite(radius_px) or radius_px < 1e-8:
        circle['error'] = f'Invalid fitted radius from mask: {radius_px}'
        _HOT3D_MASK_CIRCLE_CACHE[cache_key] = circle
        return circle

    circle.update({
        'mask_gray': mask_gray,
        'valid_mask': valid_mask,
        'height': int(valid_mask.shape[0]),
        'width': int(valid_mask.shape[1]),
        'center_x': center_x,
        'center_y': center_y,
        'radius_px': radius_px,
    })
    _HOT3D_MASK_CIRCLE_CACHE[cache_key] = circle
    return circle


def _scale_mask_circle_to_image(circle, width, height):
    sx = float(width) / max(float(circle['width']), 1.0)
    sy = float(height) / max(float(circle['height']), 1.0)
    center = np.array(
        [float(circle['center_x']) * sx, float(circle['center_y']) * sy],
        dtype=np.float64,
    )
    radius = float(circle['radius_px']) * 0.5 * (sx + sy)
    return center, radius


def _save_hot3d_mask_debug_image(circle, out_path):
    if circle.get('error') is not None:
        return circle['error']

    base = np.clip(circle['mask_gray'], 0, 255).astype(np.uint8)
    if base.ndim == 2:
        rgb = np.stack([base, base, base], axis=-1)
    else:
        rgb = base[..., :3]

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)

    cx = float(circle['center_x'])
    cy = float(circle['center_y'])
    radius = float(circle['radius_px'])

    ring_pcts = [25, 50, 75, 90, 100]
    ring_colors = [
        (255, 80, 80),
        (255, 180, 0),
        (80, 220, 80),
        (80, 180, 255),
        (255, 255, 255),
    ]
    for pct, color in zip(ring_pcts, ring_colors):
        r = radius * (float(pct) / 100.0)
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(bbox, outline=color, width=3)
        draw.text((cx + r + 5, cy - 8), f"{pct}%", fill=color)

    draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 255, 255), outline=(0, 0, 0), width=1)
    image.save(out_path)
    return None


def compute_and_print_hot3d_mask_radial_stats(mask_path=None, valid_threshold=100, output_dir=None):
    if mask_path is None:
        text = 'HOT3D mask radial stats skipped (mask disabled for this dataset variant).'
        print(text)
        return text

    circle = _get_hot3d_mask_circle(mask_path=mask_path, valid_threshold=valid_threshold)
    mask_path = circle['mask_path']

    lines = []
    lines.append(
        f"HOT3D mask-circle radial stats by HAND_RADIAL_BINS_PERCENT (valid: mask > {valid_threshold}):"
    )
    lines.append(f"Mask path: {mask_path}")

    if circle.get('error') is not None:
        line = circle['error']
        print(line)
        lines.append(line)
        return '\n'.join(lines)

    valid_mask = circle['valid_mask']
    yy, xx = np.indices(valid_mask.shape, dtype=np.float64)
    radial_pct = 100.0 * np.sqrt(
        (xx - float(circle['center_x'])) ** 2 + (yy - float(circle['center_y'])) ** 2
    ) / max(float(circle['radius_px']), 1e-8)

    inside_circle = radial_pct <= 100.0
    total_px = int(np.sum(inside_circle))
    total_valid_px = int(np.sum(valid_mask & inside_circle))
    covered = np.zeros_like(valid_mask, dtype=bool)

    overall_line = (
        f"Fitted circle: center=({circle['center_x']:.2f},{circle['center_y']:.2f}), "
        f"radius={circle['radius_px']:.2f}px | "
        f"circle_px={total_px}, valid_px_in_circle={total_valid_px}, "
        f"valid_ratio_in_circle={100.0 * total_valid_px / max(total_px, 1):6.2f}%"
    )
    lines.append(overall_line)
    print(overall_line)

    for _bin_name, left, right in HAND_RADIAL_BINS_PERCENT:
        is_open_ended = float(right) >= 100.0
        if is_open_ended:
            # Requested behavior: for 75_100 include everything >=75, including unmasked/outside-circle pixels.
            bin_mask = radial_pct >= left
        else:
            bin_mask = inside_circle & (radial_pct >= left) & (radial_pct < right)

        covered |= bin_mask

        bin_total = int(np.sum(bin_mask))
        bin_valid = int(np.sum(valid_mask & bin_mask))
        bin_valid_ratio = 100.0 * bin_valid / max(bin_total, 1)
        if is_open_ended:
            valid_ref = int(np.sum(valid_mask))
            px_ref = int(valid_mask.size)
            bin_share_of_valid = 100.0 * bin_valid / max(valid_ref, 1)
            bin_share_of_ref = 100.0 * bin_total / max(px_ref, 1)
            ref_name = 'image'
        else:
            bin_share_of_valid = 100.0 * bin_valid / max(total_valid_px, 1)
            bin_share_of_ref = 100.0 * bin_total / max(total_px, 1)
            ref_name = 'fitted circle'

        line = (
            f"RadialMask {left:>4.0f}-{right:<4.0f}% | "
            f"bin_px={bin_total:8d} ({bin_share_of_ref:6.2f}% of {ref_name}) | "
            f"valid_px={bin_valid:8d} | "
            f"valid_in_bin={bin_valid_ratio:6.2f}% | "
            f"valid_share_of_mask={bin_share_of_valid:6.2f}%"
        )
        lines.append(line)
        print(line)

    has_open_ended_bin = any(float(right) >= 100.0 for _, _, right in HAND_RADIAL_BINS_PERCENT)
    outside_mask = (~covered) if has_open_ended_bin else (inside_circle & (~covered))
    outside_total = int(np.sum(outside_mask))
    outside_valid = int(np.sum(valid_mask & outside_mask))
    if has_open_ended_bin:
        outside_label = "outside_all_bins"
    else:
        outside_label = f"{HAND_RADIAL_BINS_PERCENT[-1][2]:.0f}-100% (not in HAND_RADIAL_BINS_PERCENT)"
    outside_line = (
        f"RadialMask {outside_label} | "
        f"bin_px={outside_total:8d}, valid_px={outside_valid:8d}, "
        f"valid_in_bin={100.0 * outside_valid / max(outside_total, 1):6.2f}%"
    )
    lines.append(outside_line)
    print(outside_line)

    if output_dir is None:
        output_dir = os.getcwd()
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    debug_img_path = os.path.join(output_dir, 'hot3d_mask_radial_debug.png')
    debug_err = _save_hot3d_mask_debug_image(circle, debug_img_path)
    if debug_err is None:
        debug_line = f"Saved debug image: {debug_img_path}"
    else:
        debug_line = f"Failed to save debug image: {debug_err}"
    lines.append(debug_line)
    print(debug_line)

    return '\n'.join(lines)


def _subset_outs(outs, mask):
    subset = {}
    for key, value in outs.items():
        if isinstance(value, np.ndarray) and value.shape[0] == mask.shape[0]:
            subset[key] = value[mask]
        else:
            subset[key] = value
    return subset


def _compute_hand_radial_percent(outs, hot3d_image_mask_path=None):
    hand_j2d = outs['gt_hand_j2d']
    org_img_size = outs['org_img_size']
    n_samples = hand_j2d.shape[0]
    radial_pct = np.full(n_samples, np.nan, dtype=np.float64)
    circle = None
    use_mask_circle = False
    if hot3d_image_mask_path is not None:
        circle = _get_hot3d_mask_circle(mask_path=hot3d_image_mask_path)
        use_mask_circle = circle.get('error') is None

    for i in range(n_samples):
        if org_img_size[i].shape[0] < 2:
            continue
        width = float(org_img_size[i][0])
        height = float(org_img_size[i][1])
        if width <= 1.0 or height <= 1.0:
            continue

        if use_mask_circle:
            center, max_radius = _scale_mask_circle_to_image(circle, width=width, height=height)
        else:
            center = np.array([width * 0.5, height * 0.5], dtype=np.float64)
            max_radius = np.linalg.norm(center)
        if max_radius < 1e-8:
            continue

        kpts = hand_j2d[i].astype(np.float64)
        valid = np.isfinite(kpts).all(axis=1) & (kpts[:, 0] >= 0) & (kpts[:, 1] >= 0)
        if not np.any(valid):
            continue

        dists = np.linalg.norm(kpts[valid] - center[None, :], axis=1)
        radial_pct[i] = 100.0 * np.mean(dists / max_radius)

    return radial_pct


def _new_hand_radial_store():
    return {
        bin_name: {
            'CS_MPJPE': [],
            'RR_MPJPE': [],
            'PA_MPJPE': [],
            'ACC_ERROR': [],
            'COUNT': 0,
        }
        for bin_name, _, _ in HAND_RADIAL_BINS_PERCENT
    }


def _accumulate_hand_radial_metrics(outs, hand_radial_store, hot3d_image_mask_path=None, ignore_failure_solves=True):
    radial_pct = _compute_hand_radial_percent(outs, hot3d_image_mask_path=hot3d_image_mask_path)
    for bin_name, left, right in HAND_RADIAL_BINS_PERCENT:
        if float(right) >= 100.0:
            # Requested behavior for 75_100: include all samples from 75% outward.
            mask = np.isfinite(radial_pct) & (radial_pct >= left)
        else:
            mask = np.isfinite(radial_pct) & (radial_pct >= left) & (radial_pct < right)
        n_selected = int(np.sum(mask))
        hand_radial_store[bin_name]['COUNT'] += n_selected
        if n_selected == 0:
            continue

        subset = _subset_outs(outs, mask)
        subset_metrics = evaluate_batch(subset, ignore_failure_solves=ignore_failure_solves)
        hand_metrics = subset_metrics['hand']

        hand_radial_store[bin_name]['CS_MPJPE'].append(hand_metrics['CS_MPJPE'])
        hand_radial_store[bin_name]['RR_MPJPE'].append(hand_metrics['RR_MPJPE'])
        hand_radial_store[bin_name]['PA_MPJPE'].append(hand_metrics['PA_MPJPE'])
        hand_radial_store[bin_name]['ACC_ERROR'].append(hand_metrics['ACC_ERROR'])


def compute_and_print_hand_radial_metrics(left_hand_radial, right_hand_radial):
    lines = []
    lines.append("Hand radial bins from pinhole-mask fitted circle (% of fitted radius):")
    for bin_name, left, right in HAND_RADIAL_BINS_PERCENT:
        left_bin = left_hand_radial[bin_name]
        right_bin = right_hand_radial[bin_name]

        left_cs = _safe_mean(left_bin['CS_MPJPE'])
        left_rr = _safe_mean(left_bin['RR_MPJPE'])
        left_pa = _safe_mean(left_bin['PA_MPJPE'])
        left_acc = _safe_mean(left_bin['ACC_ERROR'])

        right_cs = _safe_mean(right_bin['CS_MPJPE'])
        right_rr = _safe_mean(right_bin['RR_MPJPE'])
        right_pa = _safe_mean(right_bin['PA_MPJPE'])
        right_acc = _safe_mean(right_bin['ACC_ERROR'])

        mean_cs = _safe_mean([left_cs, right_cs])
        mean_rr = _safe_mean([left_rr, right_rr])
        mean_pa = _safe_mean([left_pa, right_pa])
        mean_acc = _safe_mean([left_acc, right_acc])

        range_text = f"{left:>4.0f}-{right:<4.0f}%"
        if float(right) >= 100.0:
            range_text = f">={left:>4.0f}%"

        line = (
            f"HandRadial {range_text} | "
            f"Left  n={left_bin['COUNT']:6d} (CS:{left_cs:6.2f}, RR:{left_rr:6.2f}, PA:{left_pa:6.2f}, ACC:{left_acc:6.2f}) | "
            f"Right n={right_bin['COUNT']:6d} (CS:{right_cs:6.2f}, RR:{right_rr:6.2f}, PA:{right_pa:6.2f}, ACC:{right_acc:6.2f}) | "
            f"Mean  (CS:{mean_cs:6.2f}, RR:{mean_rr:6.2f}, PA:{mean_pa:6.2f}, ACC:{mean_acc:6.2f})"
        )
        lines.append(line)
        print(line)

    return '\n'.join(lines)


def _mean_metric(results_by_limb, limb, key):
    return float(np.mean(results_by_limb[limb][key]))


def _sum_metric(results_by_limb, limb, key):
    return float(np.sum(results_by_limb[limb][key]))


def _summarize_side_metrics(side_results, limb):
    return {
        'acc_error': _mean_metric(side_results, limb, 'ACC_ERROR'),
        'cs': _mean_metric(side_results, limb, 'CS_MPJPE'),
        'rr': _mean_metric(side_results, limb, 'RR_MPJPE'),
        'pa': _mean_metric(side_results, limb, 'PA_MPJPE'),
        'in_cs': _mean_metric(side_results, limb, 'INVISIBLE_CS_MPJPE'),
        'in_rr': _mean_metric(side_results, limb, 'INVISIBLE_RR_MPJPE'),
        'in_pa': _mean_metric(side_results, limb, 'INVISIBLE_PA_MPJPE'),
        'in_acc': _mean_metric(side_results, limb, 'INVISIBLE_ACC_ERROR'),
        'occ_cs': _mean_metric(side_results, limb, 'OCC_CS_MPJPE'),
        'occ_rr': _mean_metric(side_results, limb, 'OCC_RR_MPJPE'),
        'occ_pa': _mean_metric(side_results, limb, 'OCC_PA_MPJPE'),
        'vis_cs': _mean_metric(side_results, limb, 'VIS_CS_MPJPE'),
        'vis_rr': _mean_metric(side_results, limb, 'VIS_RR_MPJPE'),
        'vis_pa': _mean_metric(side_results, limb, 'VIS_PA_MPJPE'),
        'occ_cnt': _sum_metric(side_results, limb, 'OCC_CNT'),
        'vis_cnt': _sum_metric(side_results, limb, 'VIS_CNT'),
        'fail_rate': _mean_metric(side_results, limb, 'FAILURE_RATE'),
        'cs_hv_av': _mean_metric(side_results, limb, 'CS_MPJPE_WHEN_ARM_VISIBLE_GIVEN_HAND_VISIBLE'),
        'rr_hv_av': _mean_metric(side_results, limb, 'RR_MPJPE_WHEN_ARM_VISIBLE_GIVEN_HAND_VISIBLE'),
        'pa_hv_av': _mean_metric(side_results, limb, 'PA_MPJPE_WHEN_ARM_VISIBLE_GIVEN_HAND_VISIBLE'),
        'acc_hv_av': _mean_metric(side_results, limb, 'ACC_ERROR_WHEN_ARM_VISIBLE_GIVEN_HAND_VISIBLE'),
        'cs_hv_ai': _mean_metric(side_results, limb, 'CS_MPJPE_WHEN_ARM_INVISIBLE_GIVEN_HAND_VISIBLE'),
        'rr_hv_ai': _mean_metric(side_results, limb, 'RR_MPJPE_WHEN_ARM_INVISIBLE_GIVEN_HAND_VISIBLE'),
        'pa_hv_ai': _mean_metric(side_results, limb, 'PA_MPJPE_WHEN_ARM_INVISIBLE_GIVEN_HAND_VISIBLE'),
        'acc_hv_ai': _mean_metric(side_results, limb, 'ACC_ERROR_WHEN_ARM_INVISIBLE_GIVEN_HAND_VISIBLE'),
    }


def compute_and_print_metrics(left_results, right_results, limbs=None):
    if limbs is None:
        limbs = tuple(left_results.keys())

    texts = []
    for limb in limbs:
        left = _summarize_side_metrics(left_results, limb)
        right = _summarize_side_metrics(right_results, limb)

        mean_acc_error = (left['acc_error'] + right['acc_error']) / 2.0
        mean_in_acc_error = (left['in_acc'] + right['in_acc']) / 2.0

        mean_cs = (left['cs'] + right['cs']) / 2.0
        mean_rr = (left['rr'] + right['rr']) / 2.0
        mean_pa = (left['pa'] + right['pa']) / 2.0

        mean_in_cs = (left['in_cs'] + right['in_cs']) / 2.0
        mean_in_rr = (left['in_rr'] + right['in_rr']) / 2.0
        mean_in_pa = (left['in_pa'] + right['in_pa']) / 2.0

        mean_occ_cs = (left['occ_cs'] + right['occ_cs']) / 2.0
        mean_occ_rr = (left['occ_rr'] + right['occ_rr']) / 2.0
        mean_occ_pa = (left['occ_pa'] + right['occ_pa']) / 2.0

        mean_vis_cs = (left['vis_cs'] + right['vis_cs']) / 2.0
        mean_vis_rr = (left['vis_rr'] + right['vis_rr']) / 2.0
        mean_vis_pa = (left['vis_pa'] + right['vis_pa']) / 2.0

        mean_fail_rate = (left['fail_rate'] + right['fail_rate']) / 2.0

        total_occ_cnt = left['occ_cnt'] + right['occ_cnt']
        total_vis_cnt = left['vis_cnt'] + right['vis_cnt']

        mean_cs_hv_av = (left['cs_hv_av'] + right['cs_hv_av']) / 2.0
        mean_rr_hv_av = (left['rr_hv_av'] + right['rr_hv_av']) / 2.0
        mean_pa_hv_av = (left['pa_hv_av'] + right['pa_hv_av']) / 2.0
        mean_acc_hv_av = (left['acc_hv_av'] + right['acc_hv_av']) / 2.0

        mean_cs_hv_ai = (left['cs_hv_ai'] + right['cs_hv_ai']) / 2.0
        mean_rr_hv_ai = (left['rr_hv_ai'] + right['rr_hv_ai']) / 2.0
        mean_pa_hv_ai = (left['pa_hv_ai'] + right['pa_hv_ai']) / 2.0
        mean_acc_hv_ai = (left['acc_hv_ai'] + right['acc_hv_ai']) / 2.0

        text = (
            f"{limb.capitalize():<5} | "
            f"Left  (CS:{left['cs']:6.2f}, RR:{left['rr']:6.2f}, PA:{left['pa']:6.2f}) | "
            f"Right (CS:{right['cs']:6.2f}, RR:{right['rr']:6.2f}, PA:{right['pa']:6.2f}) | "
            f"Mean  (CS:{mean_cs:6.2f}, RR:{mean_rr:6.2f}, PA:{mean_pa:6.2f}) | "
            f"Mean  ACC:{mean_acc_error:6.2f} | "
            f"Left Invisible (CS:{left['in_cs']:6.2f}, RR:{left['in_rr']:6.2f}, PA:{left['in_pa']:6.2f}) | "
            f"Right Invisible (CS:{right['in_cs']:6.2f}, RR:{right['in_rr']:6.2f}, PA:{right['in_pa']:6.2f}) | "
            f"Mean Invisible (CS:{mean_in_cs:6.2f}, RR:{mean_in_rr:6.2f}, PA:{mean_in_pa:6.2f}) | "
            f"Mean Invisible ACC:{mean_in_acc_error:6.2f} | "
        )

        if limb != 'arm':
            text += (
                f"Mean Occluded (CS:{mean_occ_cs:6.2f}, RR:{mean_occ_rr:6.2f}, PA:{mean_occ_pa:6.2f}) | "
                f"Mean Vis Joints (CS:{mean_vis_cs:6.2f}, RR:{mean_vis_rr:6.2f}, PA:{mean_vis_pa:6.2f}) | "
                f"Total Joints (VIS:{int(total_vis_cnt)}, OCC:{int(total_occ_cnt)}) | "
                f"Left Fail Rate:{left['fail_rate']:5.2f}% | Right Fail Rate:{right['fail_rate']:5.2f}% | Mean Fail Rate:{mean_fail_rate:5.2f}% | "
                f"Given Hand Visible -> Arm Vis (CS:{mean_cs_hv_av:6.2f}, RR:{mean_rr_hv_av:6.2f}, PA:{mean_pa_hv_av:6.2f}, ACC:{mean_acc_hv_av:6.2f}) | "
                f"Arm Invis (CS:{mean_cs_hv_ai:6.2f}, RR:{mean_rr_hv_ai:6.2f}, PA:{mean_pa_hv_ai:6.2f}, ACC:{mean_acc_hv_ai:6.2f}) | "
            )

        texts.append(text)
        print(text)

    return '\n'.join(texts)


def compute_and_print_metrics_two_hand(two_hand_results):
    """
    Prints a single summary line for the two-hand (relative) evaluation.
    """
    cs = np.mean(two_hand_results['hand']['CS_MPJPE'])
    acc = np.mean(two_hand_results['hand']['ACC_ERROR'])

    text = (
        f"TwoHand | "
        f"Relative CS:{cs:6.2f} | "
        f"ACC:{acc:6.2f}"
    )
    print(text)
    return text


def _new_limb_metrics_store():
    return {key: [] for key in ALL_LIMB_METRIC_KEYS}


def _new_eval_results_store(limbs=None):
    if limbs is None:
        limbs = EVAL_LIMBS
    return {limb: _new_limb_metrics_store() for limb in limbs}


def _new_two_hand_results_store():
    return {'hand': {key: [] for key in TWO_HAND_METRIC_KEYS}}


def _append_limb_batch_metrics(aggregate_results, batch_results, limbs):
    for limb in limbs:
        limb_batch = batch_results[limb]
        for key in REQUIRED_LIMB_METRIC_KEYS:
            aggregate_results[limb][key].append(limb_batch[key])
        for key in OPTIONAL_LIMB_METRIC_KEYS:
            aggregate_results[limb][key].append(limb_batch.get(key, 0))


def _append_two_hand_batch_metrics(two_hand_results, two_hand_batch):
    for key in TWO_HAND_METRIC_KEYS:
        two_hand_results['hand'][key].append(two_hand_batch['hand'][key])


def evaluate_model(
    config,
    left_loader,
    right_loader,
    limb_model,
    results_dir,
    device,
    results,
    result_suffix,
    evaluate_arm=True,
    hot3d_image_mask_path=None,
    ignore_failure_solves=True,
    apply_kalman_filter=True,
):
    model_folder = cfg.DATASET.NAME
    save_path = os.path.join(results_dir, f'{model_folder}_{result_suffix}_evaluation_results.txt')

    left_iterator = iter(left_loader)
    right_iterator = iter(right_loader)

    eval_limbs = EVAL_LIMBS if evaluate_arm else ('hand',)

    left_results = _new_eval_results_store(eval_limbs)
    right_results = _new_eval_results_store(eval_limbs)
    left_hand_radial = _new_hand_radial_store()
    right_hand_radial = _new_hand_radial_store()
    two_hand_results = _new_two_hand_results_store()

    if apply_kalman_filter:
        freq = 30.0
        left_filter = KalmanFilterCV3D(q_pos=0.001, q_vel=1e-05, r_meas=0.001, freq=freq).to(device)
        right_filter = KalmanFilterCV3D(q_pos=0.001, q_vel=1e-05, r_meas=0.001, freq=freq).to(device)
    else:
        left_filter = None
        right_filter = None


    progress_bar = tqdm(total=len(left_loader), desc='evaluating', unit='frame')
    current_batch = 0
    try:
        while True:
            with torch.no_grad():
                try:
                    left_batch = next(left_iterator)
                    right_batch = next(right_iterator)
                except StopIteration:
                    break

                current_batch += 1

                try:
                    left_outs = get_outs(config, limb_model, 'left', left_batch, device, results, filter=left_filter)
                    right_outs = get_outs(config, limb_model, 'right', right_batch, device, results, filter=right_filter)

                    _accumulate_hand_radial_metrics(
                        left_outs,
                        left_hand_radial,
                        hot3d_image_mask_path=hot3d_image_mask_path,
                        ignore_failure_solves=ignore_failure_solves,
                    )
                    _accumulate_hand_radial_metrics(
                        right_outs,
                        right_hand_radial,
                        hot3d_image_mask_path=hot3d_image_mask_path,
                        ignore_failure_solves=ignore_failure_solves,
                    )

                    if evaluate_arm:
                        left_batch_results = evaluate_batch(left_outs, ignore_failure_solves=ignore_failure_solves)
                        right_batch_results = evaluate_batch(right_outs, ignore_failure_solves=ignore_failure_solves)
                    else:
                        left_batch_results = evaluate_batch_hand_only(left_outs, ignore_failure_solves=ignore_failure_solves)
                        right_batch_results = evaluate_batch_hand_only(right_outs, ignore_failure_solves=ignore_failure_solves)
                    two_hand_batch_results = evaluate_batch_two_hand(left_outs, right_outs)
                except Exception as exc:
                    print(f"Error evaluating batch {current_batch}: {exc}")
                    traceback.print_exc()
                    continue

            _append_limb_batch_metrics(left_results, left_batch_results, eval_limbs)
            _append_limb_batch_metrics(right_results, right_batch_results, eval_limbs)
            _append_two_hand_batch_metrics(two_hand_results, two_hand_batch_results)

            progress_bar.update(1)

            if current_batch % 10 == 0:
                compute_and_print_metrics(left_results, right_results, eval_limbs)
                compute_and_print_metrics_two_hand(two_hand_results)
    except KeyboardInterrupt:
        print('Stopping evaluation...')
    finally:
        progress_bar.close()

    text_single = compute_and_print_metrics(left_results, right_results, eval_limbs)
    text_two = compute_and_print_metrics_two_hand(two_hand_results)
    text_radial = compute_and_print_hand_radial_metrics(left_hand_radial, right_hand_radial)
    text_hot3d_mask = compute_and_print_hot3d_mask_radial_stats(
        mask_path=hot3d_image_mask_path,
        output_dir=results_dir,
    )
    text_str = text_single + "\n" + text_two + "\n\n" + text_radial + "\n\n" + text_hot3d_mask

    print("\nFinal Evaluation Results:\n")
    print('-' * 80)
    print(text_str)

    with open(save_path, 'w') as f:
        f.write(text_str)


def _build_eval_dataset(config, test_dataset_name, split, hot3d_mode):
    if test_dataset_name == 'ARCTIC':
        return ArcticLoader(config.DATASET.ARCTIC_ROOT, get_camera=True, split=split, config=config), 'ARCTIC'
    if test_dataset_name == 'H2O':
        return H2OLoader(config.DATASET.H2O_ROOT, get_camera=True, split='test', config=config), 'H2O'
    if test_dataset_name == 'HO3D':
        return HO3DV2Loader(config.DATASET.HO3D_ROOT, get_camera=True, split='test', config=config), 'HO3D'
    if test_dataset_name in HOT3D_DATASET_NAMES:
        return (
            HOT3DLoader(
                config.DATASET.HOT3D_ROOT,
                get_camera=True,
                split=split,
                config=config,
                conversion_mode=hot3d_mode,
            ),
            'HOT3D',
        )

    raise NotImplementedError(f'Unsupported dataset: {test_dataset_name}')


def _build_eval_loader(config, dataset, hand_type, undistort_inp):
    batch_size = 16
    n_workers = 16
    prefetch_factor = 2 if n_workers > 0 else None
    persistent_workers = False

    return torch.utils.data.DataLoader(
        Arm3DDataset(
            config,
            dataset,
            undistort_inp=undistort_inp,
            return_complete_image=False,
            hand_type=hand_type,
            no_arm=False,
            no_kpe=False,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


def _load_pickle_results(results_path):
    print(f'Loading results from: {results_path}')
    with open(results_path, 'rb') as f:
        return pickle.load(f)


def get_evaluation(args):
    cfg.DATASET.NAME = args.test_dataset_name
    hot3d_image_mask_path = _resolve_hot3d_image_mask_path(args.test_dataset_name)

    hot3d_mode = _resolve_hot3d_mode(args.test_dataset_name, args.hot3d_conversion)
    prediction_suffix, undistort_inp, output_variant = _resolve_suffix_and_output_variant(args, hot3d_mode)
    result_suffix = _resolve_result_suffix(
        prediction_suffix,
        args.disable_kalman_filter,
        args.ignore_failure_solves,
    )

    dataset, resolved_dataset_name = _build_eval_dataset(
        cfg,
        test_dataset_name=args.test_dataset_name,
        split=args.eval_split,
        hot3d_mode=hot3d_mode,
    )
    cfg.DATASET.NAME = resolved_dataset_name
    evaluate_arm = _should_evaluate_arm(args.test_dataset_name)
    if not evaluate_arm:
        print('HOT3D variant detected: skipping arm evaluation metrics.')

    results_path = _resolve_predictions_path(args.log_dir, args.log_path, resolved_dataset_name, prediction_suffix)
    results = _load_pickle_results(results_path)

    left_loader = _build_eval_loader(
        cfg,
        dataset,
        hand_type='left',
        undistort_inp=undistort_inp,
    )
    right_loader = _build_eval_loader(
        cfg,
        dataset,
        hand_type='right',
        undistort_inp=undistort_inp,
    )

    results_dir = os.path.join(args.results_root, output_variant)
    os.makedirs(results_dir, exist_ok=True)

    limb_model = LimbModel(cfg, device=device, use_pose_pca=False, n_components=5)
    evaluate_model(
        cfg,
        left_loader,
        right_loader,
        limb_model,
        results_dir,
        device,
        results,
        result_suffix,
        evaluate_arm=evaluate_arm,
        hot3d_image_mask_path=hot3d_image_mask_path,
        ignore_failure_solves=args.ignore_failure_solves,
        apply_kalman_filter=not args.disable_kalman_filter,
    )


def main(argv=None):
    args = _parse_args(argv)
    hot3d_image_mask_path = _resolve_hot3d_image_mask_path(args.test_dataset_name)

    hot3d_mode = _resolve_hot3d_mode(args.test_dataset_name, args.hot3d_conversion)
    prediction_suffix, undistort_inp, output_variant = _resolve_suffix_and_output_variant(args, hot3d_mode)
    result_suffix = _resolve_result_suffix(
        prediction_suffix,
        args.disable_kalman_filter,
        args.ignore_failure_solves,
    )

    resolved_settings = {
        'device': str(device),
        'hot3d_mode': hot3d_mode,
        'prediction_suffix': prediction_suffix,
        'result_suffix': result_suffix,
        'undistort_inp': undistort_inp,
        'output_variant': output_variant,
        'evaluate_arm': _should_evaluate_arm(args.test_dataset_name),
        'ignore_failure_solves': args.ignore_failure_solves,
        'use_kalman_filter': not args.disable_kalman_filter,
        'results_dir': os.path.join(args.results_root, output_variant),
        'hot3d_image_mask_path': hot3d_image_mask_path,
    }
    _print_runtime_settings(args, cfg, resolved_settings)

    get_evaluation(args)
 
if __name__ == '__main__':
    main()
