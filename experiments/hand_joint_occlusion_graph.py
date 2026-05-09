import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import argparse
import json
import pickle
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from settings import config as cfg
from datasets import (
    Arm3DDataset,
    ArcticLoader,
)

from utils.plot_utils import (
        _enable_crisp_rendering,
        _stroke_all_text,
        _save_png_supersampled,
        _blend_axes_to_paper,
    )


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Compare with/without-forearm predictions by hand-joint visibility bins.'
    )

    default_predictions_dir = os.path.join(ROOT_DIR, '_DATA', 'predictions')
    parser.add_argument('--predictions-dir', default=default_predictions_dir, help='Directory that contains prediction files.')
    parser.add_argument('--log-path', default='', help='Optional subdirectory under --predictions-dir.')

    parser.add_argument('--results-root', default=os.path.join(ROOT_DIR, 'results'))
    parser.add_argument('--output-dir', default='', help='Optional explicit output directory.')

    return parser.parse_args(argv)



def _prediction_search_bases(args):
    bases = []
    base = args.predictions_dir if not args.log_path else os.path.join(args.predictions_dir, args.log_path)
    bases.append(base)

    unique = []
    for p in bases:
        abs_p = os.path.abspath(p)
        if abs_p not in unique:
            unique.append(abs_p)
    return unique


def _resolve_predictions_path(search_bases, dataset_name, suffix):
    candidate_suffixes = [suffix]

    candidates = [
        os.path.join(search_base, f'{dataset_name}_{candidate_suffix}_predictions.pkl')
        for search_base in search_bases
        for candidate_suffix in candidate_suffixes
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError('Could not find saved predictions. Tried:\n' + '\n'.join(candidates))


def _load_pickle_results(results_path):
    print(f'Loading predictions from: {results_path}')
    with open(results_path, 'rb') as f:
        return pickle.load(f)


def _build_eval_dataset(config):
    return ArcticLoader(config.DATASET.ARCTIC_ROOT, get_camera=True, split='val', config=config), 'ARCTIC'


def _build_eval_loader(config, dataset, hand_type, undistort_inp, args):
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


def _safe_call(fn, *args, **kwargs):
    if fn is None:
        return None
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _decode_samplekey(raw_samplekey):
    if isinstance(raw_samplekey, str):
        return raw_samplekey
    if torch.is_tensor(raw_samplekey):
        arr = raw_samplekey.cpu().numpy()
    else:
        arr = np.asarray(raw_samplekey)
    return ''.join(map(chr, arr)).rstrip('\x00')


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


def _extract_outs(config, side, batch, predictions):
    data, meta = batch
    batch_samplekeys = meta['samplekey']

    outs = {
        'pred_hand_j3d': [],
        'gt_hand_j3d': [],
        'pred_arm_j3d': [],
        'gt_arm_j3d': [],
        'visible_hand': [],
        'visible_arm': [],
        'valid_hand_j3d': [],
        'valid_arm_j3d': [],
        'occluded_hand_jnt': [],
    }

    n_samples = data['hand_crop'].shape[0]
    missing_samplekeys = []

    for i in range(n_samples):
        samplekey = _decode_samplekey(batch_samplekeys[i])

        sample_result = predictions.get(samplekey)
        if sample_result is None:
            missing_samplekeys.append(samplekey)
            continue

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

        gt_hand_j3d_result = np.asarray(
            _get_prediction_field(result, 'hand', 'gt_j3d', default_gt_hand_j3d)
        )
        if config.DATASET.NAME == 'HO3D':
            gt_hand_j3d = default_gt_hand_j3d
        else:
            gt_hand_j3d = gt_hand_j3d_result

        gt_arm_j3d = np.asarray(_get_prediction_field(result, 'arm', 'gt_j3d', default_gt_arm_j3d))
        pred_hand_j3d = np.asarray(_get_prediction_field(result, 'hand', 'pred_j3d', gt_hand_j3d))
        pred_arm_j3d = np.asarray(_get_prediction_field(result, 'arm', 'pred_j3d', gt_arm_j3d))

        outs['pred_hand_j3d'].append(pred_hand_j3d)
        outs['gt_hand_j3d'].append(gt_hand_j3d)
        outs['pred_arm_j3d'].append(pred_arm_j3d)
        outs['gt_arm_j3d'].append(gt_arm_j3d)

        outs['visible_hand'].append(data['visible_hand'][i].item())
        outs['visible_arm'].append(data['visible_arm'][i].item())
        outs['valid_hand_j3d'].append(data['valid_hand_j3d'][i].item())
        outs['valid_arm_j3d'].append(data['valid_arm_j3d'][i].item())
        outs['occluded_hand_jnt'].append(data['occluded_hand_jnt'][i].cpu().numpy())

    if missing_samplekeys:
        preview = ', '.join(missing_samplekeys[:5])
        raise KeyError(
            f'Missing {len(missing_samplekeys)} samplekeys in predictions for side={side}. '
            f'Examples: {preview}'
        )

    for key, value in outs.items():
        outs[key] = np.array(value)

    return outs


def evaluate_batch_occlusion_percentage(data):
    pred_hand_j3d = data['pred_hand_j3d'] * 1000
    pred_arm_j3d = data['pred_arm_j3d'] * 1000

    gt_hand_j3d = data['gt_hand_j3d'] * 1000
    gt_arm_j3d = data['gt_arm_j3d'] * 1000

    visible_hand = np.array(data['visible_hand'], dtype=bool)
    valid_hand = np.array(data['valid_hand_j3d'], dtype=bool)
    visible_hand = np.logical_and(visible_hand, valid_hand)

    visible_arm = np.array(data['visible_arm'], dtype=bool)
    valid_arm = np.array(data['valid_arm_j3d'], dtype=bool)
    visible_arm = np.logical_and(visible_arm, valid_arm)

    occluded_hand_jnt = np.array(data['occluded_hand_jnt'], dtype=bool)
    visible_hand_jnt = np.logical_not(occluded_hand_jnt)

    keep_mask = np.logical_and(visible_arm, visible_hand)
    if not np.any(keep_mask):
        return {
            'visible_pct': np.array([]),
            'cs_mpjpe': np.array([]),
            'rr_mpjpe': np.array([]),
            'visible_pct_acc': np.array([]),
            'cs_acc': np.array([]),
            'rr_acc': np.array([]),
        }

    pred_h = pred_hand_j3d[keep_mask]
    gt_h = gt_hand_j3d[keep_mask]
    vis_j = visible_hand_jnt[keep_mask]

    total_kp = vis_j.shape[1]
    num_visible = vis_j.sum(axis=1).astype(np.float32)
    visible_pct = (num_visible / float(total_kp)) * 100.0

    per_joint_err = np.linalg.norm(pred_h - gt_h, axis=-1)
    cs_mpjpe = per_joint_err.mean(axis=1)

    per_joint_rr_err = np.linalg.norm((pred_h - pred_h[:, :1, :]) - (gt_h - gt_h[:, :1, :]), axis=-1)
    rr_mpjpe = per_joint_rr_err.mean(axis=1)

    fps = 30.0
    batch_size = pred_hand_j3d.shape[0]
    if batch_size < 3:
        visible_pct_acc = np.array([])
        cs_acc = np.array([])
        rr_acc = np.array([])
    else:
        triplet_keep = keep_mask[:-2] & keep_mask[1:-1] & keep_mask[2:]

        cs_gt_acc = gt_hand_j3d[2:] - 2 * gt_hand_j3d[1:-1] + gt_hand_j3d[:-2]
        cs_pred_acc = pred_hand_j3d[2:] - 2 * pred_hand_j3d[1:-1] + pred_hand_j3d[:-2]

        gt_rr_full = gt_hand_j3d - gt_hand_j3d[:, 0:1, :]
        pred_rr_full = pred_hand_j3d - pred_hand_j3d[:, 0:1, :]
        rr_gt_acc = gt_rr_full[2:] - 2 * gt_rr_full[1:-1] + gt_rr_full[:-2]
        rr_pred_acc = pred_rr_full[2:] - 2 * pred_rr_full[1:-1] + pred_rr_full[:-2]

        scale = fps ** 2
        cs_gt_acc = cs_gt_acc * scale
        cs_pred_acc = cs_pred_acc * scale
        rr_gt_acc = rr_gt_acc * scale
        rr_pred_acc = rr_pred_acc * scale

        cs_acc_err = np.linalg.norm(cs_gt_acc - cs_pred_acc, axis=-1)
        rr_acc_err = np.linalg.norm(rr_gt_acc - rr_pred_acc, axis=-1)

        cs_acc_center = cs_acc_err.mean(axis=1)
        rr_acc_center = rr_acc_err.mean(axis=1)

        cs_acc = cs_acc_center[triplet_keep]
        rr_acc = rr_acc_center[triplet_keep]

        visible_pct_full = (visible_hand_jnt.sum(axis=1) / float(total_kp)) * 100.0
        visible_pct_acc = visible_pct_full[1:-1][triplet_keep]

        cs_acc = cs_acc * 1e-3
        rr_acc = rr_acc * 1e-3

    return {
        'visible_pct': visible_pct,
        'cs_mpjpe': cs_mpjpe,
        'rr_mpjpe': rr_mpjpe,
        'visible_pct_acc': visible_pct_acc,
        'cs_acc': cs_acc,
        'rr_acc': rr_acc,
    }


def binned_stats(x_pct, y, bins):
    idx = np.digitize(x_pct, bins, right=True)
    centers, means, stds, counts = [], [], [], []
    for b in range(1, len(bins)):
        mask = idx == b
        vals = y[mask]
        centers.append(0.5 * (bins[b - 1] + bins[b]))
        if np.any(mask):
            means.append(np.nanmean(vals))
            stds.append(np.nanstd(vals))
            counts.append(np.sum(mask))
        else:
            means.append(np.nan)
            stds.append(np.nan)
            counts.append(0)
    return np.array(centers), np.array(means), np.array(stds), np.array(counts)


def percent_improvement(old, new, lower_is_better=True):
    old = np.asarray(old, dtype=float)
    new = np.asarray(new, dtype=float)
    delta = old - new if lower_is_better else new - old
    with np.errstate(divide='ignore', invalid='ignore'):
        return delta / np.abs(old) * 100.0


def overall_reduction(old_array, new_array):
    if old_array.size == 0 or new_array.size == 0:
        return float('nan')
    mu_old = np.nanmean(old_array)
    mu_new = np.nanmean(new_array)
    return float(percent_improvement(mu_old, mu_new, lower_is_better=True))


def _concat_non_empty(parts):
    valid = [np.asarray(p) for p in parts if np.asarray(p).size > 0]
    if len(valid) == 0:
        return np.array([], dtype=np.float64)
    return np.concatenate(valid, axis=0)


def _collect_metrics(left_loader, right_loader, with_arm_predictions, without_arm_predictions):
    visible_pct = []
    with_arm_cs = []
    without_arm_cs = []
    with_arm_rr = []
    without_arm_rr = []

    visible_pct_acc = []
    with_arm_cs_acc = []
    without_arm_cs_acc = []
    with_arm_rr_acc = []
    without_arm_rr_acc = []

    left_iterator = iter(left_loader)
    right_iterator = iter(right_loader)

    progress_bar = tqdm(total=len(left_loader), desc='Evaluating occlusion bins', unit='batch')
    try:
        while True:
            try:
                left_batch = next(left_iterator)
                right_batch = next(right_iterator)
            except StopIteration:
                break

            left_out_without = _extract_outs(cfg, 'left', left_batch, without_arm_predictions)
            right_out_without = _extract_outs(cfg, 'right', right_batch, without_arm_predictions)

            left_out_with = _extract_outs(cfg, 'left', left_batch, with_arm_predictions)
            right_out_with = _extract_outs(cfg, 'right', right_batch, with_arm_predictions)

            left_without = evaluate_batch_occlusion_percentage(left_out_without)
            right_without = evaluate_batch_occlusion_percentage(right_out_without)
            left_with = evaluate_batch_occlusion_percentage(left_out_with)
            right_with = evaluate_batch_occlusion_percentage(right_out_with)

            visible_pct.append(np.concatenate([left_with['visible_pct'], right_with['visible_pct']], axis=0))
            visible_pct_acc.append(np.concatenate([left_with['visible_pct_acc'], right_with['visible_pct_acc']], axis=0))

            with_arm_cs.append(np.concatenate([left_with['cs_mpjpe'], right_with['cs_mpjpe']], axis=0))
            without_arm_cs.append(np.concatenate([left_without['cs_mpjpe'], right_without['cs_mpjpe']], axis=0))

            with_arm_rr.append(np.concatenate([left_with['rr_mpjpe'], right_with['rr_mpjpe']], axis=0))
            without_arm_rr.append(np.concatenate([left_without['rr_mpjpe'], right_without['rr_mpjpe']], axis=0))

            with_arm_cs_acc.append(np.concatenate([left_with['cs_acc'], right_with['cs_acc']], axis=0))
            without_arm_cs_acc.append(np.concatenate([left_without['cs_acc'], right_without['cs_acc']], axis=0))

            with_arm_rr_acc.append(np.concatenate([left_with['rr_acc'], right_with['rr_acc']], axis=0))
            without_arm_rr_acc.append(np.concatenate([left_without['rr_acc'], right_without['rr_acc']], axis=0))

            progress_bar.update(1)
    finally:
        progress_bar.close()

    return {
        'visible_pct': _concat_non_empty(visible_pct),
        'with_arm_cs_mpjpe': _concat_non_empty(with_arm_cs),
        'without_arm_cs_mpjpe': _concat_non_empty(without_arm_cs),
        'with_arm_rr_mpjpe': _concat_non_empty(with_arm_rr),
        'without_arm_rr_mpjpe': _concat_non_empty(without_arm_rr),
        'visible_pct_acc': _concat_non_empty(visible_pct_acc),
        'with_arm_cs_acc': _concat_non_empty(with_arm_cs_acc),
        'without_arm_cs_acc': _concat_non_empty(without_arm_cs_acc),
        'with_arm_rr_acc': _concat_non_empty(with_arm_rr_acc),
        'without_arm_rr_acc': _concat_non_empty(without_arm_rr_acc),
    }


def _safe_mean(arr):
    if arr.size == 0:
        return float('nan')
    return float(np.mean(arr))


def _build_summary(metrics):
    summary = {
        'with_arm_cs_mpjpe_mm': _safe_mean(metrics['with_arm_cs_mpjpe']),
        'without_arm_cs_mpjpe_mm': _safe_mean(metrics['without_arm_cs_mpjpe']),
        'with_arm_rr_mpjpe_mm': _safe_mean(metrics['with_arm_rr_mpjpe']),
        'without_arm_rr_mpjpe_mm': _safe_mean(metrics['without_arm_rr_mpjpe']),
        'with_arm_cs_acc_mps2': _safe_mean(metrics['with_arm_cs_acc']),
        'without_arm_cs_acc_mps2': _safe_mean(metrics['without_arm_cs_acc']),
        'with_arm_rr_acc_mps2': _safe_mean(metrics['with_arm_rr_acc']),
        'without_arm_rr_acc_mps2': _safe_mean(metrics['without_arm_rr_acc']),
        'cs_mpjpe_reduction_pct': overall_reduction(
            metrics['without_arm_cs_mpjpe'], metrics['with_arm_cs_mpjpe']
        ),
        'rr_mpjpe_reduction_pct': overall_reduction(
            metrics['without_arm_rr_mpjpe'], metrics['with_arm_rr_mpjpe']
        ),
        'cs_acc_reduction_pct': overall_reduction(
            metrics['without_arm_cs_acc'], metrics['with_arm_cs_acc']
        ),
        'rr_acc_reduction_pct': overall_reduction(
            metrics['without_arm_rr_acc'], metrics['with_arm_rr_acc']
        ),
        'count_visibility_mpjpe': int(metrics['visible_pct'].size),
        'count_visibility_acc': int(metrics['visible_pct_acc'].size),
    }
    return summary


def _print_summary(summary):
    print(
        "MPJPE, ACC uses frames where both hand and arm are visible and valid, ACC uses valid 3-frame windows, "
        f"and 'with arm' vs 'without arm' compares whether removing arm input helps overall."
    )
    print('--- MPJPE (mm) ---')
    print('Mean per-joint position error in millimeters; lower is better, with CS absolute and RR root-relative.')
    print(
        'CS with/without:',
        f"{summary['with_arm_cs_mpjpe_mm']:.4f}",
        f"{summary['without_arm_cs_mpjpe_mm']:.4f}",
    )
    print(
        'RR with/without:',
        f"{summary['with_arm_rr_mpjpe_mm']:.4f}",
        f"{summary['without_arm_rr_mpjpe_mm']:.4f}",
    )

    print('--- ACC (m/s^2) ---')
    print(
        'CS with/without:',
        f"{summary['with_arm_cs_acc_mps2']:.6f}",
        f"{summary['without_arm_cs_acc_mps2']:.6f}",
    )
    print(
        'RR with/without:',
        f"{summary['with_arm_rr_acc_mps2']:.6f}",
        f"{summary['without_arm_rr_acc_mps2']:.6f}",
    )

    print('--- Percent Reduction (%) ---')
    print('CS-MPJPE:', f"{summary['cs_mpjpe_reduction_pct']:.4f}")
    print('RR-MPJPE:', f"{summary['rr_mpjpe_reduction_pct']:.4f}")
    print('CS-ACC  :', f"{summary['cs_acc_reduction_pct']:.4f}")
    print('RR-ACC  :', f"{summary['rr_acc_reduction_pct']:.4f}")


def _default_output_dir(args, with_suffix, without_suffix):
    run_name = f"ARCTIC_{with_suffix}_vs_{without_suffix}"
    return os.path.join(args.results_root, 'hand_joint_occlusion_graph', run_name)


def _json_safe(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.floating):
        v = float(value)
        return None if not np.isfinite(v) else v
    if isinstance(value, np.integer):
        return int(value)
    return value


def _save_artifacts(output_dir, summary, metrics, args, with_path, without_path):
    os.makedirs(output_dir, exist_ok=True)

    summary_txt = os.path.join(output_dir, 'summary.txt')
    summary_json = os.path.join(output_dir, 'summary.json')
    raw_npz = os.path.join(output_dir, 'raw_metrics.npz')

    lines = [
        'Hand Joint Occlusion Graph Summary',
        '=' * 40,
        f'test_dataset_name: {args.test_dataset_name}',
        f'with_arm_suffix: {args.with_arm_suffix}',
        f'without_arm_suffix: {args.without_arm_suffix}',
        f'with_arm_predictions: {with_path}',
        f'without_arm_predictions: {without_path}',
        '',
    ]
    for key in sorted(summary.keys()):
        lines.append(f'{key}: {summary[key]}')

    with open(summary_txt, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    payload = {
        'args': vars(args),
        'summary': {k: _json_safe(v) for k, v in summary.items()},
        'with_arm_predictions': with_path,
        'without_arm_predictions': without_path,
    }
    with open(summary_json, 'w') as f:
        json.dump(payload, f, indent=2)

    np.savez_compressed(raw_npz, **metrics)

    print(f'Saved summary text: {summary_txt}')
    print(f'Saved summary json: {summary_json}')
    print(f'Saved raw metrics: {raw_npz}')


def plot_percent_improvement(metrics, save_png_path, save_pdf_path, bins=None):
    if bins is None:
        bins = np.array([25, 35, 45, 55, 65, 75, 85, 95, 100], dtype=float)

    visible_pct = metrics['visible_pct']
    with_arm_mpjpes = metrics['with_arm_cs_mpjpe']
    without_arm_mpjpes = metrics['without_arm_cs_mpjpe']
    with_arm_rr_mpjpes = metrics['with_arm_rr_mpjpe']
    without_arm_rr_mpjpes = metrics['without_arm_rr_mpjpe']

    visible_pct_acc = metrics['visible_pct_acc']
    cs_acc = metrics['with_arm_cs_acc']
    cs_acc_without = metrics['without_arm_cs_acc']
    rr_acc = metrics['with_arm_rr_acc']
    rr_acc_without = metrics['without_arm_rr_acc']

    _safe_call(_enable_crisp_rendering)

    ctr, mp_cs_wi, _, _ = binned_stats(visible_pct, with_arm_mpjpes, bins)
    _, mp_cs_wo, _, _ = binned_stats(visible_pct, without_arm_mpjpes, bins)
    _, mp_rr_wi, _, _ = binned_stats(visible_pct, with_arm_rr_mpjpes, bins)
    _, mp_rr_wo, _, _ = binned_stats(visible_pct, without_arm_rr_mpjpes, bins)

    ctr_acc, ac_cs_wi, _, _ = binned_stats(visible_pct_acc, cs_acc, bins)
    _, ac_cs_wo, _, _ = binned_stats(visible_pct_acc, cs_acc_without, bins)
    _, ac_rr_wi, _, _ = binned_stats(visible_pct_acc, rr_acc, bins)
    _, ac_rr_wo, _, _ = binned_stats(visible_pct_acc, rr_acc_without, bins)

    pct_mp_cs = percent_improvement(mp_cs_wo, mp_cs_wi)
    pct_mp_rr = percent_improvement(mp_rr_wo, mp_rr_wi)
    pct_ac_cs = percent_improvement(ac_cs_wo, ac_cs_wi)
    pct_ac_rr = percent_improvement(ac_rr_wo, ac_rr_wi)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    _safe_call(_blend_axes_to_paper, ax)

    ax.axhline(0.0, linestyle='--', linewidth=1.0, alpha=0.6, color='gray')

    if np.any(np.isfinite(pct_mp_cs)):
        ax.plot(ctr, pct_mp_cs, marker='o', linewidth=2.0, label='CS-MJE % gain')
    if np.any(np.isfinite(pct_mp_rr)):
        ax.plot(ctr, pct_mp_rr, marker='o', linewidth=2.0, label='RS-MJE % gain')
    if np.any(np.isfinite(pct_ac_cs)):
        ax.plot(ctr_acc, pct_ac_cs, marker='s', linewidth=2.0, label='CS-ACC % gain')
    if np.any(np.isfinite(pct_ac_rr)):
        ax.plot(ctr_acc, pct_ac_rr, marker='s', linewidth=2.0, label='RS-ACC % gain')

    ax.set_xlim(float(bins[0]), float(bins[-1]))
    ax.set_xticks((bins[:-1] + bins[1:]) * 0.5)
    ax.set_xlabel('% of visible hand joints')
    ax.set_ylabel('% improvement (higher is better)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', frameon=False)

    _safe_call(_stroke_all_text, fig, lw=0.8, fg='white')

    Path(save_png_path).parent.mkdir(parents=True, exist_ok=True)

    if _save_png_supersampled is not None:
        _safe_call(_save_png_supersampled, fig, save_png_path, scale=3)
    else:
        fig.savefig(save_png_path, bbox_inches='tight', dpi=220)

    fig.savefig(save_pdf_path, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved plot PNG: {save_png_path}')
    print(f'Saved plot PDF: {save_pdf_path}')


def main(argv=None):
    args = _parse_args(argv)

    with_arm_suffix = 'undistort_inp_true'
    without_arm_suffix = 'undistort_inp_true_no_arm_input'
    undistort_inp = True

    args.with_arm_suffix = with_arm_suffix
    args.without_arm_suffix = without_arm_suffix
    args.test_dataset_name = 'ARCTIC'

    cfg.DATASET.NAME = args.test_dataset_name

    dataset, resolved_dataset_name = _build_eval_dataset(cfg)
    cfg.DATASET.NAME = resolved_dataset_name

    left_loader = _build_eval_loader(cfg, dataset, hand_type='left', undistort_inp=undistort_inp, args=args)
    right_loader = _build_eval_loader(cfg, dataset, hand_type='right', undistort_inp=undistort_inp, args=args)

    search_bases = _prediction_search_bases(args)

    with_arm_path = _resolve_predictions_path(
        search_bases,
        dataset_name=args.test_dataset_name,
        suffix=with_arm_suffix,
    )
    without_arm_path = _resolve_predictions_path(
        search_bases,
        dataset_name=args.test_dataset_name,
        suffix=without_arm_suffix,
    )

    with_arm_predictions = _load_pickle_results(with_arm_path)
    without_arm_predictions = _load_pickle_results(without_arm_path)

    metrics = _collect_metrics(
        left_loader=left_loader,
        right_loader=right_loader,
        with_arm_predictions=with_arm_predictions,
        without_arm_predictions=without_arm_predictions,
    )

    summary = _build_summary(metrics)
    _print_summary(summary)

    output_dir = args.output_dir if args.output_dir else _default_output_dir(args, with_arm_suffix, without_arm_suffix)
    output_dir = os.path.abspath(output_dir)

    save_png_path = os.path.join(output_dir, 'percent_improvement_by_visibility.png')
    save_pdf_path = os.path.join(output_dir, 'percent_improvement_by_visibility.pdf')

    plot_percent_improvement(metrics, save_png_path=save_png_path, save_pdf_path=save_pdf_path)
    _save_artifacts(output_dir, summary, metrics, args, with_arm_path, without_arm_path)


if __name__ == '__main__':
    main()
