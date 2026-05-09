import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import os
import pickle
import sys
import traceback
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import copy
import numpy as np
from tqdm import tqdm

from settings import config as cfg
from models import LimbModel
from core import KalmanFilterCV3D
from datasets import HOT3DLoader, Arm3DDataset
from utils.evaluation_protocols import (
    evaluate_batch_hand as evaluate_batch,
    evaluate_batch_two_hand,
)
from utils.plot_utils import (
    _enable_crisp_rendering,
    _stroke_all_text,
    _save_png_supersampled,
    _blend_axes_to_paper,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_DATA_DIR = os.path.join(ROOT_DIR, '_DATA')
DEFAULT_NOISY_PREDICTIONS_DIR = os.path.join(_DATA_DIR, 'noisy_predictions')
DEFAULT_RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'intrinsics_robustness')

DATASET_NAME = 'HOT3D'
NOISE_SUFFIX_BASE = 'undistort_inp_true'
UNDISTORT_INP = True
NOISE_PERCENTAGES = [0.5, 1, 2.5, 5, 7.5, 10, 25, 50, 75, 100, 150, 200]


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Evaluate noisy intrinsic predictions for HOT3D and summarize robustness.'
    )
    parser.add_argument('--no-cit', action='store_true', help='Disable CIT module (suffix: _no_cit).')
    parser.add_argument(
        '--noisy-predictions-dir',
        default=DEFAULT_NOISY_PREDICTIONS_DIR,
        help='Directory containing noisy prediction/cache files from save_noisy_intrinsic_predictions.py.',
    )
    parser.add_argument(
        '--results-dir',
        default=DEFAULT_RESULTS_DIR,
        help='Directory where evaluation artifacts are written (default: results/intrinsics_robustness).',
    )
    parser.add_argument(
        '--baseline-noise-percent',
        type=float,
        default=0.5,
        help='Noise level used as baseline for percent-increase summaries.',
    )
    parser.add_argument(
        '--force-recompute',
        action='store_true',
        help='Recompute evaluation metrics even when cached evaluation artifacts exist.',
    )
    parser.add_argument(
        '--ignore-failure-solves',
        dest='ignore_failure_solves',
        action='store_true',
        default=True,
        help='Ignore failed solves (HAND_CS_MJE > 1000mm) when averaging hand CS/ACC metrics. Enabled by default.',
    )
    parser.add_argument(
        '--no-ignore-failure-solves',
        dest='ignore_failure_solves',
        action='store_false',
        help='Disable filtering of failed solves in hand CS/ACC metric averaging.',
    )
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=16)
    return parser.parse_args(argv)


def percent_to_suffix_token(percent):
    p = float(percent)
    if p.is_integer():
        return str(int(p))
    return f"{p:g}".replace('.', '_')


def get_eval_cache_paths(output_dir, dataset_name, suffix):
    stem = os.path.join(output_dir, f'{dataset_name}_{suffix}')
    return {
        'result_text_path': f'{stem}_evaluation_results.txt',
        'metrics_summary_path': f'{stem}_metrics_summary.pkl',
    }


def _resolve_noise_suffix_base(no_cit=False):
    suffix_base = NOISE_SUFFIX_BASE
    if no_cit:
        suffix_base += '_no_cit'
    return suffix_base


def _load_metrics_summary_from_sweep_report(output_dir, dataset_name, suffix, sweep_suffix_base=NOISE_SUFFIX_BASE):
    report_pkl_path = os.path.join(output_dir, f'{dataset_name}_{sweep_suffix_base}_noise_sweep_report.pkl')
    if not os.path.exists(report_pkl_path):
        return None

    with open(report_pkl_path, 'rb') as f:
        report = pickle.load(f)

    for run in report.get('runs', []):
        if run.get('suffix') == suffix and isinstance(run.get('metrics_summary'), dict):
            return run['metrics_summary']
    return None


def load_cached_evaluation(output_dir, dataset_name, suffix, sweep_suffix_base=NOISE_SUFFIX_BASE):
    paths = get_eval_cache_paths(output_dir, dataset_name, suffix)
    result_text_path = paths['result_text_path']
    metrics_summary_path = paths['metrics_summary_path']

    if not os.path.exists(result_text_path):
        return None

    metrics_summary = None
    if os.path.exists(metrics_summary_path):
        with open(metrics_summary_path, 'rb') as f:
            metrics_summary = pickle.load(f)

    if metrics_summary is None:
        metrics_summary = _load_metrics_summary_from_sweep_report(
            output_dir,
            dataset_name,
            suffix,
            sweep_suffix_base=sweep_suffix_base,
        )
        if metrics_summary is not None:
            with open(metrics_summary_path, 'wb') as f:
                pickle.dump(metrics_summary, f)

    if metrics_summary is None:
        return None

    result_text = ''
    try:
        with open(result_text_path, 'r') as f:
            result_text = f.read()
    except Exception:
        result_text = ''

    return {
        'metrics_summary': metrics_summary,
        'result_text': result_text,
        'result_text_path': result_text_path,
        'metrics_summary_path': metrics_summary_path,
        'from_cache': True,
    }


def safe_mean(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr))


def safe_sum(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return np.nan
    return float(np.sum(arr))


def safe_percent_increase(current, baseline, eps=1e-9):
    if current is None or baseline is None:
        return np.nan
    if not np.isfinite(current) or not np.isfinite(baseline):
        return np.nan
    denom = abs(baseline)
    if denom < eps:
        return np.nan
    return float((current - baseline) / denom * 100.0)


def flatten_dict(d, prefix=''):
    flat = {}
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_dict(value, full_key))
        else:
            try:
                scalar = float(value)
                flat[full_key] = scalar
            except Exception:
                continue
    return flat


def try_pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return np.nan
    x = x[mask]
    y = y[mask]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def extract_camera_features(camera_analysis):
    if not camera_analysis or not camera_analysis.get('aggregate'):
        return {
            'angle_error_mean_deg': np.nan,
            'angle_error_p95_deg': np.nan,
            'depth_error_mean_m': np.nan,
            'depth_error_p95_m': np.nan,
            'depth_error_mean_mm': np.nan,
            'depth_error_p95_mm': np.nan,
            'radial_angle_mean_deg': np.nan,
            'radial_depth_mean_m': np.nan,
            'radial_depth_mean_mm': np.nan,
        }

    agg = camera_analysis['aggregate']
    radial_profile = agg.get('radial_profile', {})
    depth_error = agg.get('depth_error_m', {})
    angle_error = agg.get('angle_error_deg', {})

    depth_mean = np.asarray(depth_error.get('mean', []), dtype=np.float64)
    depth_p95 = np.asarray(depth_error.get('p95', []), dtype=np.float64)

    radial_angle = np.asarray(radial_profile.get('angle_mean_deg', []), dtype=np.float64)
    radial_depth = np.asarray(radial_profile.get('depth_mean_m', []), dtype=np.float64)

    depth_error_mean_m = float(np.nanmean(depth_mean)) if depth_mean.size else np.nan
    depth_error_p95_m = float(np.nanmean(depth_p95)) if depth_p95.size else np.nan
    radial_depth_mean_m = float(np.nanmean(radial_depth)) if radial_depth.size else np.nan
    return {
        'angle_error_mean_deg': float(angle_error.get('mean_of_means', np.nan)),
        'angle_error_p95_deg': float(angle_error.get('mean_of_p95', np.nan)),
        'depth_error_mean_m': depth_error_mean_m,
        'depth_error_p95_m': depth_error_p95_m,
        'depth_error_mean_mm': depth_error_mean_m * 1000.0 if np.isfinite(depth_error_mean_m) else np.nan,
        'depth_error_p95_mm': depth_error_p95_m * 1000.0 if np.isfinite(depth_error_p95_m) else np.nan,
        'radial_angle_mean_deg': float(np.nanmean(radial_angle)) if radial_angle.size else np.nan,
        'radial_depth_mean_m': radial_depth_mean_m,
        'radial_depth_mean_mm': radial_depth_mean_m * 1000.0 if np.isfinite(radial_depth_mean_m) else np.nan,
    }


def get_outs(config, limb_model, side, batch, device, results, filter=None):
    data, meta = batch
    
    batch_samplekeys = meta['samplekey']

    outs = {
        'pred_hand_j3d': [],
        'pred_hand_j2d': [],
        'gt_hand_j3d': [],
        'gt_hand_j2d': [],
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

        result = results[samplekey][side]
        pred_transl =  result['pred_transl'] 

        if config.DATASET.NAME == 'HO3D':
            gt_hand_j3d = data['hand_j3d'][i].cpu().numpy()
        else:
            gt_hand_j3d = result['hand']['gt_j3d']

        outs['gt_hand_j3d'].append(gt_hand_j3d)

        outs['gt_transl'].append(gt_hand_j3d[0])
        outs['pred_transl'].append(result['hand']['pred_j3d'][0])

        visible_hand = data['visible_hand'][i].item()

        if filter is not None:
            if False:#not visible_hand:
                filter.reset_state()
            else:
                pred_hand_vertices = result['hand']['pred_vertices']
                pred_hand_j3d = result['hand']['pred_j3d']
    
                inp_pred_transl = torch.tensor(pred_transl, device=device)
                ft_pred_transl = filter.step(inp_pred_transl.squeeze(0), visible_hand).unsqueeze(0).cpu().numpy()
            
                pred_hand_vertices = (pred_hand_vertices - pred_transl) + ft_pred_transl 
                pred_hand_j3d = (pred_hand_j3d - pred_transl) + ft_pred_transl 

                result['hand']['pred_vertices'] = pred_hand_vertices
                result['hand']['pred_j3d'] = pred_hand_j3d

        outs['pred_hand_j3d'].append(np.nan_to_num(result['hand']['pred_j3d'], nan=0.0))
        outs['pred_hand_j2d'].append(np.nan_to_num(result['hand']['pred_j2d'], nan=0.0))
        outs['pred_hand_vertices'].append(result['hand']['pred_vertices'])
        outs['gt_hand_j2d'].append(result['hand']['gt_j2d'])
        outs['gt_hand_vertices'].append(result['hand']['gt_vertices'])

        outs['pred_arm_j3d'].append(result['arm']['pred_j3d'])
        outs['pred_arm_j2d'].append(result['arm']['pred_j2d'])
        outs['gt_arm_j3d'].append(result['arm']['gt_j3d'])
        outs['gt_arm_j2d'].append(result['arm']['gt_j2d'])

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


def compute_and_print_metrics(left_results, right_results):
    texts = []
    for limb in ['hand']:
        left_fail_rate = np.mean(left_results[limb]['FAILURE_RATE'])
        right_fail_rate = np.mean(right_results[limb]['FAILURE_RATE'])
        mean_fail_rate = (left_fail_rate + right_fail_rate) / 2

        left_acc_error = np.mean(left_results[limb]['ACC_ERROR'])
        left_cs  = np.mean(left_results [limb]['CS_MJE'])
        left_rr  = np.mean(left_results [limb]['RR_MJE'])
        left_pa  = np.mean(left_results [limb]['PA_MJE'])
        left_in_cs  = np.mean(left_results [limb]['INVISIBLE_CS_MJE'])
        left_in_rr  = np.mean(left_results [limb]['INVISIBLE_RR_MJE'])
        left_in_pa  = np.mean(left_results [limb]['INVISIBLE_PA_MJE'])
        left_in_acc  = np.mean(left_results [limb]['INVISIBLE_ACC_ERROR'])
        left_occ_cs  = np.mean(left_results [limb]['OCC_CS_MJE'])
        left_occ_rr  = np.mean(left_results [limb]['OCC_RR_MJE'])
        left_occ_pa  = np.mean(left_results [limb]['OCC_PA_MJE'])
        left_vis_cs  = np.mean(left_results [limb]['VIS_CS_MJE'])
        left_vis_rr  = np.mean(left_results [limb]['VIS_RR_MJE'])
        left_vis_pa  = np.mean(left_results [limb]['VIS_PA_MJE'])
        left_occ_cnt  = np.sum(left_results [limb]['OCC_CNT'])
        left_vis_cnt  = np.sum(left_results [limb]['VIS_CNT'])

        right_acc_error = np.mean(right_results[limb]['ACC_ERROR'])
        right_cs = np.mean(right_results[limb]['CS_MJE'])
        right_rr = np.mean(right_results[limb]['RR_MJE'])
        right_pa = np.mean(right_results[limb]['PA_MJE'])
        right_in_cs = np.mean(right_results[limb]['INVISIBLE_CS_MJE'])
        right_in_rr = np.mean(right_results[limb]['INVISIBLE_RR_MJE'])
        right_in_pa = np.mean(right_results[limb]['INVISIBLE_PA_MJE'])
        right_in_acc = np.mean(right_results[limb]['INVISIBLE_ACC_ERROR'])
        right_occ_cs  = np.mean(right_results [limb]['OCC_CS_MJE'])
        right_occ_rr  = np.mean(right_results [limb]['OCC_RR_MJE'])
        right_occ_pa  = np.mean(right_results [limb]['OCC_PA_MJE'])
        right_vis_cs  = np.mean(right_results [limb]['VIS_CS_MJE'])
        right_vis_rr  = np.mean(right_results [limb]['VIS_RR_MJE'])
        right_vis_pa  = np.mean(right_results [limb]['VIS_PA_MJE'])
        right_occ_cnt  = np.sum(right_results [limb]['OCC_CNT'])
        right_vis_cnt  = np.sum(right_results [limb]['VIS_CNT'])
        
        mean_acc_error = (left_acc_error + right_acc_error) / 2
        mean_in_acc_error = (left_in_acc + right_in_acc) / 2

        mean_cs = (left_cs  + right_cs ) / 2
        mean_rr = (left_rr  + right_rr ) / 2
        mean_pa = (left_pa  + right_pa ) / 2
        
        mean_in_cs = (left_in_cs  + right_in_cs ) / 2
        mean_in_rr = (left_in_rr  + right_in_rr ) / 2
        mean_in_pa = (left_in_pa  + right_in_pa ) / 2

        mean_occ_cs = (left_occ_cs  + right_occ_cs ) / 2
        mean_occ_rr = (left_occ_rr  + right_occ_rr ) / 2
        mean_occ_pa = (left_occ_pa  + right_occ_pa ) / 2

        mean_vis_cs = (left_vis_cs  + right_vis_cs ) / 2
        mean_vis_rr = (left_vis_rr  + right_vis_rr ) / 2
        mean_vis_pa = (left_vis_pa  + right_vis_pa ) / 2

        total_occ_cnt = (left_occ_cnt  + right_occ_cnt )
        total_vis_cnt = (left_vis_cnt  + right_vis_cnt )

        text = (f"{limb.capitalize():<5} | "
                f"Left  (CS_mm:{left_cs:6.2f}, RR_mm:{left_rr:6.2f}, PA_mm:{left_pa:6.2f}) | "
                f"Right (CS_mm:{right_cs:6.2f}, RR_mm:{right_rr:6.2f}, PA_mm:{right_pa:6.2f}) | "
                f"Mean  (CS_mm:{mean_cs:6.2f}, RR_mm:{mean_rr:6.2f}, PA_mm:{mean_pa:6.2f}) | "
                f"Mean  ACC_mps2:{mean_acc_error:6.2f} | "
                f"Left Invisible (CS_mm:{left_in_cs:6.2f}, RR_mm:{left_in_rr:6.2f}, PA_mm:{left_in_pa:6.2f}) | "
                f"Right Invisible (CS_mm:{right_in_cs:6.2f}, RR_mm:{right_in_rr:6.2f}, PA_mm:{right_in_pa:6.2f}) | "
                f"Mean Invisible (CS_mm:{mean_in_cs:6.2f}, RR_mm:{mean_in_rr:6.2f}, PA_mm:{mean_in_pa:6.2f}) | "
                f"Mean Invisible ACC_mps2:{mean_in_acc_error:6.2f} | "
                f"Mean Occluded (CS_mm:{mean_occ_cs:6.2f}, RR_mm:{mean_occ_rr:6.2f}, PA_mm:{mean_occ_pa:6.2f}) | " 
                f"Mean Vis Joints (CS_mm:{mean_vis_cs:6.2f}, RR_mm:{mean_vis_rr:6.2f}, PA_mm:{mean_vis_pa:6.2f}) | " 
                f"Total Joints (VIS:{total_vis_cnt}, OCC:{total_occ_cnt}) | "
                f"Left Fail Rate:{left_fail_rate:5.2f}% | Right Fail Rate:{right_fail_rate:5.2f}% | Mean Fail Rate:{mean_fail_rate:5.2f}% | "
        )

        texts.append(text)

        print(text)

    return '\n'.join(texts)


def compute_and_print_metrics_two_hand(two_hand_results):
    """
    Prints a single summary line for the two-hand (relative) evaluation.
    """
    cs = np.mean(two_hand_results['hand']['CS_MJE'])
    rr = np.mean(two_hand_results['hand']['RR_MJE'])
    pa = np.mean(two_hand_results['hand']['PA_MJE'])
    acc = np.mean(two_hand_results['hand']['ACC_ERROR'])

    in_cs = np.mean(two_hand_results['hand']['INVISIBLE_CS_MJE'])
    in_rr = np.mean(two_hand_results['hand']['INVISIBLE_RR_MJE'])
    in_pa = np.mean(two_hand_results['hand']['INVISIBLE_PA_MJE'])
    in_acc = np.mean(two_hand_results['hand']['INVISIBLE_ACC_ERROR'])

    text = (
        f"TwoHand | "
        f"Relative (CS_mm:{cs:6.2f}, RR_mm:{rr:6.2f}, PA_mm:{pa:6.2f}) | "
        f"ACC_mps2:{acc:6.2f} | "
        f"Invisible (CS_mm:{in_cs:6.2f}, RR_mm:{in_rr:6.2f}, PA_mm:{in_pa:6.2f}) | "
        f"Invisible ACC_mps2:{in_acc:6.2f}"
    )
    print(text)
    return text


def summarize_metrics(left_results, right_results, two_hand_results):
    summary = {'hand': {}, 'two_hand': {}}
    for limb in ['hand']:
        left_fail_rate = safe_mean(left_results[limb]['FAILURE_RATE'])
        right_fail_rate = safe_mean(right_results[limb]['FAILURE_RATE'])
        mean_fail_rate = safe_mean([left_fail_rate, right_fail_rate])

        left_acc_error = safe_mean(left_results[limb]['ACC_ERROR'])
        left_cs  = safe_mean(left_results [limb]['CS_MJE'])
        left_rr  = safe_mean(left_results [limb]['RR_MJE'])
        left_pa  = safe_mean(left_results [limb]['PA_MJE'])
        left_in_cs  = safe_mean(left_results [limb]['INVISIBLE_CS_MJE'])
        left_in_rr  = safe_mean(left_results [limb]['INVISIBLE_RR_MJE'])
        left_in_pa  = safe_mean(left_results [limb]['INVISIBLE_PA_MJE'])
        left_in_acc  = safe_mean(left_results [limb]['INVISIBLE_ACC_ERROR'])
        left_occ_cs  = safe_mean(left_results [limb]['OCC_CS_MJE'])
        left_occ_rr  = safe_mean(left_results [limb]['OCC_RR_MJE'])
        left_occ_pa  = safe_mean(left_results [limb]['OCC_PA_MJE'])
        left_vis_cs  = safe_mean(left_results [limb]['VIS_CS_MJE'])
        left_vis_rr  = safe_mean(left_results [limb]['VIS_RR_MJE'])
        left_vis_pa  = safe_mean(left_results [limb]['VIS_PA_MJE'])
        left_occ_cnt  = safe_sum(left_results [limb]['OCC_CNT'])
        left_vis_cnt  = safe_sum(left_results [limb]['VIS_CNT'])

        right_acc_error = safe_mean(right_results[limb]['ACC_ERROR'])
        right_cs = safe_mean(right_results[limb]['CS_MJE'])
        right_rr = safe_mean(right_results[limb]['RR_MJE'])
        right_pa = safe_mean(right_results[limb]['PA_MJE'])
        right_in_cs = safe_mean(right_results[limb]['INVISIBLE_CS_MJE'])
        right_in_rr = safe_mean(right_results[limb]['INVISIBLE_RR_MJE'])
        right_in_pa = safe_mean(right_results[limb]['INVISIBLE_PA_MJE'])
        right_in_acc = safe_mean(right_results[limb]['INVISIBLE_ACC_ERROR'])
        right_occ_cs  = safe_mean(right_results [limb]['OCC_CS_MJE'])
        right_occ_rr  = safe_mean(right_results [limb]['OCC_RR_MJE'])
        right_occ_pa  = safe_mean(right_results [limb]['OCC_PA_MJE'])
        right_vis_cs  = safe_mean(right_results [limb]['VIS_CS_MJE'])
        right_vis_rr  = safe_mean(right_results [limb]['VIS_RR_MJE'])
        right_vis_pa  = safe_mean(right_results [limb]['VIS_PA_MJE'])
        right_occ_cnt  = safe_sum(right_results [limb]['OCC_CNT'])
        right_vis_cnt  = safe_sum(right_results [limb]['VIS_CNT'])
        
        mean_acc_error = safe_mean([left_acc_error, right_acc_error])
        mean_in_acc_error = safe_mean([left_in_acc, right_in_acc])

        mean_cs = safe_mean([left_cs, right_cs])
        mean_rr = safe_mean([left_rr, right_rr])
        mean_pa = safe_mean([left_pa, right_pa])
        
        mean_in_cs = safe_mean([left_in_cs, right_in_cs])
        mean_in_rr = safe_mean([left_in_rr, right_in_rr])
        mean_in_pa = safe_mean([left_in_pa, right_in_pa])

        mean_occ_cs = safe_mean([left_occ_cs, right_occ_cs])
        mean_occ_rr = safe_mean([left_occ_rr, right_occ_rr])
        mean_occ_pa = safe_mean([left_occ_pa, right_occ_pa])

        mean_vis_cs = safe_mean([left_vis_cs, right_vis_cs])
        mean_vis_rr = safe_mean([left_vis_rr, right_vis_rr])
        mean_vis_pa = safe_mean([left_vis_pa, right_vis_pa])

        total_occ_cnt = safe_sum([left_occ_cnt, right_occ_cnt])
        total_vis_cnt = safe_sum([left_vis_cnt, right_vis_cnt])

        summary[limb] = {
            'FAILURE_RATE': mean_fail_rate,
            'ACC_ERROR': mean_acc_error,
            'CS_MJE': mean_cs,
            'RR_MJE': mean_rr,
            'PA_MJE': mean_pa,
            'INVISIBLE_CS_MJE': mean_in_cs,
            'INVISIBLE_RR_MJE': mean_in_rr,
            'INVISIBLE_PA_MJE': mean_in_pa,
            'INVISIBLE_ACC_ERROR': mean_in_acc_error,
            'OCC_CS_MJE': mean_occ_cs,
            'OCC_RR_MJE': mean_occ_rr,
            'OCC_PA_MJE': mean_occ_pa,
            'VIS_CS_MJE': mean_vis_cs,
            'VIS_RR_MJE': mean_vis_rr,
            'VIS_PA_MJE': mean_vis_pa,
            'OCC_CNT': total_occ_cnt,
            'VIS_CNT': total_vis_cnt,
        }

    summary['two_hand'] = {
        'ACC_ERROR': safe_mean(two_hand_results['hand']['ACC_ERROR']),
        'CS_MJE': safe_mean(two_hand_results['hand']['CS_MJE']),
        'RR_MJE': safe_mean(two_hand_results['hand']['RR_MJE']),
        'PA_MJE': safe_mean(two_hand_results['hand']['PA_MJE']),
        'INVISIBLE_CS_MJE': safe_mean(two_hand_results['hand']['INVISIBLE_CS_MJE']),
        'INVISIBLE_RR_MJE': safe_mean(two_hand_results['hand']['INVISIBLE_RR_MJE']),
        'INVISIBLE_PA_MJE': safe_mean(two_hand_results['hand']['INVISIBLE_PA_MJE']),
        'INVISIBLE_ACC_ERROR': safe_mean(two_hand_results['hand']['INVISIBLE_ACC_ERROR']),
    }

    return summary


def build_noise_sweep_report(run_entries, baseline_noise_percent=0.5):
    if not run_entries:
        return {
            'runs': [],
            'baseline_noise_percent': baseline_noise_percent,
            'baseline_suffix': None,
            'metric_vs_camera_correlation': {},
        }

    runs_sorted = sorted(run_entries, key=lambda r: r['noise_percent'])
    baseline = None
    for run in runs_sorted:
        if abs(run['noise_percent'] - baseline_noise_percent) < 1e-9:
            baseline = run
            break
    if baseline is None:
        baseline = runs_sorted[0]

    baseline_flat = flatten_dict(baseline['metrics_summary'])

    key_metrics = [
        'hand.CS_MJE', 'hand.RR_MJE', 'hand.PA_MJE', 'hand.ACC_ERROR',
        'two_hand.CS_MJE', 'two_hand.RR_MJE', 'two_hand.PA_MJE', 'two_hand.ACC_ERROR',
    ]

    for run in runs_sorted:
        run_flat = flatten_dict(run['metrics_summary'])
        metric_increase_pct = {}
        for key, value in run_flat.items():
            metric_increase_pct[key] = safe_percent_increase(value, baseline_flat.get(key, np.nan))
        run['metric_increase_pct'] = metric_increase_pct

        depth_err = run['camera_features'].get('depth_error_mean_m', np.nan)
        depth_err_mm = run['camera_features'].get('depth_error_mean_mm', np.nan)
        radial_angle = run['camera_features']['radial_angle_mean_deg']
        radial_depth = run['camera_features'].get('radial_depth_mean_m', np.nan)
        radial_depth_mm = run['camera_features'].get('radial_depth_mean_mm', np.nan)

        run['key_metric_vs_camera'] = {}
        for key in key_metrics:
            inc = metric_increase_pct.get(key, np.nan)
            run['key_metric_vs_camera'][key] = {
                'increase_pct': inc,
                'increase_pct_per_depth_error_m': (inc / depth_err) if np.isfinite(inc) and np.isfinite(depth_err) and abs(depth_err) > 1e-9 else np.nan,
                'increase_pct_per_depth_error_mm': (inc / depth_err_mm) if np.isfinite(inc) and np.isfinite(depth_err_mm) and abs(depth_err_mm) > 1e-9 else np.nan,
                'increase_pct_per_radial_angle_deg': (inc / radial_angle) if np.isfinite(inc) and np.isfinite(radial_angle) and abs(radial_angle) > 1e-9 else np.nan,
                'increase_pct_per_radial_depth_error_m': (inc / radial_depth) if np.isfinite(inc) and np.isfinite(radial_depth) and abs(radial_depth) > 1e-9 else np.nan,
                'increase_pct_per_radial_depth_error_mm': (inc / radial_depth_mm) if np.isfinite(inc) and np.isfinite(radial_depth_mm) and abs(radial_depth_mm) > 1e-9 else np.nan,
            }

    metric_vs_camera_correlation = {}
    for key in key_metrics:
        inc = np.array([r['metric_increase_pct'].get(key, np.nan) for r in runs_sorted], dtype=np.float64)
        depth = np.array([r['camera_features'].get('depth_error_mean_m', np.nan) for r in runs_sorted], dtype=np.float64)
        depth_mm = np.array([r['camera_features'].get('depth_error_mean_mm', np.nan) for r in runs_sorted], dtype=np.float64)
        radial_angle = np.array([r['camera_features']['radial_angle_mean_deg'] for r in runs_sorted], dtype=np.float64)
        radial_depth = np.array([r['camera_features'].get('radial_depth_mean_m', np.nan) for r in runs_sorted], dtype=np.float64)
        radial_depth_mm = np.array([r['camera_features'].get('radial_depth_mean_mm', np.nan) for r in runs_sorted], dtype=np.float64)

        metric_vs_camera_correlation[key] = {
            'corr_increase_pct_vs_depth_error_mean_m': try_pearson(inc, depth),
            'corr_increase_pct_vs_depth_error_mean_mm': try_pearson(inc, depth_mm),
            'corr_increase_pct_vs_radial_angle_mean_deg': try_pearson(inc, radial_angle),
            'corr_increase_pct_vs_radial_depth_mean_m': try_pearson(inc, radial_depth),
            'corr_increase_pct_vs_radial_depth_mean_mm': try_pearson(inc, radial_depth_mm),
        }

    return {
        'baseline_noise_percent': baseline['noise_percent'],
        'baseline_suffix': baseline['suffix'],
        'runs': runs_sorted,
        'metric_vs_camera_correlation': metric_vs_camera_correlation,
    }


def _format_report_value(value, precision):
    try:
        value = float(value)
    except Exception:
        return 'nan'
    if not np.isfinite(value):
        return 'nan'
    return f'{value:.{precision}f}'


def report_to_text(report):
    lines = []
    if not report.get('runs'):
        lines.append('No runs found. Please generate noisy prediction files first.')
        return "\n".join(lines)

    lines.append(f"Baseline noise level: {report['baseline_noise_percent']}% ({report['baseline_suffix']})")
    lines.append("")

    columns = [
        ('Noise%', 1, lambda run: run['noise_percent']),
        ('Hand CS(mm)', 3, lambda run: run['metrics_summary']['hand']['CS_MJE']),
        ('Hand CS inc%', 3, lambda run: run['metric_increase_pct'].get('hand.CS_MJE', np.nan)),
        ('Hand CS-ACC(m/s^2)', 3, lambda run: run['metrics_summary']['hand'].get('ACC_ERROR', np.nan)),
        ('Hand CS-ACC inc%', 3, lambda run: run['metric_increase_pct'].get('hand.ACC_ERROR', np.nan)),
        ('Hand RS(mm)', 3, lambda run: run['metrics_summary']['hand']['RR_MJE']),
        ('Hand RS inc%', 3, lambda run: run['metric_increase_pct'].get('hand.RR_MJE', np.nan)),
        ('Hand PS(mm)', 3, lambda run: run['metrics_summary']['hand']['PA_MJE']),
        ('Hand PS inc%', 3, lambda run: run['metric_increase_pct'].get('hand.PA_MJE', np.nan)),
        ('Hand Fail%', 2, lambda run: run['metrics_summary']['hand'].get('FAILURE_RATE', np.nan)),
        ('TwoHand CS(mm)', 3, lambda run: run['metrics_summary']['two_hand']['CS_MJE']),
        ('TwoHand CS inc%', 3, lambda run: run['metric_increase_pct'].get('two_hand.CS_MJE', np.nan)),
        ('CamDepthErr(mm)', 3, lambda run: run['camera_features'].get('depth_error_mean_mm', np.nan)),
    ]

    rows = []
    for run in report['runs']:
        rows.append([
            _format_report_value(getter(run), precision)
            for _, precision, getter in columns
        ])

    widths = []
    for col_idx, (header, _, _) in enumerate(columns):
        widths.append(max(len(header), *(len(row[col_idx]) for row in rows)))

    header_line = " | ".join(header.ljust(widths[idx]) for idx, (header, _, _) in enumerate(columns))
    separator_line = "-+-".join('-' * widths[idx] for idx in range(len(columns)))

    lines.append(header_line)
    lines.append(separator_line)

    for row in rows:
        lines.append(" | ".join(row[idx].rjust(widths[idx]) for idx in range(len(columns))))

    lines.append("")
    lines.append("Correlation: metric increase (%) vs camera error features")
    for key, vals in report['metric_vs_camera_correlation'].items():
        lines.append(
            f"{key}: depth(mm)={vals['corr_increase_pct_vs_depth_error_mean_mm']:.4f}"
        )
    return "\n".join(lines)


def save_noise_sweep_plots(report, out_prefix):
    runs = report.get('runs', [])
    if not runs:
        return

    marker_size = 3.0

    noise = np.array([r['noise_percent'] for r in runs], dtype=np.float64)
    hand_cs = np.array([r['metrics_summary']['hand']['CS_MJE'] for r in runs], dtype=np.float64)
    two_cs = np.array([r['metrics_summary']['two_hand']['CS_MJE'] for r in runs], dtype=np.float64)

    hand_inc = np.array([r['metric_increase_pct'].get('hand.CS_MJE', np.nan) for r in runs], dtype=np.float64)
    two_inc = np.array([r['metric_increase_pct'].get('two_hand.CS_MJE', np.nan) for r in runs], dtype=np.float64)

    depth_mm = np.array([r['camera_features'].get('depth_error_mean_mm', np.nan) for r in runs], dtype=np.float64)

    hand_rs = np.array([r['metrics_summary']['hand']['RR_MJE'] for r in runs], dtype=np.float64)
    two_rs = np.array([r['metrics_summary']['two_hand']['RR_MJE'] for r in runs], dtype=np.float64)

    finite_depth_mm = depth_mm[np.isfinite(depth_mm)]
    depth_xmax = int(np.ceil(np.max(finite_depth_mm))) if finite_depth_mm.size else None

    def _set_camera_error_xticks(ax):
        if depth_xmax is None:
            return

        xticks = np.asarray(ax.get_xticks(), dtype=np.float64)
        xticks = xticks[np.isfinite(xticks)]
        xticks = xticks[xticks >= -1e-9]
        xticks = xticks[xticks <= depth_xmax + 1e-9]

        if xticks.size >= 2:
            diffs = np.diff(np.sort(xticks))
            diffs = diffs[np.abs(diffs) > 1e-9]
            tick_step = float(np.median(diffs)) if diffs.size else np.nan
        else:
            tick_step = np.nan

        if xticks.size and np.isfinite(tick_step):
            last_auto_tick = float(np.max(xticks))
            if not np.isclose(last_auto_tick, depth_xmax) and (depth_xmax - last_auto_tick) < 0.5 * tick_step:
                xticks = xticks[~np.isclose(xticks, last_auto_tick)]

        if not np.any(np.isclose(xticks, depth_xmax)):
            xticks = np.append(xticks, depth_xmax)
        if not np.any(np.isclose(xticks, 0.0)):
            xticks = np.append(xticks, 0.0)

        ax.set_xticks(np.sort(xticks))

    _enable_crisp_rendering()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(noise, hand_cs, marker='o', markersize=marker_size, label='Hand')
    ax.plot(noise, two_cs, marker='o', markersize=marker_size, label='Two Hand Relative')
    ax.set_xlabel('Noise level (%)')
    ax.set_ylabel('CS-MJE (mm)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _blend_axes_to_paper(ax)
    _stroke_all_text(fig)
    _save_png_supersampled(fig, f'{out_prefix}_cs_mje_vs_noise', scale=2)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(noise, hand_inc, marker='o', markersize=marker_size, label='Hand %')
    ax.plot(noise, two_inc, marker='o', markersize=marker_size, label='Two Hand Relative %')
    ax.set_xlabel('Noise level (%)')
    ax.set_ylabel('CS-MJE Increase (%)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _blend_axes_to_paper(ax)
    _stroke_all_text(fig)
    _save_png_supersampled(fig, f'{out_prefix}_cs_mje_increase_vs_noise', scale=2)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depth_mm, hand_cs, marker='o', markersize=marker_size, zorder=3, label='Hand')
    ax.plot(depth_mm, two_cs, marker='o', markersize=marker_size, zorder=3, label='Two Hand Relative')
    ax.set_xlabel('Mean Camera Geometry Error (mm)')
    ax.set_ylabel('CS-MJE (mm)')
    _set_camera_error_xticks(ax)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _blend_axes_to_paper(ax)
    _stroke_all_text(fig)
    _save_png_supersampled(fig, f'{out_prefix}_camera_error_vs_hand_cs_mje', scale=2)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depth_mm, hand_rs, marker='o', markersize=marker_size, zorder=3, label='Hand')
    ax.set_xlabel('Mean Camera Geometry Error (mm)')
    ax.set_ylabel('RS-MJE (mm)')
    _set_camera_error_xticks(ax)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _blend_axes_to_paper(ax)
    _stroke_all_text(fig)
    _save_png_supersampled(fig, f'{out_prefix}_camera_error_vs_hand_rs_mje', scale=2)
    plt.close(fig)

def evaluate_model(config, left_loader, right_loader, limb_model, output_dir, device, batch_size, results, suffix,
                   force_recompute=False, sweep_suffix_base=NOISE_SUFFIX_BASE, ignore_failure_solves=True):
    model_folder = cfg.DATASET.NAME
    cache_paths = get_eval_cache_paths(output_dir, model_folder, suffix)
    save_path = cache_paths['result_text_path']
    metrics_summary_path = cache_paths['metrics_summary_path']

    if not force_recompute:
        cached = load_cached_evaluation(
            output_dir,
            model_folder,
            suffix,
            sweep_suffix_base=sweep_suffix_base,
        )
        if cached is not None:
            print(f'Using cached evaluation for {suffix}: {cached["result_text_path"]}')
            return cached

    left_iterator = iter(left_loader)
    right_iterator = iter(right_loader)

    left_results = {
        'hand': {
            'FAILURE_RATE': [],
            'ACC_ERROR': [],

            'CS_MJE': [],
            'RR_MJE': [],
            'PA_MJE': [],

            'INVISIBLE_CS_MJE': [],
            'INVISIBLE_RR_MJE': [],
            'INVISIBLE_PA_MJE': [],
            'INVISIBLE_ACC_ERROR': [],

            'OCC_CS_MJE': [],
            'OCC_RR_MJE': [],
            'OCC_PA_MJE': [],

            'VIS_CS_MJE': [],
            'VIS_RR_MJE': [],
            'VIS_PA_MJE': [],

            'OCC_CNT': [],
            'VIS_CNT': [],
        },
    }
    right_results = copy.deepcopy(left_results)

    two_hand_results = {
        'hand': {
            'ACC_ERROR': [],
            'CS_MJE': [], 'RR_MJE': [], 'PA_MJE': [],
            'INVISIBLE_CS_MJE': [], 'INVISIBLE_RR_MJE': [], 'INVISIBLE_PA_MJE': [], 'INVISIBLE_ACC_ERROR': [],
        }
    }

    freq = 30.0
    left_one_euro_filter = KalmanFilterCV3D(q_pos=0.001, q_vel=1e-05, r_meas=0.001, freq=freq).to(device)
    right_one_euro_filter = KalmanFilterCV3D(q_pos=0.001, q_vel=1e-05, r_meas=0.001, freq=freq).to(device)

    progress_bar = tqdm(total=len(left_loader), desc='evaluating', unit='frame')
    current_batch = 0
    try:
        while True:
            with torch.no_grad():
                try:
                    try:
                        left_batch = next(left_iterator)
                        right_batch = next(right_iterator)

                        current_batch += 1
                    except StopIteration:
                        break
                    
                    left_outs = get_outs(config, limb_model, 'left', left_batch, device, results, filter=left_one_euro_filter)
                    right_outs = get_outs(config, limb_model, 'right', right_batch, device, results, filter=right_one_euro_filter)

                    left_results_batch = evaluate_batch(left_outs, ignore_failure_solves=ignore_failure_solves)
                    right_results_batch = evaluate_batch(right_outs, ignore_failure_solves=ignore_failure_solves)

                    two_hand_batch = evaluate_batch_two_hand(left_outs, right_outs)

                except Exception as e:
                    print(f"Error evaluating batch {current_batch}: {e}")
                    traceback.print_exc()
                    continue
                
            for limb in ['hand']:
                left_results[limb]['FAILURE_RATE'].append(left_results_batch[limb].get('FAILURE_RATE', 0))
                left_results[limb]['ACC_ERROR'].append(left_results_batch[limb]['ACC_ERROR'])

                left_results[limb]['CS_MJE'].append(left_results_batch[limb]['CS_MPJPE'])
                left_results[limb]['RR_MJE'].append(left_results_batch[limb]['RR_MPJPE'])
                left_results[limb]['PA_MJE'].append(left_results_batch[limb]['PA_MPJPE'])

                left_results[limb]['INVISIBLE_CS_MJE'].append(left_results_batch[limb].get('INVISIBLE_CS_MPJPE', 0))
                left_results[limb]['INVISIBLE_RR_MJE'].append(left_results_batch[limb].get('INVISIBLE_RR_MPJPE', 0))
                left_results[limb]['INVISIBLE_PA_MJE'].append(left_results_batch[limb].get('INVISIBLE_PA_MPJPE', 0))  
                left_results[limb]['INVISIBLE_ACC_ERROR'].append(left_results_batch[limb].get('INVISIBLE_ACC_ERROR', 0))  

                left_results[limb]['OCC_CS_MJE'].append(left_results_batch[limb].get('OCC_CS_MPJPE', 0))
                left_results[limb]['OCC_RR_MJE'].append(left_results_batch[limb].get('OCC_RR_MPJPE', 0))
                left_results[limb]['OCC_PA_MJE'].append(left_results_batch[limb].get('OCC_PA_MPJPE', 0))  

                left_results[limb]['VIS_CS_MJE'].append(left_results_batch[limb].get('VIS_CS_MPJPE', 0))
                left_results[limb]['VIS_RR_MJE'].append(left_results_batch[limb].get('VIS_RR_MPJPE', 0))
                left_results[limb]['VIS_PA_MJE'].append(left_results_batch[limb].get('VIS_PA_MPJPE', 0))  

                left_results[limb]['OCC_CNT'].append(left_results_batch[limb].get('OCC_CNT', 0))  
                left_results[limb]['VIS_CNT'].append(left_results_batch[limb].get('VIS_CNT', 0))  

                right_results[limb]['FAILURE_RATE'].append(right_results_batch[limb].get('FAILURE_RATE', 0))
                right_results[limb]['ACC_ERROR'].append(right_results_batch[limb]['ACC_ERROR'])
                right_results[limb]['CS_MJE'].append(right_results_batch[limb]['CS_MPJPE'])
                right_results[limb]['RR_MJE'].append(right_results_batch[limb]['RR_MPJPE'])
                right_results[limb]['PA_MJE'].append(right_results_batch[limb]['PA_MPJPE'])

                right_results[limb]['INVISIBLE_CS_MJE'].append(right_results_batch[limb].get('INVISIBLE_CS_MPJPE', 0))
                right_results[limb]['INVISIBLE_RR_MJE'].append(right_results_batch[limb].get('INVISIBLE_RR_MPJPE', 0))
                right_results[limb]['INVISIBLE_PA_MJE'].append(right_results_batch[limb].get('INVISIBLE_PA_MPJPE', 0))    
                right_results[limb]['INVISIBLE_ACC_ERROR'].append(right_results_batch[limb].get('INVISIBLE_ACC_ERROR', 0))    

                right_results[limb]['OCC_CS_MJE'].append(right_results_batch[limb].get('OCC_CS_MPJPE', 0))
                right_results[limb]['OCC_RR_MJE'].append(right_results_batch[limb].get('OCC_RR_MPJPE', 0))
                right_results[limb]['OCC_PA_MJE'].append(right_results_batch[limb].get('OCC_PA_MPJPE', 0))  

                right_results[limb]['VIS_CS_MJE'].append(right_results_batch[limb].get('VIS_CS_MPJPE', 0))
                right_results[limb]['VIS_RR_MJE'].append(right_results_batch[limb].get('VIS_RR_MPJPE', 0))
                right_results[limb]['VIS_PA_MJE'].append(right_results_batch[limb].get('VIS_PA_MPJPE', 0))  

                right_results[limb]['OCC_CNT'].append(right_results_batch[limb].get('OCC_CNT', 0))  
                right_results[limb]['VIS_CNT'].append(right_results_batch[limb].get('VIS_CNT', 0))  



            for k in two_hand_results['hand'].keys():
                batch_key = k.replace('MJE', 'MPJPE') if 'MJE' in k else k
                two_hand_results['hand'][k].append(two_hand_batch['hand'][batch_key])

            progress_bar.update(1)

            if current_batch % 10 == 0:
                compute_and_print_metrics(left_results, right_results)
                compute_and_print_metrics_two_hand(two_hand_results)
        
        progress_bar.close()    

        text_single = compute_and_print_metrics(left_results, right_results)
        text_two = compute_and_print_metrics_two_hand(two_hand_results)
        text_str = text_single + "\n" + text_two
    
        with open(save_path, 'w') as f: f.write(text_str)

        metrics_summary = summarize_metrics(left_results, right_results, two_hand_results)
        with open(metrics_summary_path, 'wb') as f:
            pickle.dump(metrics_summary, f)
        return {
            'metrics_summary': metrics_summary,
            'result_text': text_str,
            'result_text_path': save_path,
            'metrics_summary_path': metrics_summary_path,
            'from_cache': False,
        }

    except KeyboardInterrupt:
        print('Stopping evaluation...')


def get_evaluation(args):
    dataset_name = DATASET_NAME
    suffix_base = _resolve_noise_suffix_base(no_cit=args.no_cit)
    undistort_inp = UNDISTORT_INP
    split = 'val'

    noisy_predictions_dir = os.path.abspath(args.noisy_predictions_dir)
    results_dir = os.path.join(os.path.abspath(args.results_dir), f'{dataset_name}_{suffix_base}')
    baseline_noise_percent = float(args.baseline_noise_percent)
    force_recompute = bool(args.force_recompute)

    os.makedirs(noisy_predictions_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    cfg.DATASET.NAME = dataset_name
    dataset = HOT3DLoader(cfg.DATASET.HOT3D_ROOT, get_camera=True, split=split, config=cfg)

    batch_size = max(1, int(args.batch_size))
    n_workers = max(0, int(args.num_workers))

    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': n_workers,
        'pin_memory': True,
    }
    if n_workers > 0:
        loader_kwargs['prefetch_factor'] = 2
        loader_kwargs['persistent_workers'] = True

    left_loader = torch.utils.data.DataLoader(
        Arm3DDataset(
            cfg,
            dataset,
            undistort_inp=undistort_inp,
            return_complete_image=False,
            hand_type='left',
        ),
        **loader_kwargs,
    )
    right_loader = torch.utils.data.DataLoader(
        Arm3DDataset(
            cfg,
            dataset,
            undistort_inp=undistort_inp,
            return_complete_image=False,
            hand_type='right',
        ),
        **loader_kwargs,
    )

    limb_model = LimbModel(cfg, device=device, use_pose_pca=False, n_components=5)

    print('Evaluation settings:')
    print(
        {
            'dataset_name': dataset_name,
            'suffix_base': suffix_base,
            'undistort_inp': undistort_inp,
            'noise_percentages': NOISE_PERCENTAGES,
            'noisy_predictions_dir': noisy_predictions_dir,
            'results_dir': results_dir,
            'no_cit': bool(args.no_cit),
            'ignore_failure_solves': bool(args.ignore_failure_solves),
            'baseline_noise_percent': baseline_noise_percent,
            'force_recompute': force_recompute,
            'batch_size': batch_size,
            'num_workers': n_workers,
        }
    )

    run_entries = []
    for noise_percent in NOISE_PERCENTAGES:
        suffix = f'{suffix_base}_noise_{percent_to_suffix_token(noise_percent)}_percent'
        result_path = os.path.join(noisy_predictions_dir, f'{dataset_name}_{suffix}_predictions.pkl')
        camera_analysis_path = os.path.join(noisy_predictions_dir, f'{dataset_name}_{suffix}_camera_noise_analysis.pkl')

        if not os.path.exists(result_path):
            print(f'Skipping {noise_percent}%: missing predictions file {result_path}')
            continue

        eval_out = None
        if not force_recompute:
            eval_out = load_cached_evaluation(
                noisy_predictions_dir,
                dataset_name,
                suffix,
                sweep_suffix_base=suffix_base,
            )
            if eval_out is not None:
                print(f'Using cached evaluation artifacts for {noise_percent}% ({suffix}).')

        if eval_out is None:
            print(f'Loading predictions from: {result_path}')
            with open(result_path, 'rb') as f:
                results = pickle.load(f)

            eval_out = evaluate_model(
                cfg,
                left_loader,
                right_loader,
                limb_model,
                noisy_predictions_dir,
                device,
                batch_size,
                results,
                suffix,
                force_recompute=force_recompute,
                sweep_suffix_base=suffix_base,
                ignore_failure_solves=args.ignore_failure_solves,
            )

        camera_analysis = None
        camera_features = extract_camera_features(None)
        if os.path.exists(camera_analysis_path):
            with open(camera_analysis_path, 'rb') as f:
                camera_analysis = pickle.load(f)
            camera_features = extract_camera_features(camera_analysis)
        else:
            print(f'Camera analysis not found for {noise_percent}%: {camera_analysis_path}')

        run_entries.append(
            {
                'noise_percent': float(noise_percent),
                'suffix': suffix,
                'result_path': result_path,
                'camera_analysis_path': camera_analysis_path if camera_analysis is not None else None,
                'metrics_summary': eval_out['metrics_summary'],
                'result_text_path': eval_out['result_text_path'],
                'camera_features': camera_features,
            }
        )

    report = build_noise_sweep_report(run_entries, baseline_noise_percent=baseline_noise_percent)
    report_text = report_to_text(report)

    report_txt_path = os.path.join(results_dir, f'{dataset_name}_{suffix_base}_noise_sweep_report.txt')
    with open(report_txt_path, 'w') as f:
        f.write(report_text)

    plot_prefix = os.path.join(results_dir, f'{dataset_name}_{suffix_base}_noise_sweep')
    save_noise_sweep_plots(report, plot_prefix)

    print('\nNoise sweep summary:')
    print(report_text)
    print(f'\nSaved report text to: {report_txt_path}')
    print(f'Saved all evaluation artifacts under: {results_dir}')


def main(argv=None):
    args = _parse_args(argv)
    get_evaluation(args)
 

if __name__ == '__main__':
    main()
