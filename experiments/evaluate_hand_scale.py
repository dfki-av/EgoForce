import argparse
import csv
import glob
import os
import pickle
import re
import sys
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.metrics import compute_similarity_transform_batch, compute_3d_errors_batch, compute_acceleration_error
from core.kalman_filter import KalmanFilterCV3DNP
from utils.plot_utils import (
    _enable_crisp_rendering,
    _stroke_all_text,
    _save_png_supersampled,
    _blend_axes_to_paper,
)

# Requested distance breakpoints (mm). Last bucket is > 10000mm.
DISTANCE_THRESHOLDS_MM = [10, 25, 50, 75, 100, 150, 200, 300, 500, 1000, 10000]
 

@dataclass
class HandSample:
    samplekey: str
    dataset: str
    index: int
    annotation_key: str
    sequence_id: str
    subject_id: str
    frame_id: int
    side: str
    visible: bool
    valid_scale: bool
    gt_j3d_m: np.ndarray
    pred_j3d_m: np.ndarray
    gt_scale_mm: float
    pred_scale_mm: float
    scale_error_mm: float
    scale_error_percent: float
    wrist_distance_mm: float


def _safe_float(v: Any, default: float = np.nan) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = -1) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _last_int_from_text(text: str, default: int = -1) -> int:
    nums = re.findall(r"(\d+)", str(text))
    if not nums:
        return default
    return int(nums[-1])


def _to_joint_array(j: Any) -> np.ndarray:
    arr = np.asarray(j, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        arr = arr.reshape(-1, 3)
    if arr.shape[-1] != 3:
        arr = arr.reshape(-1, 3)
    return arr


def _parse_samplekey(samplekey: str) -> Dict[str, Any]:
    parts = str(samplekey).split("@", 2)
    if len(parts) < 3:
        return {
            "dataset": "UNKNOWN",
            "index": -1,
            "annotation_key": samplekey,
            "sequence_id": samplekey,
            "subject_id": "unknown",
            "frame_id": -1,
        }

    dataset_raw, index_raw, annotation_key = parts
    dataset_upper = dataset_raw.upper()
    index = _safe_int(index_raw, -1)

    if "HOT3D" in dataset_upper:
        sequence_id = annotation_key.split(".")[0]
        subject_id = sequence_id.split("_")[0] if sequence_id else "unknown"
        frame_id = _last_int_from_text(annotation_key, default=index)
    elif "ARCTIC" in dataset_upper or dataset_raw.lower() == "arctic":
        aparts = annotation_key.split("@")
        if len(aparts) >= 5:
            subject_id = aparts[0]
            sequence_id = f"{aparts[0]}_{aparts[1]}"
            frame_id = _safe_int(aparts[2], _last_int_from_text(annotation_key, default=index))
        else:
            sequence_id = annotation_key.split(".")[0]
            subject_id = sequence_id.split("_")[0] if sequence_id else "unknown"
            frame_id = _last_int_from_text(annotation_key, default=index)
    else:
        sequence_id = annotation_key.split("@")[0].split(".")[0]
        subject_id = sequence_id.split("_")[0] if sequence_id else "unknown"
        frame_id = _last_int_from_text(annotation_key, default=index)

    return {
        "dataset": dataset_upper,
        "index": index,
        "annotation_key": annotation_key,
        "sequence_id": sequence_id,
        "subject_id": subject_id,
        "frame_id": frame_id,
    }


def _distance_bin_labels(thresholds: Sequence[float]) -> List[str]:
    labels = []
    prev = 0.0
    for i, t in enumerate(thresholds):
        if i == 0:
            labels.append(f"<= {int(t)}")
        else:
            labels.append(f"{int(prev)}-{int(t)}")
        prev = t
    labels.append(f"> {int(thresholds[-1])}")
    return labels


BIN_LABELS = _distance_bin_labels(DISTANCE_THRESHOLDS_MM)


def _distance_bin_index(dist_mm: float, thresholds: Sequence[float]) -> int:
    for i, t in enumerate(thresholds):
        if dist_mm <= t:
            return i
    return len(thresholds)


def _as_bool(v: Any) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v) > 0
    return bool(v)


def _resolve_prediction_suffix(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "suffix", "")).strip()
    if explicit:
        return explicit

    suffix = "undistort_inp_true"

    if getattr(args, "no_undistort_inp", False):
        suffix = "undistort_inp_false"

    if getattr(args, "no_cit", False):
        suffix += "_no_cit"

    if getattr(args, "no_arm_prior", False):
        suffix += "_no_arm_prior"

    if getattr(args, "no_arm_input", False):
        suffix += "_no_arm_input"

    if getattr(args, "anycalib_624", False):
        suffix += "_anycalib_624"

    if getattr(args, "anycalib_pin", False):
        suffix += "_anycalib_pin"

    if getattr(args, "depth_model", False):
        suffix += "_depth_model"

    if getattr(args, "dgp_model", False):
        suffix += "_DGP_model"

    return suffix


def _resolve_prediction_search_root(log_dir: str) -> str:
    return os.path.abspath(log_dir)


def load_predictions(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Predictions file is not a dict: {path}")
    return data


def extract_samples(predictions: Dict[str, Any]) -> List[HandSample]:
    samples: List[HandSample] = []

    for samplekey, hand_data in predictions.items():
        meta = _parse_samplekey(samplekey)

        for side in ["left", "right"]:
            if side not in hand_data:
                continue
            side_data = hand_data[side]
            if "hand" not in side_data:
                continue

            hand_entry = side_data["hand"]
            gt_j3d_m = _to_joint_array(hand_entry.get("gt_j3d", np.zeros((21, 3), dtype=np.float64)))
            pred_j3d_m = _to_joint_array(hand_entry.get("pred_j3d", np.zeros((21, 3), dtype=np.float64)))

            if gt_j3d_m.shape[0] <= 9 or pred_j3d_m.shape[0] <= 9:
                continue

            gt_scale_mm = float(np.linalg.norm((gt_j3d_m[9] - gt_j3d_m[0]) * 1000.0))
            pred_scale_mm = float(np.linalg.norm((pred_j3d_m[9] - pred_j3d_m[0]) * 1000.0))
            scale_error_mm = abs(pred_scale_mm - gt_scale_mm)
            scale_error_percent = (scale_error_mm / gt_scale_mm * 100.0) if gt_scale_mm > 1e-8 else np.nan
            wrist_distance_mm = float(np.linalg.norm(gt_j3d_m[0] * 1000.0))

            valid_scale = bool(
                np.all(np.isfinite(gt_j3d_m))
                and np.all(np.isfinite(pred_j3d_m))
                and gt_scale_mm > 1e-6
            )

            samples.append(
                HandSample(
                    samplekey=str(samplekey),
                    dataset=meta["dataset"],
                    index=meta["index"],
                    annotation_key=meta["annotation_key"],
                    sequence_id=meta["sequence_id"],
                    subject_id=meta["subject_id"],
                    frame_id=meta["frame_id"],
                    side=side,
                    visible=_as_bool(hand_entry.get("visible", True)),
                    valid_scale=valid_scale,
                    gt_j3d_m=gt_j3d_m,
                    pred_j3d_m=pred_j3d_m,
                    gt_scale_mm=gt_scale_mm,
                    pred_scale_mm=pred_scale_mm,
                    scale_error_mm=scale_error_mm,
                    scale_error_percent=scale_error_percent,
                    wrist_distance_mm=wrist_distance_mm,
                )
            )

    return samples


def _recompute_sample_with_new_prediction(sample: HandSample, pred_j3d_m: np.ndarray) -> HandSample:
    pred_j3d_m = np.asarray(pred_j3d_m, dtype=np.float64)
    gt_j3d_m = np.asarray(sample.gt_j3d_m, dtype=np.float64)

    gt_scale_mm = float(np.linalg.norm((gt_j3d_m[9] - gt_j3d_m[0]) * 1000.0))
    pred_scale_mm = float(np.linalg.norm((pred_j3d_m[9] - pred_j3d_m[0]) * 1000.0))
    scale_error_mm = abs(pred_scale_mm - gt_scale_mm)
    scale_error_percent = (scale_error_mm / gt_scale_mm * 100.0) if gt_scale_mm > 1e-8 else np.nan

    valid_scale = bool(
        np.all(np.isfinite(gt_j3d_m))
        and np.all(np.isfinite(pred_j3d_m))
        and gt_scale_mm > 1e-6
    )

    return HandSample(
        samplekey=sample.samplekey,
        dataset=sample.dataset,
        index=sample.index,
        annotation_key=sample.annotation_key,
        sequence_id=sample.sequence_id,
        subject_id=sample.subject_id,
        frame_id=sample.frame_id,
        side=sample.side,
        visible=sample.visible,
        valid_scale=valid_scale,
        gt_j3d_m=gt_j3d_m,
        pred_j3d_m=pred_j3d_m,
        gt_scale_mm=gt_scale_mm,
        pred_scale_mm=pred_scale_mm,
        scale_error_mm=scale_error_mm,
        scale_error_percent=scale_error_percent,
        wrist_distance_mm=sample.wrist_distance_mm,
    )


def apply_temporal_filter(
    samples: Sequence[HandSample],
    mode: str,
    q_pos: float,
    q_vel: float,
    r_meas: float,
    freq: float,
) -> Tuple[List[HandSample], Dict[str, Any]]:
    mode = str(mode).lower().strip()
    if mode in {"none", "off", "false", "0"}:
        return list(samples), {"enabled": False, "mode": "none"}
    if mode != "kalman_cv":
        raise ValueError(f"Unsupported temporal filter mode: {mode}")

    grouped: Dict[Tuple[str, str, str], List[Tuple[int, HandSample]]] = defaultdict(list)
    for idx, s in enumerate(samples):
        grouped[(s.dataset, s.sequence_id, s.side)].append((idx, s))

    filtered_samples: List[Optional[HandSample]] = [None] * len(samples)
    delta_norms_mm: List[float] = []
    processed_groups = 0
    processed_frames = 0

    for _, idx_seq in grouped.items():
        seq_sorted = sorted(idx_seq, key=lambda x: (x[1].frame_id, x[1].index, x[0]))
        kf = KalmanFilterCV3DNP(q_pos=q_pos, q_vel=q_vel, r_meas=r_meas, freq=freq)
        kf.reset_state()
        processed_groups += 1

        for idx, s in seq_sorted:
            pred = np.asarray(s.pred_j3d_m, dtype=np.float64).copy()
            if pred.ndim != 2 or pred.shape[0] == 0:
                filtered_samples[idx] = s
                continue

            raw_transl = pred[0].copy()
            filt_transl = kf.step(raw_transl, s.visible)
            delta = filt_transl - raw_transl
            pred = pred + delta[None, :]

            delta_norms_mm.append(float(np.linalg.norm(delta * 1000.0)))
            processed_frames += 1
            filtered_samples[idx] = _recompute_sample_with_new_prediction(s, pred)

    # Fallback for untouched entries.
    out: List[HandSample] = []
    for i, s in enumerate(filtered_samples):
        out.append(samples[i] if s is None else s)

    summary = {
        "enabled": True,
        "mode": "kalman_cv",
        "q_pos": float(q_pos),
        "q_vel": float(q_vel),
        "r_meas": float(r_meas),
        "freq": float(freq),
        "num_sequence_side_groups": int(processed_groups),
        "num_frames_filtered": int(processed_frames),
        "mean_translation_shift_mm": float(np.mean(delta_norms_mm)) if delta_norms_mm else np.nan,
        "p95_translation_shift_mm": float(np.percentile(delta_norms_mm, 95)) if delta_norms_mm else np.nan,
    }
    return out, summary


def _mean(v: Iterable[float]) -> float:
    arr = np.asarray(list(v), dtype=np.float64)
    if arr.size == 0:
        return np.nan
    return float(np.nanmean(arr))


def _median(v: Iterable[float]) -> float:
    arr = np.asarray(list(v), dtype=np.float64)
    if arr.size == 0:
        return np.nan
    return float(np.nanmedian(arr))


def _std(v: Iterable[float]) -> float:
    arr = np.asarray(list(v), dtype=np.float64)
    if arr.size == 0:
        return np.nan
    return float(np.nanstd(arr))


def aggregate_scale_by_distance(samples: Sequence[HandSample]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, int], List[HandSample]] = defaultdict(list)

    for s in samples:
        if not (s.visible and s.valid_scale):
            continue
        bin_idx = _distance_bin_index(s.wrist_distance_mm, DISTANCE_THRESHOLDS_MM)
        grouped[(s.dataset, s.side, bin_idx)].append(s)

    rows: List[Dict[str, Any]] = []
    for (dataset, side, bin_idx), seq in sorted(grouped.items()):
        rows.append(
            {
                "dataset": dataset,
                "side": side,
                "bin_index": bin_idx,
                "bin_label": BIN_LABELS[bin_idx],
                "count": len(seq),
                "mean_abs_scale_error_mm": _mean(x.scale_error_mm for x in seq),
                "std_abs_scale_error_mm": _std(x.scale_error_mm for x in seq),
                "median_abs_scale_error_mm": _median(x.scale_error_mm for x in seq),
                "mean_rel_scale_error_percent": _mean(x.scale_error_percent for x in seq),
                "std_rel_scale_error_percent": _std(x.scale_error_percent for x in seq),
                "mean_gt_scale_mm": _mean(x.gt_scale_mm for x in seq),
                "mean_pred_scale_mm": _mean(x.pred_scale_mm for x in seq),
                "mean_wrist_distance_mm": _mean(x.wrist_distance_mm for x in seq),
            }
        )

    # Add combined mean over left/right.
    mean_grouped: Dict[Tuple[str, int], List[HandSample]] = defaultdict(list)
    for s in samples:
        if not (s.visible and s.valid_scale):
            continue
        bin_idx = _distance_bin_index(s.wrist_distance_mm, DISTANCE_THRESHOLDS_MM)
        mean_grouped[(s.dataset, bin_idx)].append(s)

    for (dataset, bin_idx), seq in sorted(mean_grouped.items()):
        rows.append(
            {
                "dataset": dataset,
                "side": "mean",
                "bin_index": bin_idx,
                "bin_label": BIN_LABELS[bin_idx],
                "count": len(seq),
                "mean_abs_scale_error_mm": _mean(x.scale_error_mm for x in seq),
                "std_abs_scale_error_mm": _std(x.scale_error_mm for x in seq),
                "median_abs_scale_error_mm": _median(x.scale_error_mm for x in seq),
                "mean_rel_scale_error_percent": _mean(x.scale_error_percent for x in seq),
                "std_rel_scale_error_percent": _std(x.scale_error_percent for x in seq),
                "mean_gt_scale_mm": _mean(x.gt_scale_mm for x in seq),
                "mean_pred_scale_mm": _mean(x.pred_scale_mm for x in seq),
                "mean_wrist_distance_mm": _mean(x.wrist_distance_mm for x in seq),
            }
        )

    return rows


def aggregate_scale_by_sequence(samples: Sequence[HandSample]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str], List[HandSample]] = defaultdict(list)
    for s in samples:
        if not (s.visible and s.valid_scale):
            continue
        grouped[(s.dataset, s.subject_id, s.sequence_id, s.side)].append(s)

    rows: List[Dict[str, Any]] = []
    for (dataset, subject_id, sequence_id, side), seq in sorted(grouped.items()):
        seq_sorted = sorted(seq, key=lambda x: x.frame_id)
        rows.append(
            {
                "dataset": dataset,
                "subject_id": subject_id,
                "sequence_id": sequence_id,
                "side": side,
                "count": len(seq_sorted),
                "first_frame": seq_sorted[0].frame_id,
                "last_frame": seq_sorted[-1].frame_id,
                "mean_abs_scale_error_mm": _mean(x.scale_error_mm for x in seq_sorted),
                "std_abs_scale_error_mm": _std(x.scale_error_mm for x in seq_sorted),
                "median_abs_scale_error_mm": _median(x.scale_error_mm for x in seq_sorted),
                "mean_rel_scale_error_percent": _mean(x.scale_error_percent for x in seq_sorted),
                "std_rel_scale_error_percent": _std(x.scale_error_percent for x in seq_sorted),
                "mean_gt_scale_mm": _mean(x.gt_scale_mm for x in seq_sorted),
                "mean_pred_scale_mm": _mean(x.pred_scale_mm for x in seq_sorted),
                "mean_wrist_distance_mm": _mean(x.wrist_distance_mm for x in seq_sorted),
            }
        )

    return rows


def compute_hand_metrics(
    gt_j3d_m: np.ndarray,
    pred_j3d_m: np.ndarray,
    valid_mask: np.ndarray,
    fps: float,
    acc_threshold_mm: float,
    ignore_failure_solves: bool = True,
) -> Dict[str, float]:
    valid_mask = np.asarray(valid_mask, dtype=bool)
    gt_j3d_m = np.asarray(gt_j3d_m, dtype=np.float64)
    pred_j3d_m = np.asarray(pred_j3d_m, dtype=np.float64)

    gt_j3d_mm = gt_j3d_m * 1000.0
    pred_j3d_mm = pred_j3d_m * 1000.0

    cs_joint, rr_joint, pa_joint = compute_3d_errors_batch(gt_j3d_mm, pred_j3d_mm, valid_mask, root_joint=0)
    acc_joint = compute_acceleration_error(gt_j3d_m, pred_j3d_m, valid_mask, fps=fps)

    # remove CS outlier joints (>1000 mm) and apply same mask to ACC.
    failure_cases = cs_joint > 1000
    failure_rate = float(np.mean(failure_cases.astype(np.float64)) * 100.0) if cs_joint.size else np.nan
    if ignore_failure_solves:
        cs_joint_filtered = cs_joint[~failure_cases]
        acc_joint_filtered = acc_joint[~failure_cases]
    else:
        cs_joint_filtered = cs_joint
        acc_joint_filtered = acc_joint

    if cs_joint_filtered.size == 0:
        cs_joint_filtered = np.array([0.0], dtype=np.float64)
        acc_joint_filtered = np.array([0.0], dtype=np.float64)

    cs_mpjpe = float(np.mean(cs_joint_filtered)) if cs_joint_filtered.size else np.nan
    rr_mpjpe = float(np.mean(rr_joint)) if rr_joint.size else np.nan
    pa_mpjpe = float(np.mean(pa_joint)) if pa_joint.size else np.nan
    # Acceleration error is reported in m/s^2.
    acc_error = float(np.mean(acc_joint_filtered)) if acc_joint_filtered.size else np.nan

    if np.any(valid_mask):
        gt_v = gt_j3d_mm[valid_mask]
        pred_v = pred_j3d_mm[valid_mask]

        err_cs = np.linalg.norm(gt_v - pred_v, axis=-1)
        err_rr = np.linalg.norm(
            (gt_v - gt_v[:, :1, :]) - (pred_v - pred_v[:, :1, :]),
            axis=-1,
        )

        try:
            pred_pa, _ = compute_similarity_transform_batch(pred_v, gt_v)
        except Exception:
            pred_pa = pred_v
        err_pa = np.linalg.norm(gt_v - pred_pa, axis=-1)

        cs_acc = float(np.mean(err_cs <= acc_threshold_mm) * 100.0)
        rr_acc = float(np.mean(err_rr <= acc_threshold_mm) * 100.0)
        pa_acc = float(np.mean(err_pa <= acc_threshold_mm) * 100.0)
    else:
        cs_acc = np.nan
        rr_acc = np.nan
        pa_acc = np.nan

    return {
        "CS_MPJPE": cs_mpjpe,
        "RR_MPJPE": rr_mpjpe,
        "PA_MPJPE": pa_mpjpe,
        "ACC_ERROR": acc_error,
        "FAILURE_RATE": failure_rate,
        "CS_ACC": cs_acc,
        "RR_ACC": rr_acc,
        "PA_ACC": pa_acc,
    }


def _safe_percent_improvement(before: float, after: float) -> float:
    if not (np.isfinite(before) and np.isfinite(after)):
        return np.nan
    if abs(before) < 1e-9:
        return np.nan
    return float((before - after) / abs(before) * 100.0)


def _apply_scale_factor_to_pred(pred_j3d_m: np.ndarray, scale_factor: float) -> np.ndarray:
    pred = np.asarray(pred_j3d_m, dtype=np.float64)
    return (pred - pred[:1]) * float(scale_factor) + pred[:1]


def _aggregate_batch_metrics(metric_rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not metric_rows:
        return {}
    keys = sorted(set().union(*(row.keys() for row in metric_rows)))
    out: Dict[str, float] = {}
    for k in keys:
        vals = np.asarray([_safe_float(row.get(k, np.nan)) for row in metric_rows], dtype=np.float64)
        out[k] = float(np.nanmean(vals)) if vals.size else np.nan
    return out


def _compute_dataset_overall_eval_ours_style(
    samples: Sequence[HandSample],
    dataset_name: str,
    seq_scale_factors: Dict[Tuple[str, str], float],
    fps: float,
    acc_threshold_mm: float,
    eval_batch_size: int,
    ignore_failure_solves: bool,
) -> List[Dict[str, Any]]:
    eval_batch_size = max(1, int(eval_batch_size))
    overall_rows: List[Dict[str, Any]] = []
    dataset_name = str(dataset_name).upper()

    per_side_rows: Dict[str, Dict[str, Any]] = {}
    for side in ["left", "right"]:
        side_samples = [s for s in samples if s.dataset == dataset_name and s.side == side]
        if not side_samples:
            continue

        before_batches: List[Dict[str, float]] = []
        after_batches: List[Dict[str, float]] = []
        all_before_err_mm: List[float] = []
        all_after_err_mm: List[float] = []
        all_before_rel_err_percent: List[float] = []
        all_after_rel_err_percent: List[float] = []
        total_valid_frames = 0

        for start in range(0, len(side_samples), eval_batch_size):
            batch = side_samples[start : start + eval_batch_size]
            gt_arr = np.asarray([x.gt_j3d_m for x in batch], dtype=np.float64)
            pred_before = np.asarray([x.pred_j3d_m for x in batch], dtype=np.float64)
            pred_after = np.asarray(
                [
                    _apply_scale_factor_to_pred(
                        x.pred_j3d_m,
                        seq_scale_factors.get((x.sequence_id, x.side), 1.0),
                    )
                    for x in batch
                ],
                dtype=np.float64,
            )
            valid_mask = np.asarray([x.visible and x.valid_scale for x in batch], dtype=bool)

            total_valid_frames += int(np.sum(valid_mask))
            before_batches.append(
                compute_hand_metrics(
                    gt_arr,
                    pred_before,
                    valid_mask,
                    fps=fps,
                    acc_threshold_mm=acc_threshold_mm,
                    ignore_failure_solves=ignore_failure_solves,
                )
            )
            after_batches.append(
                compute_hand_metrics(
                    gt_arr,
                    pred_after,
                    valid_mask,
                    fps=fps,
                    acc_threshold_mm=acc_threshold_mm,
                    ignore_failure_solves=ignore_failure_solves,
                )
            )

            for x, p_after in zip(batch, pred_after):
                all_before_err_mm.append(float(x.scale_error_mm))
                all_before_rel_err_percent.append(float(x.scale_error_percent))
                gt_scale = float(x.gt_scale_mm)
                pred_scale_after = float(np.linalg.norm((p_after[9] - p_after[0]) * 1000.0))
                after_err_mm = abs(pred_scale_after - gt_scale)
                all_after_err_mm.append(after_err_mm)
                all_after_rel_err_percent.append(
                    (after_err_mm / gt_scale * 100.0) if gt_scale > 1e-8 else np.nan
                )

        before = _aggregate_batch_metrics(before_batches)
        after = _aggregate_batch_metrics(after_batches)

        mean_scale_factor = _mean(seq_scale_factors.get((x.sequence_id, x.side), 1.0) for x in side_samples)
        row: Dict[str, Any] = {
            "side": side,
            "num_batches": len(before_batches),
            "total_frames": len(side_samples),
            "total_valid_frames": total_valid_frames,
            "mean_scale_factor": mean_scale_factor,
            "mean_abs_scale_error_mm_before": _mean(all_before_err_mm),
            "mean_abs_scale_error_mm_after": _mean(all_after_err_mm),
            "std_abs_scale_error_mm_before": _std(all_before_err_mm),
            "std_abs_scale_error_mm_after": _std(all_after_err_mm),
            "mean_rel_scale_error_percent_before": _mean(all_before_rel_err_percent),
            "mean_rel_scale_error_percent_after": _mean(all_after_rel_err_percent),
            "std_rel_scale_error_percent_before": _std(all_before_rel_err_percent),
            "std_rel_scale_error_percent_after": _std(all_after_rel_err_percent),
        }
        row["mean_abs_scale_error_mm_improvement_percent"] = _safe_percent_improvement(
            row["mean_abs_scale_error_mm_before"], row["mean_abs_scale_error_mm_after"]
        )
        row["std_abs_scale_error_mm_improvement_percent"] = _safe_percent_improvement(
            row["std_abs_scale_error_mm_before"], row["std_abs_scale_error_mm_after"]
        )
        row["mean_rel_scale_error_percent_improvement_percent"] = _safe_percent_improvement(
            row["mean_rel_scale_error_percent_before"], row["mean_rel_scale_error_percent_after"]
        )
        row["std_rel_scale_error_percent_improvement_percent"] = _safe_percent_improvement(
            row["std_rel_scale_error_percent_before"], row["std_rel_scale_error_percent_after"]
        )

        for k, v in before.items():
            row[f"{k}_before"] = v
        for k, v in after.items():
            row[f"{k}_after"] = v
            row[f"{k}_improvement_percent"] = _safe_percent_improvement(before.get(k, np.nan), v)

        overall_rows.append(row)
        per_side_rows[side] = row

    if "left" in per_side_rows and "right" in per_side_rows:
        left = per_side_rows["left"]
        right = per_side_rows["right"]
        mean_row: Dict[str, Any] = {
            "side": "mean",
            "num_batches": int(left.get("num_batches", 0)) + int(right.get("num_batches", 0)),
            "total_frames": int(left.get("total_frames", 0)) + int(right.get("total_frames", 0)),
            "total_valid_frames": int(left.get("total_valid_frames", 0)) + int(right.get("total_valid_frames", 0)),
            "mean_scale_factor": (_safe_float(left.get("mean_scale_factor")) + _safe_float(right.get("mean_scale_factor"))) / 2.0,
        }

        numeric_keys = sorted(set(k for k in left.keys() if isinstance(left[k], (int, float, np.floating, np.integer))))
        for k in numeric_keys:
            if k in {"num_batches", "total_frames", "total_valid_frames", "side"}:
                continue
            if k in right:
                mean_row[k] = (_safe_float(left[k]) + _safe_float(right[k])) / 2.0
        overall_rows.append(mean_row)

    return overall_rows


def calibrate_dataset_sequences(
    samples: Sequence[HandSample],
    dataset_name: str,
    n_calibration_samples: int,
    fps: float,
    acc_threshold_mm: float,
    eval_batch_size: int,
    ignore_failure_solves: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[Tuple[str, str], float]]:
    dataset_name = str(dataset_name).upper()
    ds_samples = [s for s in samples if s.dataset == dataset_name and s.valid_scale]
    grouped: Dict[Tuple[str, str], List[HandSample]] = defaultdict(list)
    for s in ds_samples:
        grouped[(s.sequence_id, s.side)].append(s)

    seq_rows: List[Dict[str, Any]] = []
    seq_scale_factors: Dict[Tuple[str, str], float] = {}

    for (sequence_id, side), seq_samples in sorted(grouped.items()):
        seq_sorted = sorted(seq_samples, key=lambda x: x.frame_id)
        valid_for_calib = [x for x in seq_sorted if x.visible and x.pred_scale_mm > 1e-8 and x.gt_scale_mm > 1e-8]

        if not valid_for_calib:
            continue

        calib_subset = valid_for_calib
        if n_calibration_samples > 0:
            calib_subset = valid_for_calib[: min(n_calibration_samples, len(valid_for_calib))]

        ratios = np.asarray([x.gt_scale_mm / x.pred_scale_mm for x in calib_subset], dtype=np.float64)
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size == 0:
            continue

        scale_factor = float(np.mean(ratios))
        seq_scale_factors[(sequence_id, side)] = scale_factor

        gt_arr = np.asarray([x.gt_j3d_m for x in seq_sorted], dtype=np.float64)
        pred_arr = np.asarray([x.pred_j3d_m for x in seq_sorted], dtype=np.float64)
        valid_mask = np.asarray([x.visible and x.valid_scale for x in seq_sorted], dtype=bool)

        pred_calib = (pred_arr - pred_arr[:, :1, :]) * scale_factor + pred_arr[:, :1, :]

        before = compute_hand_metrics(
            gt_arr,
            pred_arr,
            valid_mask,
            fps=fps,
            acc_threshold_mm=acc_threshold_mm,
            ignore_failure_solves=ignore_failure_solves,
        )
        after = compute_hand_metrics(
            gt_arr,
            pred_calib,
            valid_mask,
            fps=fps,
            acc_threshold_mm=acc_threshold_mm,
            ignore_failure_solves=ignore_failure_solves,
        )

        row = {
            "sequence_id": sequence_id,
            "subject_id": seq_sorted[0].subject_id,
            "side": side,
            "valid_frames": int(np.sum(valid_mask)),
            "total_frames": len(seq_sorted),
            "calibration_samples_used": len(calib_subset),
            "scale_factor": scale_factor,
            "mean_gt_scale_mm": _mean(x.gt_scale_mm for x in seq_sorted),
            "mean_pred_scale_mm_before": _mean(x.pred_scale_mm for x in seq_sorted),
            "mean_pred_scale_mm_after": _mean(x.pred_scale_mm * scale_factor for x in seq_sorted),
            "mean_abs_scale_error_mm_before": _mean(x.scale_error_mm for x in seq_sorted),
            "mean_abs_scale_error_mm_after": _mean(abs(x.pred_scale_mm * scale_factor - x.gt_scale_mm) for x in seq_sorted),
            "std_abs_scale_error_mm_before": _std(x.scale_error_mm for x in seq_sorted),
            "std_abs_scale_error_mm_after": _std(
                abs(x.pred_scale_mm * scale_factor - x.gt_scale_mm) for x in seq_sorted
            ),
            "mean_rel_scale_error_percent_before": _mean(x.scale_error_percent for x in seq_sorted),
            "mean_rel_scale_error_percent_after": _mean(
                abs(x.pred_scale_mm * scale_factor - x.gt_scale_mm) / x.gt_scale_mm * 100.0 if x.gt_scale_mm > 1e-8 else np.nan
                for x in seq_sorted
            ),
            "std_rel_scale_error_percent_before": _std(x.scale_error_percent for x in seq_sorted),
            "std_rel_scale_error_percent_after": _std(
                abs(x.pred_scale_mm * scale_factor - x.gt_scale_mm) / x.gt_scale_mm * 100.0
                if x.gt_scale_mm > 1e-8
                else np.nan
                for x in seq_sorted
            ),
        }
        row["mean_abs_scale_error_mm_improvement_percent"] = _safe_percent_improvement(
            row["mean_abs_scale_error_mm_before"], row["mean_abs_scale_error_mm_after"]
        )
        row["std_abs_scale_error_mm_improvement_percent"] = _safe_percent_improvement(
            row["std_abs_scale_error_mm_before"], row["std_abs_scale_error_mm_after"]
        )
        row["mean_rel_scale_error_percent_improvement_percent"] = _safe_percent_improvement(
            row["mean_rel_scale_error_percent_before"], row["mean_rel_scale_error_percent_after"]
        )
        row["std_rel_scale_error_percent_improvement_percent"] = _safe_percent_improvement(
            row["std_rel_scale_error_percent_before"], row["std_rel_scale_error_percent_after"]
        )

        for k, v in before.items():
            row[f"{k}_before"] = v
        for k, v in after.items():
            row[f"{k}_after"] = v
            row[f"{k}_improvement_percent"] = _safe_percent_improvement(before[k], v)

        seq_rows.append(row)

    # Sequence-level mean over both hands.
    by_seq: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in seq_rows:
        by_seq[row["sequence_id"]].append(row)

    for sequence_id, rows in sorted(by_seq.items()):
        if len(rows) < 2:
            continue
        w = np.asarray([max(1, int(r["valid_frames"])) for r in rows], dtype=np.float64)
        w = w / np.sum(w)

        mean_row = {
            "sequence_id": sequence_id,
            "subject_id": rows[0]["subject_id"],
            "side": "mean",
            "valid_frames": int(np.sum([r["valid_frames"] for r in rows])),
            "total_frames": int(np.sum([r["total_frames"] for r in rows])),
            "calibration_samples_used": int(np.sum([r["calibration_samples_used"] for r in rows])),
            "scale_factor": float(np.sum([r["scale_factor"] * wi for r, wi in zip(rows, w)])),
        }

        metric_keys = [
            "mean_gt_scale_mm",
            "mean_pred_scale_mm_before",
            "mean_pred_scale_mm_after",
            "mean_abs_scale_error_mm_before",
            "mean_abs_scale_error_mm_after",
            "std_abs_scale_error_mm_before",
            "std_abs_scale_error_mm_after",
            "mean_abs_scale_error_mm_improvement_percent",
            "std_abs_scale_error_mm_improvement_percent",
            "mean_rel_scale_error_percent_before",
            "mean_rel_scale_error_percent_after",
            "std_rel_scale_error_percent_before",
            "std_rel_scale_error_percent_after",
            "mean_rel_scale_error_percent_improvement_percent",
            "std_rel_scale_error_percent_improvement_percent",
            "CS_MPJPE_before",
            "CS_MPJPE_after",
            "CS_MPJPE_improvement_percent",
            "RR_MPJPE_before",
            "RR_MPJPE_after",
            "RR_MPJPE_improvement_percent",
            "PA_MPJPE_before",
            "PA_MPJPE_after",
            "PA_MPJPE_improvement_percent",
            "ACC_ERROR_before",
            "ACC_ERROR_after",
            "ACC_ERROR_improvement_percent",
            "CS_ACC_before",
            "CS_ACC_after",
            "CS_ACC_improvement_percent",
            "RR_ACC_before",
            "RR_ACC_after",
            "RR_ACC_improvement_percent",
            "PA_ACC_before",
            "PA_ACC_after",
            "PA_ACC_improvement_percent",
        ]
        for k in metric_keys:
            mean_row[k] = float(np.sum([_safe_float(r.get(k, np.nan)) * wi for r, wi in zip(rows, w)]))

        seq_rows.append(mean_row)

    overall_rows = _compute_dataset_overall_eval_ours_style(
        samples=samples,
        dataset_name=dataset_name,
        seq_scale_factors=seq_scale_factors,
        fps=fps,
        acc_threshold_mm=acc_threshold_mm,
        eval_batch_size=eval_batch_size,
        ignore_failure_solves=ignore_failure_solves,
    )

    return seq_rows, overall_rows, seq_scale_factors


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _round_float(v: float, decimals: int = 1) -> float:
    if not np.isfinite(v):
        return float(v)
    return float(np.round(v, decimals))


def round_rows_for_artifacts(rows: Sequence[Dict[str, Any]], decimals: int = 1) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        nr: Dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, (np.integer, int)):
                nr[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                nr[k] = _round_float(float(v), decimals=decimals)
            else:
                nr[k] = v
        out.append(nr)
    return out


def round_nested_for_artifacts(obj: Any, decimals: int = 1) -> Any:
    if isinstance(obj, dict):
        return {k: round_nested_for_artifacts(v, decimals=decimals) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_nested_for_artifacts(v, decimals=decimals) for v in obj]
    if isinstance(obj, tuple):
        return tuple(round_nested_for_artifacts(v, decimals=decimals) for v in obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return _round_float(float(obj), decimals=decimals)
    return obj


def _plot_or_warn(msg: str) -> Optional[Any]:
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:
        print(f"{msg}: matplotlib unavailable ({exc})")
        return None


def _safe_call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    if fn is None:
        return None
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _style_and_save_plot(fig: Any, axes: Any, save_png_path: str) -> None:
    _safe_call(_enable_crisp_rendering)

    axis_list: List[Any] = []
    if isinstance(axes, np.ndarray):
        axis_list = [ax for ax in axes.ravel().tolist() if ax is not None]
    elif isinstance(axes, (list, tuple)):
        axis_list = [ax for ax in axes if ax is not None]
    elif axes is not None:
        axis_list = [axes]

    for ax in axis_list:
        _safe_call(_blend_axes_to_paper, ax)

    _safe_call(_stroke_all_text, fig, lw=0.8, fg="white")

    os.makedirs(os.path.dirname(save_png_path), exist_ok=True)
    if _save_png_supersampled is not None:
        try:
            _save_png_supersampled(fig, save_png_path, scale=3)
            return
        except Exception:
            pass
    fig.savefig(save_png_path, bbox_inches="tight", dpi=220)


def plot_distance_profiles(distance_rows: Sequence[Dict[str, Any]], output_dir: str) -> None:
    plt = _plot_or_warn("Skipping distance profile plots")
    if plt is None:
        return

    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in distance_rows:
        by_dataset[str(r["dataset"])].append(r)

    x = np.arange(len(BIN_LABELS))

    for dataset, rows in by_dataset.items():
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        for side, style in [("left", "o-"), ("right", "s-"), ("mean", "^-" )]:
            ys = np.full(len(BIN_LABELS), np.nan, dtype=np.float64)
            for r in rows:
                if r["side"] != side:
                    continue
                ys[int(r["bin_index"])] = _safe_float(r["mean_abs_scale_error_mm"])
            if np.any(np.isfinite(ys)):
                ax.plot(x, ys, style, label=side)

        ax.set_xticks(x)
        ax.set_xticklabels(BIN_LABELS, rotation=35, ha="right")
        ax.set_ylabel("Mean absolute hand-scale error (mm)")
        ax.set_xlabel("Wrist distance from camera origin (mm)")
        ax.set_title(f"{dataset} hand-scale error vs hand distance")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        _style_and_save_plot(
            fig,
            ax,
            os.path.join(output_dir, f"{dataset}_scale_error_vs_distance_bins.png"),
        )
        plt.close(fig)


def plot_scale_scatter(samples: Sequence[HandSample], output_dir: str) -> None:
    plt = _plot_or_warn("Skipping scatter plots")
    if plt is None:
        return

    by_dataset: Dict[str, List[HandSample]] = defaultdict(list)
    for s in samples:
        if s.visible and s.valid_scale:
            by_dataset[s.dataset].append(s)

    for dataset, seq in by_dataset.items():
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for side, color in [("left", "tab:blue"), ("right", "tab:orange")]:
            cur = [x for x in seq if x.side == side]
            if not cur:
                continue
            x = np.asarray([x.wrist_distance_mm for x in cur], dtype=np.float64)
            y = np.asarray([x.scale_error_mm for x in cur], dtype=np.float64)
            ax.scatter(x, y, s=9, alpha=0.28, label=side, color=color)

        ax.set_xscale("log")
        ax.set_xlabel("Wrist distance from camera origin (mm, log scale)")
        ax.set_ylabel("Absolute hand-scale error (mm)")
        ax.set_title(f"{dataset} per-frame hand-scale error distribution")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        _style_and_save_plot(
            fig,
            ax,
            os.path.join(output_dir, f"{dataset}_scale_error_scatter.png"),
        )
        plt.close(fig)


def plot_sequence_scale_errors(sequence_rows: Sequence[Dict[str, Any]], output_dir: str) -> None:
    plt = _plot_or_warn("Skipping sequence plots")
    if plt is None:
        return

    rows = [r for r in sequence_rows if r.get("side") == "mean"]
    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_dataset[str(r["dataset"])].append(r)

    for dataset, seq in by_dataset.items():
        seq = sorted(seq, key=lambda x: _safe_float(x.get("mean_abs_scale_error_mm", np.nan)), reverse=True)
        if not seq:
            continue
        top_n = min(40, len(seq))
        cur = seq[:top_n]

        fig, ax = plt.subplots(1, 1, figsize=(15, 7))
        vals = [_safe_float(x["mean_abs_scale_error_mm"]) for x in cur]
        labels = [str(x["sequence_id"]) for x in cur]
        ax.bar(np.arange(top_n), vals, color="tab:green", alpha=0.8)
        ax.set_xticks(np.arange(top_n))
        ax.set_xticklabels(labels, rotation=70, ha="right")
        ax.set_ylabel("Mean absolute hand-scale error (mm)")
        ax.set_title(f"{dataset} sequence-level hand-scale error (top {top_n})")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        _style_and_save_plot(
            fig,
            ax,
            os.path.join(output_dir, f"{dataset}_sequence_scale_error_top{top_n}.png"),
        )
        plt.close(fig)


def plot_dataset_calibration(
    dataset_name: str,
    seq_rows: Sequence[Dict[str, Any]],
    overall_rows: Sequence[Dict[str, Any]],
    output_dir: str,
) -> None:
    dataset_name = str(dataset_name).upper()
    if not seq_rows and not overall_rows:
        return

    plt = _plot_or_warn(f"Skipping {dataset_name} calibration plots")
    if plt is None:
        return

    if overall_rows:
        side_order = ["left", "right", "mean"]
        metric_names = ["CS_MPJPE", "RR_MPJPE", "PA_MPJPE", "ACC_ERROR", "CS_ACC", "RR_ACC", "PA_ACC"]

        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        axes = axes.flatten()

        for i, metric in enumerate(metric_names):
            ax = axes[i]
            before = []
            after = []
            for side in side_order:
                row = next((x for x in overall_rows if x["side"] == side), None)
                before.append(_safe_float(row.get(f"{metric}_before", np.nan)) if row else np.nan)
                after.append(_safe_float(row.get(f"{metric}_after", np.nan)) if row else np.nan)

            xx = np.arange(len(side_order))
            w = 0.38
            ax.bar(xx - w / 2, before, width=w, label="before", color="tab:red", alpha=0.8)
            ax.bar(xx + w / 2, after, width=w, label="after", color="tab:blue", alpha=0.8)
            ax.set_xticks(xx)
            ax.set_xticklabels(side_order)
            ax.set_title(metric)
            ax.grid(True, axis="y", alpha=0.25)

        if len(metric_names) < len(axes):
            for j in range(len(metric_names), len(axes)):
                axes[j].axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.suptitle(f"{dataset_name} sequence-wise scale calibration: before vs after", y=1.02)
        fig.tight_layout()
        _style_and_save_plot(
            fig,
            axes,
            os.path.join(output_dir, f"{dataset_name}_calibration_overall_before_after.png"),
        )
        plt.close(fig)

    side_rows = [r for r in seq_rows if r.get("side") in {"left", "right"}]
    if side_rows:
        # Scale factor histogram.
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        factors = np.asarray([_safe_float(r.get("scale_factor", np.nan)) for r in side_rows], dtype=np.float64)
        factors = factors[np.isfinite(factors)]
        if factors.size:
            ax.hist(factors, bins=30, color="tab:purple", alpha=0.8)
        ax.set_xlabel("Per-sequence calibration scale factor")
        ax.set_ylabel("Count")
        ax.set_title(f"{dataset_name} calibration scale-factor distribution")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        _style_and_save_plot(
            fig,
            ax,
            os.path.join(output_dir, f"{dataset_name}_calibration_scale_factor_hist.png"),
        )
        plt.close(fig)

        # CS MPJPE improvement by sequence.
        seq_mean_rows = [r for r in seq_rows if r.get("side") == "mean"]
        seq_mean_rows = sorted(
            seq_mean_rows,
            key=lambda x: _safe_float(x.get("CS_MPJPE_improvement_percent", np.nan)),
            reverse=True,
        )
        if seq_mean_rows:
            top_n = min(50, len(seq_mean_rows))
            cur = seq_mean_rows[:top_n]
            fig, ax = plt.subplots(1, 1, figsize=(17, 7))
            vals = [_safe_float(x.get("CS_MPJPE_improvement_percent", np.nan)) for x in cur]
            labs = [str(x.get("sequence_id", "")) for x in cur]
            ax.bar(np.arange(top_n), vals, color="tab:cyan", alpha=0.85)
            ax.set_xticks(np.arange(top_n))
            ax.set_xticklabels(labs, rotation=70, ha="right")
            ax.set_ylabel("CS_MPJPE improvement (%)")
            ax.set_title(f"{dataset_name} sequence-wise calibration improvement (top {top_n})")
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            _style_and_save_plot(
                fig,
                ax,
                os.path.join(output_dir, f"{dataset_name}_calibration_cs_improvement_top.png"),
            )
            plt.close(fig)


def rows_from_samples(samples: Sequence[HandSample]) -> List[Dict[str, Any]]:
    rows = []
    for s in samples:
        rows.append(
            {
                "samplekey": s.samplekey,
                "dataset": s.dataset,
                "index": s.index,
                "annotation_key": s.annotation_key,
                "subject_id": s.subject_id,
                "sequence_id": s.sequence_id,
                "frame_id": s.frame_id,
                "side": s.side,
                "visible": int(s.visible),
                "valid_scale": int(s.valid_scale),
                "wrist_distance_mm": s.wrist_distance_mm,
                "gt_scale_mm": s.gt_scale_mm,
                "pred_scale_mm": s.pred_scale_mm,
                "abs_scale_error_mm": s.scale_error_mm,
                "rel_scale_error_percent": s.scale_error_percent,
                "distance_bin_index": _distance_bin_index(s.wrist_distance_mm, DISTANCE_THRESHOLDS_MM),
                "distance_bin_label": BIN_LABELS[_distance_bin_index(s.wrist_distance_mm, DISTANCE_THRESHOLDS_MM)],
            }
        )
    return rows


def aggregate_overall_scale(samples: Sequence[HandSample]) -> List[Dict[str, Any]]:
    rows = []

    def _emit(dataset: str, side: str, seq: List[HandSample]) -> None:
        rows.append(
            {
                "dataset": dataset,
                "side": side,
                "count": len(seq),
                "mean_abs_scale_error_mm": _mean(x.scale_error_mm for x in seq),
                "std_abs_scale_error_mm": _std(x.scale_error_mm for x in seq),
                "median_abs_scale_error_mm": _median(x.scale_error_mm for x in seq),
                "mean_rel_scale_error_percent": _mean(x.scale_error_percent for x in seq),
                "std_rel_scale_error_percent": _std(x.scale_error_percent for x in seq),
                "mean_gt_scale_mm": _mean(x.gt_scale_mm for x in seq),
                "mean_pred_scale_mm": _mean(x.pred_scale_mm for x in seq),
                "mean_wrist_distance_mm": _mean(x.wrist_distance_mm for x in seq),
            }
        )

    filtered = [s for s in samples if s.visible and s.valid_scale]
    by_ds_side: Dict[Tuple[str, str], List[HandSample]] = defaultdict(list)
    by_ds: Dict[str, List[HandSample]] = defaultdict(list)
    for s in filtered:
        by_ds_side[(s.dataset, s.side)].append(s)
        by_ds[s.dataset].append(s)

    for (dataset, side), seq in sorted(by_ds_side.items()):
        _emit(dataset, side, seq)

    for dataset, seq in sorted(by_ds.items()):
        _emit(dataset, "mean", seq)

    return rows


def discover_prediction_paths(
    explicit_hot3d: str,
    explicit_arctic: str,
    log_dir: str,
    suffix: str,
) -> List[str]:
    paths: List[str] = []

    for p in [explicit_hot3d, explicit_arctic]:
        if p:
            paths.append(p)

    search_root = _resolve_prediction_search_root(log_dir=log_dir)

    if not paths:
        suffix = str(suffix).strip()

        if suffix:
            # Keep explicit HOT3D/ARCTIC lookup first for stable ordering.
            for ds in ["HOT3D", "ARCTIC"]:
                candidate = os.path.join(search_root, f"{ds}_{suffix}_predictions.pkl")
                if os.path.exists(candidate):
                    paths.append(candidate)

            # Fallback to any dataset using the same suffix naming convention.
            if not paths:
                paths.extend(sorted(glob.glob(os.path.join(search_root, f"*_{suffix}_predictions.pkl"))))
        else:
            # No suffix provided: discover all prediction files in the search root.
            paths.extend(sorted(glob.glob(os.path.join(search_root, "*_predictions.pkl"))))

    # Unique preserve order.
    out = []
    seen = set()
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            out.append(ap)

    return out


def build_text_report(
    prediction_paths: Sequence[str],
    prediction_search_root: str,
    prediction_suffix: str,
    overall_rows: Sequence[Dict[str, Any]],
    sequence_rows: Sequence[Dict[str, Any]],
    hot3d_overall_rows: Sequence[Dict[str, Any]],
    arctic_overall_rows: Sequence[Dict[str, Any]],
    output_dir: str,
    n_calibration_samples: int,
    acc_threshold_mm: float,
    ignore_failure_solves: bool,
    temporal_filter_summary: Dict[str, Any],
) -> str:
    lines = []
    lines.append("Hand Scale Evaluation Report")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("Prediction files:")
    for p in prediction_paths:
        lines.append(f"- {p}")
    lines.append(f"Prediction search root: {prediction_search_root}")
    lines.append(f"Prediction suffix: {prediction_suffix}")

    lines.append("")
    lines.append(f"Calibration samples per dataset sequence-side: {n_calibration_samples}")
    lines.append(f"Accuracy threshold for CS_ACC/RR_ACC/PA_ACC: {acc_threshold_mm:.1f} mm")
    lines.append(f"Ignore failure solves (CS > 1000mm): {bool(ignore_failure_solves)}")
    lines.append("Metric units: MPJPE in mm, ACC_ERROR in m/s^2, ACC metrics in %")
    lines.append("Temporal filtering:")
    if temporal_filter_summary.get("enabled", False):
        lines.append(
            "- enabled mode={mode}, q_pos={q_pos}, q_vel={q_vel}, r_meas={r_meas}, freq={freq}, "
            "groups={groups}, frames={frames}, mean_shift_mm={mean_shift:.1f}, p95_shift_mm={p95_shift:.1f}".format(
                mode=temporal_filter_summary.get("mode"),
                q_pos=temporal_filter_summary.get("q_pos"),
                q_vel=temporal_filter_summary.get("q_vel"),
                r_meas=temporal_filter_summary.get("r_meas"),
                freq=temporal_filter_summary.get("freq"),
                groups=temporal_filter_summary.get("num_sequence_side_groups"),
                frames=temporal_filter_summary.get("num_frames_filtered"),
                mean_shift=_safe_float(temporal_filter_summary.get("mean_translation_shift_mm")),
                p95_shift=_safe_float(temporal_filter_summary.get("p95_translation_shift_mm")),
            )
        )
    else:
        lines.append("- disabled")

    lines.append("")
    lines.append("Overall scale error summary:")
    for r in overall_rows:
        lines.append(
            f"- {r['dataset']} {r['side']}: "
            f"count={r['count']}, "
            f"mean_abs_err_mm={_safe_float(r['mean_abs_scale_error_mm']):.1f}, "
            f"std_abs_err_mm={_safe_float(r.get('std_abs_scale_error_mm', np.nan)):.1f}, "
            f"mean_rel_err_percent={_safe_float(r['mean_rel_scale_error_percent']):.1f}, "
            f"std_rel_err_percent={_safe_float(r.get('std_rel_scale_error_percent', np.nan)):.1f}"
        )

    lines.append("")
    lines.append("Sequence IDs (dataset | subject | sequence):")
    seen_seq = set()
    for r in sorted(sequence_rows, key=lambda x: (x["dataset"], x["subject_id"], x["sequence_id"], x["side"])):
        key = (r["dataset"], r["subject_id"], r["sequence_id"])
        if key in seen_seq:
            continue
        seen_seq.add(key)
        lines.append(f"- {r['dataset']} | {r['subject_id']} | {r['sequence_id']}")

    if hot3d_overall_rows:
        lines.append("")
        lines.append("HOT3D calibration overall (before -> after):")
        for r in hot3d_overall_rows:
            lines.append(
                f"- {r['side']}: "
                f"CS_MPJPE_mm {r['CS_MPJPE_before']:.1f}->{r['CS_MPJPE_after']:.1f} "
                f"({r['CS_MPJPE_improvement_percent']:.1f}%), "
                f"RR_MPJPE_mm {r['RR_MPJPE_before']:.1f}->{r['RR_MPJPE_after']:.1f} "
                f"({r['RR_MPJPE_improvement_percent']:.1f}%), "
                f"PA_MPJPE_mm {r['PA_MPJPE_before']:.1f}->{r['PA_MPJPE_after']:.1f} "
                f"({r['PA_MPJPE_improvement_percent']:.1f}%), "
                f"ACC_ERROR_m_per_s2 {r['ACC_ERROR_before']:.1f}->{r['ACC_ERROR_after']:.1f} "
                f"({r['ACC_ERROR_improvement_percent']:.1f}%)"
            )

    if arctic_overall_rows:
        lines.append("")
        lines.append("ARCTIC calibration overall (before -> after):")
        for r in arctic_overall_rows:
            lines.append(
                f"- {r['side']}: "
                f"CS_MPJPE_mm {r['CS_MPJPE_before']:.1f}->{r['CS_MPJPE_after']:.1f} "
                f"({r['CS_MPJPE_improvement_percent']:.1f}%), "
                f"RR_MPJPE_mm {r['RR_MPJPE_before']:.1f}->{r['RR_MPJPE_after']:.1f} "
                f"({r['RR_MPJPE_improvement_percent']:.1f}%), "
                f"PA_MPJPE_mm {r['PA_MPJPE_before']:.1f}->{r['PA_MPJPE_after']:.1f} "
                f"({r['PA_MPJPE_improvement_percent']:.1f}%), "
                f"ACC_ERROR_m_per_s2 {r['ACC_ERROR_before']:.1f}->{r['ACC_ERROR_after']:.1f} "
                f"({r['ACC_ERROR_improvement_percent']:.1f}%)"
            )

    lines.append("")
    lines.append("Output artifacts:")
    lines.append(f"- {os.path.join(output_dir, 'hand_scale_per_sample.csv')}")
    lines.append(f"- {os.path.join(output_dir, 'hand_scale_overall.csv')}")
    lines.append(f"- {os.path.join(output_dir, 'hand_scale_by_distance_bin.csv')}")
    lines.append(f"- {os.path.join(output_dir, 'hand_scale_by_sequence.csv')}")
    lines.append(f"- {os.path.join(output_dir, 'hot3d_calibration_sequence_metrics.csv')}")
    lines.append(f"- {os.path.join(output_dir, 'hot3d_calibration_overall.csv')}")
    lines.append(f"- {os.path.join(output_dir, 'arctic_calibration_sequence_metrics.csv')}")
    lines.append(f"- {os.path.join(output_dir, 'arctic_calibration_overall.csv')}")
    lines.append(f"- {os.path.join(output_dir, 'hand_scale_evaluation.pkl')}")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hand scale evaluator for HOT3D and ARCTIC predictions")

    default_predictions_dir = os.path.join(ROOT_DIR, "_DATA", "predictions")
    default_results_root = os.path.join(ROOT_DIR, "results", "hand_scale_eval")

    parser.add_argument("--hot3d-predictions", type=str, default="", help="Path to HOT3D predictions PKL")
    parser.add_argument("--arctic-predictions", type=str, default="", help="Path to ARCTIC predictions PKL")

    parser.add_argument(
        "--log-dir",
        type=str,
        default=default_predictions_dir,
        help="Directory containing prediction PKLs (default: _DATA/predictions).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=os.environ.get("SUFFIX", ""),
        help="Prediction filename suffix (e.g. undistort_inp_true_no_arm_prior). "
             "If empty, it is derived from workflow flags.",
    )

    parser.add_argument("--no-undistort-inp", action="store_true", help="Match save_predictions suffix behavior.")
    parser.add_argument("--no-cit", action="store_true", help="Match save_predictions suffix behavior.")
    parser.add_argument("--no-arm-prior", action="store_true", help="Match save_predictions suffix behavior.")
    parser.add_argument("--no-arm-input", action="store_true", help="Match save_predictions suffix behavior.")

    anycalib_group = parser.add_mutually_exclusive_group()
    anycalib_group.add_argument("--anycalib-624", action="store_true", help="Match save_predictions suffix behavior.")
    anycalib_group.add_argument("--anycalib-pin", action="store_true", help="Match save_predictions suffix behavior.")
    parser.add_argument("--depth-model", action="store_true", help="Match save_predictions suffix behavior.")
    parser.add_argument("--dgp-model", action="store_true", help="Match save_predictions suffix behavior.")

    parser.add_argument("--output-dir", type=str, default="", help="Directory for CSV/plots/logs")
    parser.add_argument(
        "--results-root",
        type=str,
        default=default_results_root,
        help="Root directory for outputs when --output-dir is not provided.",
    )
    parser.add_argument("--n-calibration-samples", type=int, default=30)
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Batch size used for evaluate_ours-style metric averaging.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--acc-threshold-mm", type=float, default=20.0)
    parser.add_argument(
        "--ignore-failure-solves",
        dest="ignore_failure_solves",
        action="store_true",
        default=True,
        help="Ignore failed solves (CS_MPJPE > 1000mm) when averaging CS/ACC metrics. Enabled by default.",
    )
    parser.add_argument(
        "--no-ignore-failure-solves",
        dest="ignore_failure_solves",
        action="store_false",
        help="Disable filtering of failed solves (CS_MPJPE > 1000mm) in CS/ACC metric averaging.",
    )
    parser.add_argument(
        "--temporal-filter",
        type=str,
        default="kalman_cv",
        choices=["none", "kalman_cv"],
        help="Temporal smoothing mode. 'kalman_cv' matches evaluate_ours Kalman-CV params by default.",
    )
    parser.add_argument("--kalman-q-pos", type=float, default=0.001)
    parser.add_argument("--kalman-q-vel", type=float, default=1e-05)
    parser.add_argument("--kalman-r-meas", type=float, default=0.001)
    parser.add_argument("--kalman-freq", type=float, default=30.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_suffix = _resolve_prediction_suffix(args)
    prediction_search_root = _resolve_prediction_search_root(log_dir=args.log_dir)

    print(
        "Prediction discovery settings: "
        f"search_root={prediction_search_root}, suffix={resolved_suffix or '<any>'}"
    )

    prediction_paths = discover_prediction_paths(
        explicit_hot3d=args.hot3d_predictions,
        explicit_arctic=args.arctic_predictions,
        log_dir=args.log_dir,
        suffix=resolved_suffix,
    )

    if not prediction_paths:
        raise FileNotFoundError(
            "No prediction files found. Provide --hot3d-predictions/--arctic-predictions, "
            f"or place files under {prediction_search_root} matching '*_{resolved_suffix}_predictions.pkl'."
        )

    for p in prediction_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Prediction file does not exist: {p}")

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        results_root = os.path.abspath(args.results_root)
        suffix_part = resolved_suffix if resolved_suffix else "all_predictions"
        output_dir = os.path.abspath(os.path.join(results_root, suffix_part))

    os.makedirs(output_dir, exist_ok=True)

    print("Loading prediction files:")
    all_samples: List[HandSample] = []
    for path in prediction_paths:
        preds = load_predictions(path)
        samples = extract_samples(preds)
        all_samples.extend(samples)

        by_ds = defaultdict(int)
        for s in samples:
            by_ds[s.dataset] += 1
        ds_text = ", ".join([f"{k}:{v}" for k, v in sorted(by_ds.items())])
        print(f"- {path} -> {len(samples)} ({ds_text})")

    if not all_samples:
        raise RuntimeError("No usable side-samples were extracted from prediction files.")

    all_samples, temporal_filter_summary = apply_temporal_filter(
        all_samples,
        mode=args.temporal_filter,
        q_pos=float(args.kalman_q_pos),
        q_vel=float(args.kalman_q_vel),
        r_meas=float(args.kalman_r_meas),
        freq=float(args.kalman_freq),
    )
    if temporal_filter_summary.get("enabled", False):
        print(
            "Applied temporal filter: mode={mode}, q_pos={q_pos}, q_vel={q_vel}, r_meas={r_meas}, freq={freq}, ".format(
                mode=temporal_filter_summary.get("mode"),
                q_pos=temporal_filter_summary.get("q_pos"),
                q_vel=temporal_filter_summary.get("q_vel"),
                r_meas=temporal_filter_summary.get("r_meas"),
                freq=temporal_filter_summary.get("freq"),
            )
        )
    else:
        print("Temporal filter disabled.")

    per_sample_rows = rows_from_samples(all_samples)
    overall_rows = aggregate_overall_scale(all_samples)
    distance_rows = aggregate_scale_by_distance(all_samples)
    sequence_rows = aggregate_scale_by_sequence(all_samples)

    hot3d_seq_rows, hot3d_overall_rows, hot3d_scale_factors = calibrate_dataset_sequences(
        all_samples,
        dataset_name="HOT3D",
        n_calibration_samples=max(0, int(args.n_calibration_samples)),
        fps=float(args.fps),
        acc_threshold_mm=float(args.acc_threshold_mm),
        eval_batch_size=max(1, int(args.eval_batch_size)),
        ignore_failure_solves=bool(args.ignore_failure_solves),
    )
    arctic_seq_rows, arctic_overall_rows, arctic_scale_factors = calibrate_dataset_sequences(
        all_samples,
        dataset_name="ARCTIC",
        n_calibration_samples=max(0, int(args.n_calibration_samples)),
        fps=float(args.fps),
        acc_threshold_mm=float(args.acc_threshold_mm),
        eval_batch_size=max(1, int(args.eval_batch_size)),
        ignore_failure_solves=bool(args.ignore_failure_solves),
    )

    # Final artifact rows are rounded to 1 decimal as requested.
    per_sample_rows_out = round_rows_for_artifacts(per_sample_rows, decimals=1)
    overall_rows_out = round_rows_for_artifacts(overall_rows, decimals=1)
    distance_rows_out = round_rows_for_artifacts(distance_rows, decimals=1)
    sequence_rows_out = round_rows_for_artifacts(sequence_rows, decimals=1)
    hot3d_seq_rows_out = round_rows_for_artifacts(hot3d_seq_rows, decimals=1)
    hot3d_overall_rows_out = round_rows_for_artifacts(hot3d_overall_rows, decimals=1)
    arctic_seq_rows_out = round_rows_for_artifacts(arctic_seq_rows, decimals=1)
    arctic_overall_rows_out = round_rows_for_artifacts(arctic_overall_rows, decimals=1)

    # Write tabular outputs.
    write_csv(os.path.join(output_dir, "hand_scale_per_sample.csv"), per_sample_rows_out)
    write_csv(os.path.join(output_dir, "hand_scale_overall.csv"), overall_rows_out)
    write_csv(os.path.join(output_dir, "hand_scale_by_distance_bin.csv"), distance_rows_out)
    write_csv(os.path.join(output_dir, "hand_scale_by_sequence.csv"), sequence_rows_out)
    write_csv(os.path.join(output_dir, "hot3d_calibration_sequence_metrics.csv"), hot3d_seq_rows_out)
    write_csv(os.path.join(output_dir, "hot3d_calibration_overall.csv"), hot3d_overall_rows_out)
    write_csv(os.path.join(output_dir, "arctic_calibration_sequence_metrics.csv"), arctic_seq_rows_out)
    write_csv(os.path.join(output_dir, "arctic_calibration_overall.csv"), arctic_overall_rows_out)

    # Plot outputs.
    plot_distance_profiles(distance_rows, output_dir)
    plot_scale_scatter(all_samples, output_dir)
    plot_sequence_scale_errors(sequence_rows, output_dir)
    plot_dataset_calibration("HOT3D", hot3d_seq_rows, hot3d_overall_rows, output_dir)
    plot_dataset_calibration("ARCTIC", arctic_seq_rows, arctic_overall_rows, output_dir)

    # Save full payload for further analysis.
    payload = {
        "prediction_paths": prediction_paths,
        "prediction_search_root": prediction_search_root,
        "prediction_suffix": resolved_suffix,
        "distance_thresholds_mm": DISTANCE_THRESHOLDS_MM,
        "distance_bin_labels": BIN_LABELS,
        "n_calibration_samples": int(args.n_calibration_samples),
        "eval_batch_size": int(args.eval_batch_size),
        "fps": float(args.fps),
        "acc_threshold_mm": float(args.acc_threshold_mm),
        "ignore_failure_solves": bool(args.ignore_failure_solves),
        "temporal_filter_summary": temporal_filter_summary,
        "overall_rows": overall_rows_out,
        "distance_rows": distance_rows_out,
        "sequence_rows": sequence_rows_out,
        "hot3d_calibration_sequence_rows": hot3d_seq_rows_out,
        "hot3d_calibration_overall_rows": hot3d_overall_rows_out,
        "hot3d_scale_factors": {f"{k[0]}::{k[1]}": v for k, v in hot3d_scale_factors.items()},
        "arctic_calibration_sequence_rows": arctic_seq_rows_out,
        "arctic_calibration_overall_rows": arctic_overall_rows_out,
        "arctic_scale_factors": {f"{k[0]}::{k[1]}": v for k, v in arctic_scale_factors.items()},
    }
    payload = round_nested_for_artifacts(payload, decimals=1)

    report_text = build_text_report(
        prediction_paths=prediction_paths,
        prediction_search_root=prediction_search_root,
        prediction_suffix=resolved_suffix,
        overall_rows=overall_rows_out,
        sequence_rows=sequence_rows_out,
        hot3d_overall_rows=hot3d_overall_rows_out,
        arctic_overall_rows=arctic_overall_rows_out,
        output_dir=output_dir,
        n_calibration_samples=int(args.n_calibration_samples),
        acc_threshold_mm=float(args.acc_threshold_mm),
        ignore_failure_solves=bool(args.ignore_failure_solves),
        temporal_filter_summary=temporal_filter_summary,
    )
    report_path = os.path.join(output_dir, "hand_scale_evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    print("\nDone. Sequence IDs (dataset | subject | sequence):")
    printed = set()
    for r in sorted(sequence_rows_out, key=lambda x: (x["dataset"], x["subject_id"], x["sequence_id"])):
        key = (r["dataset"], r["subject_id"], r["sequence_id"])
        if key in printed:
            continue
        printed.add(key)
        print(f"- {r['dataset']} | {r['subject_id']} | {r['sequence_id']}")

    print("\nOverall scale metrics:")
    for r in overall_rows_out:
        print(
            f"- {r['dataset']} {r['side']}: count={r['count']}, "
            f"mean_abs_scale_error_mm={_safe_float(r['mean_abs_scale_error_mm']):.1f}, "
            f"std_abs_scale_error_mm={_safe_float(r.get('std_abs_scale_error_mm', np.nan)):.1f}, "
            f"mean_rel_scale_error_percent={_safe_float(r['mean_rel_scale_error_percent']):.1f}, "
            f"std_rel_scale_error_percent={_safe_float(r.get('std_rel_scale_error_percent', np.nan)):.1f}"
        )

    if hot3d_overall_rows_out:
        print("\nHOT3D calibration metrics (before -> after):")
        for r in hot3d_overall_rows_out:
            print(
                f"- {r['side']}: "
                f"CS_MPJPE_mm {r['CS_MPJPE_before']:.1f}->{r['CS_MPJPE_after']:.1f}, "
                f"RR_MPJPE_mm {r['RR_MPJPE_before']:.1f}->{r['RR_MPJPE_after']:.1f}, "
                f"PA_MPJPE_mm {r['PA_MPJPE_before']:.1f}->{r['PA_MPJPE_after']:.1f}, "
                f"ACC_ERROR_m_per_s2 {r['ACC_ERROR_before']:.1f}->{r['ACC_ERROR_after']:.1f}, "
                f"CS_ACC_% {r['CS_ACC_before']:.1f}->{r['CS_ACC_after']:.1f}, "
                f"RR_ACC_% {r['RR_ACC_before']:.1f}->{r['RR_ACC_after']:.1f}, "
                f"PA_ACC_% {r['PA_ACC_before']:.1f}->{r['PA_ACC_after']:.1f}"
            )

    if arctic_overall_rows_out:
        print("\nARCTIC calibration metrics (before -> after):")
        for r in arctic_overall_rows_out:
            print(
                f"- {r['side']}: "
                f"CS_MPJPE_mm {r['CS_MPJPE_before']:.1f}->{r['CS_MPJPE_after']:.1f}, "
                f"RR_MPJPE_mm {r['RR_MPJPE_before']:.1f}->{r['RR_MPJPE_after']:.1f}, "
                f"PA_MPJPE_mm {r['PA_MPJPE_before']:.1f}->{r['PA_MPJPE_after']:.1f}, "
                f"ACC_ERROR_m_per_s2 {r['ACC_ERROR_before']:.1f}->{r['ACC_ERROR_after']:.1f}, "
                f"CS_ACC_% {r['CS_ACC_before']:.1f}->{r['CS_ACC_after']:.1f}, "
                f"RR_ACC_% {r['RR_ACC_before']:.1f}->{r['RR_ACC_after']:.1f}, "
                f"PA_ACC_% {r['PA_ACC_before']:.1f}->{r['PA_ACC_after']:.1f}"
            )

    print(f"\nSaved report to: {report_path}")
    print(f"Saved all artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
