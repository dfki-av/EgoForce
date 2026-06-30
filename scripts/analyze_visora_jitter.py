#!/usr/bin/env python
import argparse
import contextlib
import io
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGOFORCE_DISABLE_TRT", "1")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

import scripts.run_visora_video as rv


def point_center(points, confidences=None):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2:
        return None
    valid = np.isfinite(points[:, :2]).all(axis=1)
    if confidences is not None:
        confidences = np.asarray(confidences, dtype=np.float32)
        valid &= confidences[: len(points)] >= rv.ARKIT_CROP_CONFIDENCE_THRESHOLD
    if valid.sum() < 3:
        return None
    return points[valid, :2].mean(axis=0).tolist()


def bbox_center_size(record):
    if record is None:
        return None, None
    bbox = np.asarray(record["bbox"], dtype=np.float32)
    center = [float(0.5 * (bbox[0] + bbox[2])), float(0.5 * (bbox[1] + bbox[3]))]
    size = [float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])]
    return center, size


def median_error(points, reference, confidences):
    points = np.asarray(points, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    confidences = np.asarray(confidences, dtype=np.float32)
    n = min(len(points), len(reference), len(confidences))
    if n == 0:
        return None
    valid = (
        np.isfinite(points[:n, :2]).all(axis=1)
        & np.isfinite(reference[:n, :2]).all(axis=1)
        & (confidences[:n] >= rv.ARKIT_CROP_CONFIDENCE_THRESHOLD)
    )
    if valid.sum() < 3:
        return None
    return float(np.median(np.linalg.norm(points[:n, :2][valid] - reference[:n, :2][valid], axis=1)))


def finite_delta(values):
    output = []
    prev = None
    for value in values:
        if value is None:
            prev = None
            output.append(None)
            continue
        arr = np.asarray(value, dtype=np.float32)
        if prev is None:
            output.append(None)
        else:
            output.append(float(np.linalg.norm(arr - prev)))
        prev = arr
    return output


def percentile(values, q):
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return None
    return float(np.percentile(clean, q))


def summarize_side(frames, side):
    rows = [frame["sides"][side] for frame in frames if side in frame["sides"]]
    source_counts = Counter(row.get("crop_source", "missing") for row in rows)
    switches = 0
    previous_source = None
    for row in rows:
        source = row.get("crop_source", "missing")
        if previous_source is not None and source != previous_source:
            switches += 1
        previous_source = source

    signals = {}
    for key in (
        "arkit_center",
        "crop_center",
        "direct_center",
        "camera_pre_center",
        "camera_after_center",
        "translation_after",
    ):
        deltas = finite_delta([row.get(key) for row in rows])
        signals[key] = {
            "delta_median": percentile(deltas, 50),
            "delta_p90": percentile(deltas, 90),
            "delta_p95": percentile(deltas, 95),
            "delta_max": percentile(deltas, 100),
        }

    return {
        "frames": len(rows),
        "source_counts": dict(source_counts),
        "source_switches": switches,
        "direct_error_median": percentile([row.get("direct_error") for row in rows], 50),
        "direct_error_p90": percentile([row.get("direct_error") for row in rows], 90),
        "camera_pre_error_median": percentile([row.get("camera_pre_error") for row in rows], 50),
        "camera_pre_error_p90": percentile([row.get("camera_pre_error") for row in rows], 90),
        "camera_after_error_median": percentile([row.get("camera_after_error") for row in rows], 50),
        "camera_after_error_p90": percentile([row.get("camera_after_error") for row in rows], 90),
        "lidar_anchor_frames": sum(1 for row in rows if row.get("lidar_anchor_points")),
        "lidar_anchor_points_median": percentile([row.get("lidar_anchor_points") for row in rows], 50),
        "lidar_depth_delta_p95": percentile(
            finite_delta([[row["lidar_depth_median"]] for row in rows if row.get("lidar_depth_median") is not None]),
            95,
        ),
        "translation_z_median": percentile(
            [
                None if row.get("translation_after") is None else row["translation_after"][2]
                for row in rows
            ],
            50,
        ),
        "translation_z_p05": percentile(
            [
                None if row.get("translation_after") is None else row["translation_after"][2]
                for row in rows
            ],
            5,
        ),
        "near_camera_frames": sum(
            1
            for row in rows
            if row.get("translation_after") is not None and row["translation_after"][2] <= 0.05
        ),
        "missing_crop_frames": sum(1 for row in rows if row.get("crop_source") is None),
        "low_confidence_hand_frames": sum(
            1
            for row in rows
            if row.get("hand_weight_sum") is None
            or row.get("hand_weight_sum") < rv.EGOFORCE_MIN_HAND_WEIGHT_SUM
        ),
        "hand_weight_sum_median": percentile([row.get("hand_weight_sum") for row in rows], 50),
        "hand_weight_sum_p05": percentile([row.get("hand_weight_sum") for row in rows], 5),
        "arm_weight_sum_median": percentile([row.get("arm_weight_sum") for row in rows], 50),
        "arm_weight_sum_p05": percentile([row.get("arm_weight_sum") for row in rows], 5),
        "signals": signals,
    }


def top_events(frames, side, key, count=8):
    rows = [(idx, frame["sides"][side]) for idx, frame in enumerate(frames) if side in frame["sides"]]
    deltas = finite_delta([row.get(key) for _, row in rows])
    ranked = []
    for (idx, row), delta in zip(rows, deltas):
        if delta is None:
            continue
        ranked.append(
            {
                "frame": frames[idx]["video_frame_est"],
                "video_time": frames[idx]["video_time"],
                "delta": delta,
                "crop_source": row.get("crop_source"),
                "direct_error": row.get("direct_error"),
                "camera_after_error": row.get("camera_after_error"),
            }
        )
    ranked.sort(key=lambda item: item["delta"], reverse=True)
    return ranked[:count]


def main():
    parser = argparse.ArgumentParser(description="Measure Visora/EgoForce annotation jitter.")
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--dataset-root", default=str(ROOT_DIR / "visora-arkit-v1"))
    parser.add_argument("--start-seconds", type=float, default=48.0)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--crop-source", default="arkit-yolo-hybrid", choices=("arkit", "yolo", "arkit-yolo-hybrid", "original"))
    parser.add_argument("--arm-source", default="synthetic-arkit", choices=("none", "drop", "synthetic-arkit"))
    parser.add_argument("--arkit-side-source", default="screen", choices=("screen", "chirality"))
    parser.add_argument("--depth-anchor", default="lidar", choices=("none", "lidar"))
    parser.add_argument("--egoforce-input-cleanup", default="none", choices=("none", "inpaint-markers"))
    parser.add_argument("--detector-mode", default="guarded-current", choices=("current", "guarded-current", "tracked-screen", "upstream"))
    parser.add_argument(
        "--egoforce-undistort-inp",
        action="store_true",
        help="Use EgoForce's original local crop undistortion path instead of raw video crops.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pair_dir = Path(args.dataset_root) / args.pair_id
    video_path = pair_dir / "video.mp4"
    arkit_zip_path = pair_dir / "arkit.zip"

    _, Inference, _, _ = rv.ensure_egoforce_runtime()
    inference = Inference()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    fps = rv.get_capture_fps(capture)
    start_time = max(0.0, float(args.start_seconds))

    arkit_archive = zipfile.ZipFile(arkit_zip_path)
    try:
        arkit_frames, hand_pose_rows = rv.read_arkit_metadata(arkit_zip_path)
        archive_root = rv.archive_root_from_zip(arkit_archive)
        arkit_start_timestamp = float(arkit_frames[0].get("timestamp", 0.0))
        selected_rows = rv.arkit_rows_for_time_window(
            arkit_frames,
            arkit_start_timestamp,
            start_time,
            args.seconds,
            stride=args.stride,
        )
        if not selected_rows:
            raise RuntimeError("No ARKit rows found for requested time window.")

        row_fps = rv.fps_from_arkit_rows(selected_rows, fps)
        camera_row = selected_rows[0]
        start_frame = max(0, int(camera_row.get("frame_index", round(start_time * fps))))
        inference.reset_runtime_state()
        rv.configure_detector_mode(inference, args.detector_mode)
        inference.set_camera_model(
            rv.camera_model_from_arkit_row(camera_row),
            undistort_inp=args.egoforce_undistort_inp,
        )
        inference.set_kalman_filter_frequency(row_fps)

        frames = []
        frame_reader = rv.IndexedVideoFrameReader(capture)
        for processed, arkit_row in enumerate(selected_rows):
            video_frame_index = int(arkit_row.get("frame_index", start_frame + processed))
            bgr = frame_reader.read(video_frame_index)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            video_time = float(arkit_row.get("timestamp", arkit_start_timestamp)) - arkit_start_timestamp
            target_timestamp = arkit_start_timestamp + video_time
            hand_pose_row = rv.row_nearest_frame_index(hand_pose_rows, video_frame_index)
            inference.set_camera_model(
                rv.camera_model_from_arkit_row(arkit_row),
                undistort_inp=args.egoforce_undistort_inp,
            )

            if args.depth_anchor == "lidar":
                depth, confidence = rv.read_depth_confidence_maps(arkit_archive, archive_root, arkit_row)
            else:
                depth, confidence = None, None

            arkit_boxes = rv.arkit_bounding_boxes(hand_pose_row, rgb.shape, side_source=args.arkit_side_source)
            with contextlib.redirect_stdout(io.StringIO()):
                outs, bounding_boxes = rv.run_egoforce_outputs(
                    inference,
                    rgb.copy(),
                    hand_pose_row=hand_pose_row,
                    crop_source=args.crop_source,
                    arkit_side_source=args.arkit_side_source,
                    arm_source=args.arm_source,
                    input_cleanup=args.egoforce_input_cleanup,
                )

            direct_points = np.asarray(outs.get("pred_hand_j2d_direct", outs["pred_j2d"]), dtype=np.float32).copy()
            camera_pre = np.asarray(outs["pred_j2d"], dtype=np.float32).copy()
            translation_pre = np.asarray(outs.get("pred_transl"), dtype=np.float32).copy()

            if args.depth_anchor == "lidar":
                rv.apply_lidar_depth_anchor(
                    outs,
                    bounding_boxes,
                    depth,
                    confidence,
                    inference.camera_model,
                    rgb.shape,
                )

            camera_after = np.asarray(outs["pred_j2d"], dtype=np.float32).copy()
            translation_after = np.asarray(outs.get("pred_transl"), dtype=np.float32).copy()
            anchors = outs.get("lidar_depth_anchor", {})

            frame_record = {
                "processed": processed,
                "video_frame_est": video_frame_index,
                "video_time": video_time,
                "target_timestamp": target_timestamp,
                "arkit_frame": None if arkit_row is None else int(arkit_row.get("frame_index", -1)),
                "arkit_dt": None if arkit_row is None else float(float(arkit_row.get("timestamp", 0.0)) - target_timestamp),
                "hand_frame": None if hand_pose_row is None else int(hand_pose_row.get("frame_index", -1)),
                "input_cleanup": outs.get("input_cleanup"),
                "sides": {},
            }

            for hdx, side in enumerate(("left", "right")):
                record = bounding_boxes.get(side, {}).get("hand")
                arkit_record = arkit_boxes.get(side, {}).get("hand")
                crop_center, crop_size = bbox_center_size(record)
                arkit_center = None
                arkit_conf = None
                arkit_points = None
                if arkit_record is not None:
                    arkit_points = np.asarray(arkit_record.get("keypoint", []), dtype=np.float32)
                    arkit_conf = np.asarray(arkit_record.get("conf", []), dtype=np.float32)
                    arkit_center = point_center(arkit_points, arkit_conf)

                anchor = anchors.get(side, {})
                hand_weights = np.asarray(outs.get("pred_hand_kpt_w", [])[hdx], dtype=np.float32)
                arm_weights = np.asarray(outs.get("pred_arm_kpt_w", [])[hdx], dtype=np.float32)
                side_record = {
                    "crop_source": None if record is None else record.get("source", "unknown"),
                    "crop_center": crop_center,
                    "crop_size": crop_size,
                    "arkit_center": arkit_center,
                    "direct_center": point_center(direct_points[hdx]),
                    "camera_pre_center": point_center(camera_pre[hdx]),
                    "camera_after_center": point_center(camera_after[hdx]),
                    "translation_pre": translation_pre[hdx].reshape(-1, 3)[0].tolist(),
                    "translation_after": translation_after[hdx].reshape(-1, 3)[0].tolist(),
                    "lidar_anchor_points": anchor.get("points"),
                    "lidar_depth_median": anchor.get("depth_median"),
                    "hand_weight_sum": float(np.sum(hand_weights)) if hand_weights.size else None,
                    "hand_weight_mean": float(np.mean(hand_weights)) if hand_weights.size else None,
                    "arm_weight_sum": float(np.sum(arm_weights)) if arm_weights.size else None,
                    "arm_weight_mean": float(np.mean(arm_weights)) if arm_weights.size else None,
                }
                if arkit_points is not None and arkit_conf is not None:
                    side_record["direct_error"] = median_error(direct_points[hdx], arkit_points, arkit_conf)
                    side_record["camera_pre_error"] = median_error(camera_pre[hdx], arkit_points, arkit_conf)
                    side_record["camera_after_error"] = median_error(camera_after[hdx], arkit_points, arkit_conf)
                frame_record["sides"][side] = side_record

            frames.append(frame_record)
    finally:
        capture.release()
        arkit_archive.close()

    summary = {
        "pair_id": args.pair_id,
        "start_seconds": args.start_seconds,
        "seconds": args.seconds,
        "stride": args.stride,
        "crop_source": args.crop_source,
        "arm_source": args.arm_source,
        "arkit_side_source": args.arkit_side_source,
        "depth_anchor": args.depth_anchor,
        "egoforce_input_cleanup": args.egoforce_input_cleanup,
        "detector_mode": args.detector_mode,
        "egoforce_undistort_inp": args.egoforce_undistort_inp,
        "video_fps": fps,
        "row_fps": row_fps,
        "frames_analyzed": len(frames),
        "input_cleanup_masked_pixels_median": percentile(
            [
                None if frame.get("input_cleanup") is None else frame["input_cleanup"].get("masked_pixels")
                for frame in frames
            ],
            50,
        ),
        "input_cleanup_masked_pixels_p95": percentile(
            [
                None if frame.get("input_cleanup") is None else frame["input_cleanup"].get("masked_pixels")
                for frame in frames
            ],
            95,
        ),
        "left": summarize_side(frames, "left"),
        "right": summarize_side(frames, "right"),
        "top_camera_after_jumps": {
            "left": top_events(frames, "left", "camera_after_center"),
            "right": top_events(frames, "right", "camera_after_center"),
        },
        "top_direct_jumps": {
            "left": top_events(frames, "left", "direct_center"),
            "right": top_events(frames, "right", "direct_center"),
        },
        "top_crop_jumps": {
            "left": top_events(frames, "left", "crop_center"),
            "right": top_events(frames, "right", "crop_center"),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "frames": frames}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
