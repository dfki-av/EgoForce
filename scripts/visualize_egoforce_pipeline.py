#!/usr/bin/env python
import argparse
import contextlib
import io
import os
import sys
import zipfile
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


PANEL_W = 480
PANEL_H = 360


def fit_panel(rgb, width=PANEL_W, height=PANEL_H):
    rgb = np.asarray(rgb)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    scale = min(width / rgb.shape[1], height / rgb.shape[0])
    new_w = max(1, int(round(rgb.shape[1] * scale)))
    new_h = max(1, int(round(rgb.shape[0] * scale)))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def label_panel(panel, label):
    output = panel.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), base = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(output, (0, 0), (tw + 14, th + base + 12), (0, 0, 0), -1)
    cv2.putText(output, label, (7, th + 6), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return output


def text_panel(lines, width=PANEL_W, height=PANEL_H):
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 28
    for line in lines:
        cv2.putText(panel, line[:58], (14, y), font, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        y += 23
        if y > height - 12:
            break
    return panel


def colorize_depth(depth, confidence=None):
    if depth is None:
        return np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.05) & (depth < 5.0)
    if confidence is not None:
        valid &= np.asarray(confidence) >= rv.LIDAR_DEPTH_CONFIDENCE_THRESHOLD
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
    else:
        lo, hi = 0.2, 1.5
    if hi <= lo:
        hi = lo + 1.0
    normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    image = (255 - normalized * 255).astype(np.uint8)
    color = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color[~valid] = (20, 20, 20)
    return color


def draw_boxes(rgb, boxes):
    output = rgb.copy()
    for side, color in (("left", rv.EGOFORCE_HAND_COLORS["left"]), ("right", rv.EGOFORCE_HAND_COLORS["right"])):
        for box_name in ("hand", "arm"):
            record = boxes.get(side, {}).get(box_name)
            if record is None:
                continue
            x1, y1, x2, y2 = np.round(record["bbox"]).astype(int)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3 if box_name == "hand" else 2)
            label = f"{side[0].upper()} {box_name}"
            cv2.putText(output, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return output


def draw_egoforce_skeleton(rgb, points_by_side, boxes=None):
    output = rgb.copy()
    for side, color in (("left", rv.EGOFORCE_HAND_COLORS["left"]), ("right", rv.EGOFORCE_HAND_COLORS["right"])):
        points = points_by_side.get(side)
        if points is None:
            continue
        points_by_name = {
            name: np.array([points[idx, 0], points[idx, 1], 1.0], dtype=np.float32)
            for idx, name in enumerate(rv.HAND_JOINT_ORDER)
            if idx < len(points) and np.isfinite(points[idx]).all()
        }
        rv.draw_hand_skeleton(output, points_by_name, color)
        if boxes is not None and "hand" in boxes.get(side, {}):
            x1, y1, x2, y2 = np.round(boxes[side]["hand"]["bbox"]).astype(int)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
    return cv2.addWeighted(output, 0.9, rgb, 0.1, 0)


def crop_montage(outs):
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    crops = [
        ("L hand", outs["hand_crop"][0]),
        ("R hand", outs["hand_crop"][1]),
        ("L arm", outs["arm_crop"][0]),
        ("R arm", outs["arm_crop"][1]),
    ]
    positions = [(0, 0), (PANEL_W // 2, 0), (0, PANEL_H // 2), (PANEL_W // 2, PANEL_H // 2)]
    for (label, crop), (x, y) in zip(crops, positions):
        crop = fit_panel(crop, PANEL_W // 2, PANEL_H // 2)
        panel[y : y + PANEL_H // 2, x : x + PANEL_W // 2] = crop
        cv2.putText(panel, label, (x + 8, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def copy_outs(outs):
    copied = {}
    for key, value in outs.items():
        copied[key] = value.copy() if isinstance(value, np.ndarray) else value
    return copied


def apply_lidar_anchor_from_egoforce_direct(outs, depth, confidence, camera_model, image_shape):
    anchored = copy_outs(outs)
    anchors = {}
    if depth is None or "pred_transl" not in anchored or "pred_hand_j2d_direct" not in anchored:
        anchored["lidar_depth_anchor"] = anchors
        return anchored

    for hdx, side in enumerate(("left", "right")):
        old_translation = rv.get_pred_translation(anchored["pred_transl"], hdx).copy()
        relative_joints = np.asarray(anchored["pred_j3d"][hdx], dtype=np.float32) - old_translation[None, :]
        points = np.asarray(anchored["pred_hand_j2d_direct"][hdx], dtype=np.float32)
        deltas = []
        sampled_depths = []
        for jdx, uv in enumerate(points[: len(relative_joints)]):
            if not np.isfinite(uv).all():
                continue
            depth_z = rv.sample_depth_at_uv(depth, confidence, uv, image_shape)
            if depth_z is None:
                continue
            target_xyz = rv.camera_xyz_from_uv_z(camera_model, uv, depth_z)
            deltas.append(target_xyz - relative_joints[jdx])
            sampled_depths.append(depth_z)
        if len(deltas) < rv.LIDAR_MIN_ANCHOR_POINTS:
            continue
        new_translation = np.median(np.asarray(deltas, dtype=np.float32), axis=0)
        if not np.isfinite(new_translation).all() or new_translation[2] <= 0.05:
            continue
        delta = new_translation - old_translation
        if np.linalg.norm(delta) > rv.LIDAR_MAX_ANCHOR_TRANSLATION_DELTA:
            continue
        for key in ("pred_j3d", "pred_vertices", "pred_arm_j3d", "pred_arm_vertices"):
            if key in anchored:
                anchored[key][hdx] = anchored[key][hdx] + delta
        rv.set_pred_translation(anchored["pred_transl"], hdx, new_translation)
        anchored["pred_j2d"][hdx] = camera_model.camera_to_uv(anchored["pred_j3d"][hdx])
        if "pred_arm_j2d" in anchored and "pred_arm_j3d" in anchored:
            anchored["pred_arm_j2d"][hdx] = camera_model.camera_to_uv(anchored["pred_arm_j3d"][hdx])
        anchors[side] = {
            "points": len(deltas),
            "depth_median": float(np.median(sampled_depths)),
            "old_translation": old_translation.tolist(),
            "new_translation": new_translation.tolist(),
        }
    anchored["lidar_depth_anchor"] = anchors
    return anchored


def run_outputs_with_boxes(inference, rgb, boxes):
    torch, _, infer_fn, cfg = rv.ensure_egoforce_runtime()
    left_data = inference.left_dataset.transform(rgb, boxes["left"])
    right_data = inference.right_dataset.transform(rgb, boxes["right"])
    with torch.no_grad():
        return infer_fn(inference, cfg, inference.model, inference.limb_model, left_data, right_data, inference.device)


def get_boxes(inference, rgb, crop_source, hand_pose_row, arm_source):
    if crop_source == "original":
        boxes = inference.detect_bounding_boxes(rgb)
    elif crop_source == "yolo-screen":
        boxes = rv.yolo_screen_bounding_boxes(inference, rgb)
    elif crop_source == "arkit":
        boxes = rv.arkit_bounding_boxes(hand_pose_row, rgb.shape, side_source="screen")
    else:
        raise ValueError(f"Unknown crop source: {crop_source}")

    if arm_source == "synthetic":
        boxes = rv.add_synthetic_arm_records(boxes, rgb.shape)
    elif arm_source != "original":
        raise ValueError(f"Unknown arm source: {arm_source}")
    return boxes


def compose_frame(rgb, depth, confidence, boxes, outs, outs_lidar, diagnostics):
    direct = {
        "left": np.asarray(outs["pred_hand_j2d_direct"][0], dtype=np.float32),
        "right": np.asarray(outs["pred_hand_j2d_direct"][1], dtype=np.float32),
    }
    camera = {
        "left": np.asarray(outs["pred_j2d"][0], dtype=np.float32),
        "right": np.asarray(outs["pred_j2d"][1], dtype=np.float32),
    }
    lidar = {
        "left": np.asarray(outs_lidar["pred_j2d"][0], dtype=np.float32),
        "right": np.asarray(outs_lidar["pred_j2d"][1], dtype=np.float32),
    }

    panels = [
        label_panel(fit_panel(rgb), "1 RGB input"),
        label_panel(fit_panel(colorize_depth(depth, confidence)), "2 LiDAR depth"),
        label_panel(fit_panel(draw_boxes(rgb, boxes)), "3 EgoForce detector/crops"),
        label_panel(crop_montage(outs), "4 tensors fed to model"),
        label_panel(fit_panel(draw_egoforce_skeleton(rgb, direct, boxes)), "5 direct 2D head"),
        label_panel(fit_panel(draw_egoforce_skeleton(rgb, camera, boxes)), "6 RSS camera projection"),
        label_panel(fit_panel(draw_egoforce_skeleton(rgb, lidar, boxes)), "7 LiDAR-anchored projection"),
        label_panel(text_panel(diagnostics), "8 diagnostics"),
    ]
    top = np.hstack(panels[:4])
    bottom = np.hstack(panels[4:])
    return np.vstack([top, bottom])


def main():
    parser = argparse.ArgumentParser(description="Visualize EgoForce pipeline stages on Visora RGB/LiDAR data.")
    parser.add_argument("--dataset-root", default=str(ROOT_DIR / "visora-arkit-v1"))
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--start-seconds", type=float, default=48.0)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "_DATA" / "egoforce_pipeline_viz"))
    parser.add_argument("--crop-source", choices=("original", "yolo-screen", "arkit"), default="original")
    parser.add_argument("--arm-source", choices=("original", "synthetic"), default="original")
    args = parser.parse_args()

    pair_dir = Path(args.dataset_root) / args.pair_id
    video_path = pair_dir / "video.mp4"
    arkit_zip_path = pair_dir / "arkit.zip"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, Inference, _, _ = rv.ensure_egoforce_runtime()
    inference = Inference()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = rv.get_capture_fps(cap)

    arkit_archive = zipfile.ZipFile(arkit_zip_path)
    writer = None
    processed = 0
    try:
        arkit_frames, hand_pose_rows = rv.read_arkit_metadata(arkit_zip_path)
        archive_root = rv.archive_root_from_zip(arkit_archive)
        arkit_start_timestamp = float(arkit_frames[0].get("timestamp", 0.0))
        selected_rows = rv.arkit_rows_for_time_window(
            arkit_frames,
            arkit_start_timestamp,
            args.start_seconds,
            args.seconds,
        )
        if not selected_rows:
            raise RuntimeError("No ARKit rows found for requested time window.")

        camera_row = selected_rows[0]
        start_frame = max(0, int(camera_row.get("frame_index", round(max(0.0, args.start_seconds) * fps))))
        inference.reset_runtime_state()
        inference.set_camera_model(rv.camera_model_from_arkit_row(camera_row), undistort_inp=False)
        inference.set_kalman_filter_frequency(fps)

        output_path = output_dir / f"{args.pair_id}_pipeline_{args.crop_source}_{args.arm_source}.mp4"
        for arkit_row in selected_rows:
            video_frame_index = int(arkit_row.get("frame_index", start_frame + processed))
            bgr = rv.read_frame_by_index(cap, video_frame_index)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            video_time = float(arkit_row.get("timestamp", arkit_start_timestamp)) - arkit_start_timestamp
            hand_pose_row = rv.row_nearest_frame_index(hand_pose_rows, video_frame_index)
            inference.set_camera_model(rv.camera_model_from_arkit_row(arkit_row), undistort_inp=False)
            depth, confidence = rv.read_depth_confidence_maps(arkit_archive, archive_root, arkit_row)

            with contextlib.redirect_stdout(io.StringIO()):
                boxes = get_boxes(inference, rgb.copy(), args.crop_source, hand_pose_row, args.arm_source)
                outs = run_outputs_with_boxes(inference, rgb.copy(), boxes)
            outs_lidar = apply_lidar_anchor_from_egoforce_direct(outs, depth, confidence, inference.camera_model, rgb.shape)

            anchors = outs_lidar.get("lidar_depth_anchor", {})
            diagnostics = [
                f"pair: {args.pair_id}",
                f"t={video_time:.3f}s  frame={video_frame_index}",
                f"crop source: {args.crop_source}  arm: {args.arm_source}",
                "ARKit hand landmarks are not drawn",
            ]
            for hdx, side in enumerate(("left", "right")):
                hand = boxes.get(side, {}).get("hand")
                arm = boxes.get(side, {}).get("arm")
                z0 = rv.get_pred_translation(outs["pred_transl"], hdx)[2]
                z1 = rv.get_pred_translation(outs_lidar["pred_transl"], hdx)[2]
                anchor = anchors.get(side)
                diagnostics.append(
                    f"{side}: hand={'yes' if hand is not None else 'no'} arm={'yes' if arm is not None else 'no'} "
                    f"z {z0:.3f}->{z1:.3f}"
                )
                if anchor:
                    diagnostics.append(f"  lidar pts={anchor['points']} depth_med={anchor['depth_median']:.3f}m")
                else:
                    diagnostics.append("  lidar anchor skipped/no valid samples")

            frame = compose_frame(rgb, depth, confidence, boxes, outs, outs_lidar, diagnostics)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (frame.shape[1], frame.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create output video: {output_path}")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            processed += 1
            if processed == 1 or processed % max(1, int(round(fps))) == 0:
                print(f"{args.pair_id}: processed {processed} frames")
    finally:
        cap.release()
        arkit_archive.close()
        if writer is not None:
            writer.release()

    if processed == 0:
        raise RuntimeError("No frames processed")
    rv.transcode_browser_mp4(output_path)
    print(f"Wrote {output_path} ({processed} frames)")


if __name__ == "__main__":
    main()
