#!/usr/bin/env python
import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGOFORCE_DISABLE_TRT", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

ROOT_DIR = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT_DIR / "demo"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import cv2
import numpy as np

from camera_models import PinholeCameraModel
from demo_utils import create_square_bbox


MAX_STACKED_OUTPUT_WIDTH = (1080 * 3) - 2
_TORCH = None
_INFERENCE_CLASS = None
_INFER_FN = None
_CFG = None
HAND_JOINT_ORDER = [
    "wrist",
    "thumb_cmc",
    "thumb_mp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "little_mcp",
    "little_pip",
    "little_dip",
    "little_tip",
]
HAND_SKELETON_EDGES = [
    ("wrist", "thumb_cmc"),
    ("thumb_cmc", "thumb_mp"),
    ("thumb_mp", "thumb_ip"),
    ("thumb_ip", "thumb_tip"),
    ("wrist", "index_mcp"),
    ("index_mcp", "index_pip"),
    ("index_pip", "index_dip"),
    ("index_dip", "index_tip"),
    ("wrist", "middle_mcp"),
    ("middle_mcp", "middle_pip"),
    ("middle_pip", "middle_dip"),
    ("middle_dip", "middle_tip"),
    ("wrist", "ring_mcp"),
    ("ring_mcp", "ring_pip"),
    ("ring_pip", "ring_dip"),
    ("ring_dip", "ring_tip"),
    ("wrist", "little_mcp"),
    ("little_mcp", "little_pip"),
    ("little_pip", "little_dip"),
    ("little_dip", "little_tip"),
]
APPLE_HAND_COLORS = {
    "left": (60, 255, 80),
    "right": (255, 230, 40),
}
EGOFORCE_HAND_COLORS = {
    "left": (255, 55, 55),
    "right": (55, 145, 255),
}
ARKIT_CROP_CONFIDENCE_THRESHOLD = 0.35
ARKIT_CROP_PADDING_FACTOR = 0.08
LIDAR_DEPTH_CONFIDENCE_THRESHOLD = 1
LIDAR_DEPTH_SAMPLE_RADIUS = 2
LIDAR_DEPTH_PERCENTILE = 25
LIDAR_MIN_ANCHOR_POINTS = 6
LIDAR_MAX_ANCHOR_TRANSLATION_DELTA = 0.35
SYNTHETIC_ARM_HALF_WIDTH_FACTOR = 0.85
SYNTHETIC_ARM_TOP_FACTOR = 0.25
SYNTHETIC_ARM_LENGTH_FACTOR = 1.9
YOLO_CROP_PADDING_FACTOR = 0.12
HYBRID_MIN_ARKIT_YOLO_IOU = 0.05
HYBRID_MAX_CENTER_DISTANCE_FACTOR = 0.85
HYBRID_MIN_CENTER_DISTANCE_PIXELS = 160.0
MARKER_CLEANUP_BOX_MARGIN = 10
MARKER_CLEANUP_MAX_ROI_FRACTION = 0.18
EGOFORCE_MIN_HAND_WEIGHT_SUM = 0.05
EGOFORCE_DRAW_CONFIDENCE_THRESHOLD = 0.004
EGOFORCE_MIN_DRAW_JOINTS = 6


def ensure_egoforce_runtime():
    global _TORCH, _INFERENCE_CLASS, _INFER_FN, _CFG
    if _INFERENCE_CLASS is None:
        import torch as torch_module
        from inference import Inference, infer
        from settings import config as cfg

        _TORCH = torch_module
        _INFERENCE_CLASS = Inference
        _INFER_FN = infer
        _CFG = cfg
    return _TORCH, _INFERENCE_CLASS, _INFER_FN, _CFG


def clamp_stacked_output_frame_size(frame_rgb):
    height, width = frame_rgb.shape[:2]
    if width <= MAX_STACKED_OUTPUT_WIDTH:
        return frame_rgb

    scale = MAX_STACKED_OUTPUT_WIDTH / float(width)
    target_width = max(2, int(round(width * scale)))
    target_height = max(2, int(round(height * scale)))
    target_width -= target_width % 2
    target_height -= target_height % 2
    return cv2.resize(frame_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)


def draw_panel_label(image, label):
    output = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.9, min(image.shape[:2]) / 900.0)
    thickness = max(2, int(round(scale * 2)))
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    pad = int(round(12 * scale))
    x, y = pad, pad + text_height
    cv2.rectangle(
        output,
        (x - pad // 2, y - text_height - pad // 2),
        (x + text_width + pad // 2, y + baseline + pad // 2),
        (0, 0, 0),
        -1,
    )
    cv2.putText(output, label, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return output


def compose_comparison_split(rgb_image, egoforce_rgb, arkit_rgb, combined_rgb):
    panels = [
        draw_panel_label(arkit_rgb, "ARKit"),
        draw_panel_label(egoforce_rgb, "EgoForce"),
        draw_panel_label(combined_rgb, "Both"),
    ]
    return clamp_stacked_output_frame_size(np.hstack(panels))


def transcode_browser_mp4(video_path):
    video_path = Path(video_path)
    temp_path = video_path.with_name(f"{video_path.stem}.h264.tmp{video_path.suffix}")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(temp_path),
    ]
    try:
        subprocess.run(command, check=True)
        temp_path.replace(video_path)
        print(f"Transcoded browser-playable H.264 MP4: {video_path}")
    except FileNotFoundError:
        print("ffmpeg not found; leaving OpenCV mp4v output in place.")
    except subprocess.CalledProcessError as exc:
        print(f"ffmpeg H.264 transcode failed for {video_path}: {exc}")
        if temp_path.exists():
            temp_path.unlink()

    return video_path


def iter_pair_dirs(dataset_root):
    for pair_dir in sorted(Path(dataset_root).glob("pair-*")):
        if (pair_dir / "video.mp4").exists() and (pair_dir / "arkit.zip").exists():
            yield pair_dir


def read_jsonl_archive_member(archive, suffix):
    name = next(name for name in archive.namelist() if name.endswith(suffix))
    with archive.open(name) as handle:
        return [json.loads(raw_line) for raw_line in handle]


def read_arkit_metadata(arkit_zip_path):
    with zipfile.ZipFile(arkit_zip_path) as archive:
        frames = read_jsonl_archive_member(archive, "/frames.jsonl")
        hand_poses = read_jsonl_archive_member(archive, "/apple/hand_poses.jsonl")
    return frames, hand_poses


def archive_root_from_zip(archive):
    frame_member = next(name for name in archive.namelist() if name.endswith("/frames.jsonl"))
    return frame_member.rsplit("/", 1)[0]


def archive_relative_path(archive_root, relative_path):
    if relative_path.startswith(f"{archive_root}/"):
        return relative_path
    return f"{archive_root}/{relative_path}"


def row_at_or_after(rows, target_frame_index):
    if not rows:
        return None
    selected = rows[-1]
    for row in rows:
        if int(row.get("frame_index", 0)) >= target_frame_index:
            selected = row
            break
    return selected


def row_nearest_timestamp(rows, target_timestamp):
    if not rows:
        return None

    lo, hi = 0, len(rows)
    while lo < hi:
        mid = (lo + hi) // 2
        if float(rows[mid].get("timestamp", 0.0)) < target_timestamp:
            lo = mid + 1
        else:
            hi = mid

    candidates = []
    if lo < len(rows):
        candidates.append(rows[lo])
    if lo > 0:
        candidates.append(rows[lo - 1])
    if not candidates:
        return None

    return min(candidates, key=lambda row: abs(float(row.get("timestamp", 0.0)) - target_timestamp))


def row_nearest_frame_index(rows, frame_index):
    if not rows:
        return None

    lo, hi = 0, len(rows)
    while lo < hi:
        mid = (lo + hi) // 2
        if int(rows[mid].get("frame_index", 0)) < frame_index:
            lo = mid + 1
        else:
            hi = mid

    candidates = []
    if lo < len(rows):
        candidates.append(rows[lo])
    if lo > 0:
        candidates.append(rows[lo - 1])
    if not candidates:
        return None

    return min(candidates, key=lambda row: abs(int(row.get("frame_index", 0)) - frame_index))


def hand_pose_nearest(hand_pose_rows, frame_index):
    return row_nearest_frame_index(hand_pose_rows, frame_index)


def arkit_rows_for_time_window(
    arkit_frames,
    arkit_start_timestamp,
    start_seconds,
    seconds=None,
    *,
    stride=1,
    max_frames=None,
):
    start_timestamp = float(arkit_start_timestamp) + max(0.0, float(start_seconds))
    end_timestamp = None
    if seconds is not None and float(seconds) >= 0:
        end_timestamp = start_timestamp + float(seconds)

    selected = []
    for row in arkit_frames:
        timestamp = row.get("timestamp")
        if timestamp is None:
            continue
        timestamp = float(timestamp)
        if timestamp < start_timestamp:
            continue
        if end_timestamp is not None and timestamp >= end_timestamp:
            continue
        selected.append(row)

    if not selected:
        nearest = row_nearest_timestamp(arkit_frames, start_timestamp)
        if nearest is not None:
            selected = [nearest]

    selected = selected[:: max(1, int(stride))]
    if max_frames is not None:
        selected = selected[: max(0, int(max_frames))]
    return selected


def fps_from_arkit_rows(rows, fallback_fps):
    if len(rows) < 2:
        return float(fallback_fps)

    first_timestamp = rows[0].get("timestamp")
    last_timestamp = rows[-1].get("timestamp")
    if first_timestamp is None or last_timestamp is None:
        return float(fallback_fps)

    duration = float(last_timestamp) - float(first_timestamp)
    if duration <= 0:
        return float(fallback_fps)
    return float((len(rows) - 1) / duration)


def read_frame_by_index(capture, frame_index):
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, bgr_image = capture.read()
    if not ok:
        return None
    return bgr_image


class IndexedVideoFrameReader:
    """Read monotonically indexed frames by decoding forward when possible."""

    def __init__(self, capture, sequential_skip_limit=None):
        self.capture = capture
        self.next_frame_index = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
        self.sequential_skip_limit = None if sequential_skip_limit is None else int(sequential_skip_limit)

    def read(self, frame_index):
        frame_index = int(frame_index)
        if self.next_frame_index is not None and frame_index >= self.next_frame_index:
            skip_count = frame_index - self.next_frame_index
            if self.sequential_skip_limit is None or skip_count <= self.sequential_skip_limit:
                for _ in range(skip_count):
                    if not self.capture.grab():
                        self.next_frame_index = None
                        break
                else:
                    ok, bgr_image = self.capture.read()
                    if not ok:
                        self.next_frame_index = None
                        return None
                    self.next_frame_index = frame_index + 1
                    return bgr_image

        bgr_image = read_frame_by_index(self.capture, frame_index)
        self.next_frame_index = frame_index + 1 if bgr_image is not None else None
        return bgr_image


def camera_model_from_arkit_row(row):
    intrinsics_cols = row["camera_intrinsics_3x3"]
    fx = float(intrinsics_cols[0][0])
    fy = float(intrinsics_cols[1][1])
    cx = float(intrinsics_cols[2][0])
    cy = float(intrinsics_cols[2][1])
    width, height = [int(v) for v in row["image_resolution"]]

    camera_model = PinholeCameraModel(
        np.array([fx, fy], dtype=np.float32),
        np.array([cx, cy], dtype=np.float32),
        width,
        height,
    )
    return camera_model


def read_arkit_camera_model(arkit_zip_path, target_frame_index=0):
    frames, _ = read_arkit_metadata(arkit_zip_path)
    row = row_at_or_after(frames, target_frame_index)
    if row is None:
        raise ValueError(f"No frames found in {arkit_zip_path}")
    return camera_model_from_arkit_row(row), row


def get_capture_fps(capture):
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        return 30.0
    return float(fps)


def draw_hand_skeleton(image, points_by_name, color, confidence_threshold=0.0):
    for a, b in HAND_SKELETON_EDGES:
        point_a = points_by_name.get(a)
        point_b = points_by_name.get(b)
        if point_a is None or point_b is None:
            continue
        if point_a[2] < confidence_threshold or point_b[2] < confidence_threshold:
            continue
        cv2.line(
            image,
            tuple(np.round(point_a[:2]).astype(int)),
            tuple(np.round(point_b[:2]).astype(int)),
            color,
            3,
            cv2.LINE_AA,
        )

    for point in points_by_name.values():
        if point[2] < confidence_threshold:
            continue
        cv2.circle(
            image,
            tuple(np.round(point[:2]).astype(int)),
            5,
            color,
            -1,
            cv2.LINE_AA,
        )


def draw_apple_hand_overlay(rgb_image, hand_pose_row, confidence_threshold=0.35):
    if hand_pose_row is None:
        return rgb_image

    height, width = rgb_image.shape[:2]
    overlay = rgb_image.copy()
    hands = list(hand_pose_row.get("hands", []))
    screen_ordered_hands = sorted(
        hands,
        key=lambda hand: float(hand.get("joints", {}).get("wrist", {}).get("x", 0.5)),
    )
    position_colors = {}
    if len(screen_ordered_hands) >= 2:
        position_colors[id(screen_ordered_hands[0])] = APPLE_HAND_COLORS["left"]
        position_colors[id(screen_ordered_hands[-1])] = APPLE_HAND_COLORS["right"]

    for hand in hands:
        chirality = hand.get("chirality", "left")
        color = position_colors.get(id(hand), APPLE_HAND_COLORS.get(chirality, (255, 255, 255)))
        points_by_name = {}
        for joint_name, joint in hand.get("joints", {}).items():
            # Apple/Vision-style normalized points are bottom-left origin.
            x = float(joint["x"]) * width
            y = (1.0 - float(joint["y"])) * height
            confidence = float(joint.get("confidence", hand.get("confidence", 1.0)))
            points_by_name[joint_name] = np.array([x, y, confidence], dtype=np.float32)
        draw_hand_skeleton(overlay, points_by_name, color, confidence_threshold)

    return cv2.addWeighted(overlay, 0.85, rgb_image, 0.15, 0)


def arkit_hand_points(hand, width, height):
    points_by_name = {}
    for joint_name, joint in hand.get("joints", {}).items():
        if "x" not in joint or "y" not in joint:
            continue
        # Apple/Vision-style normalized points are bottom-left origin.
        x = float(joint["x"]) * width
        y = (1.0 - float(joint["y"])) * height
        confidence = float(joint.get("confidence", hand.get("confidence", 1.0)))
        points_by_name[joint_name] = np.array([x, y, confidence], dtype=np.float32)
    return points_by_name


def arkit_points_to_array(points_by_name, fallback_xy):
    fallback = np.asarray(fallback_xy, dtype=np.float32)
    points = []
    for joint_name in HAND_JOINT_ORDER:
        point = points_by_name.get(joint_name)
        if point is None or not np.isfinite(point[:2]).all():
            points.append(fallback)
        else:
            points.append(point[:2])
    return np.asarray(points, dtype=np.float32)


def arkit_confidence_to_array(points_by_name):
    confidences = []
    for joint_name in HAND_JOINT_ORDER:
        point = points_by_name.get(joint_name)
        confidences.append(0.0 if point is None else float(point[2]))
    return np.asarray(confidences, dtype=np.float32)


def arkit_hand_records(hand_pose_row, image_shape):
    if hand_pose_row is None:
        return []

    height, width = image_shape[:2]
    records = []
    for hand in hand_pose_row.get("hands", []):
        points_by_name = arkit_hand_points(hand, width, height)
        confident_points = [
            point[:2]
            for point in points_by_name.values()
            if point[2] >= ARKIT_CROP_CONFIDENCE_THRESHOLD and np.isfinite(point[:2]).all()
        ]
        if len(confident_points) < 6:
            continue

        confident_points = np.asarray(confident_points, dtype=np.float32)
        bbox = np.asarray(
            create_square_bbox(confident_points, width, height, ARKIT_CROP_PADDING_FACTOR),
            dtype=np.float32,
        )
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue

        center = confident_points.mean(axis=0)
        keypoints = arkit_points_to_array(points_by_name, center)
        confidences = arkit_confidence_to_array(points_by_name)
        records.append(
            {
                "bbox": bbox,
                "keypoint": keypoints,
                "conf": confidences,
                "score": float(np.mean(confidences)),
                "x_center": float(0.5 * (bbox[0] + bbox[2])),
                "handedness": hand.get("chirality"),
                "source": "arkit",
            }
        )
    return records


def assign_arkit_hand_records(records, width, side_source="screen"):
    bounding_boxes = {"left": {}, "right": {}}
    if not records:
        return bounding_boxes

    if side_source == "chirality":
        for record in sorted(records, key=lambda item: item["score"], reverse=True):
            side = record.get("handedness")
            if side in bounding_boxes and "hand" not in bounding_boxes[side]:
                bounding_boxes[side]["hand"] = dict(record, handedness=side)
        return bounding_boxes

    ordered = sorted(records, key=lambda item: item["x_center"])
    if len(ordered) >= 2:
        bounding_boxes["left"]["hand"] = dict(ordered[0], handedness="left")
        bounding_boxes["right"]["hand"] = dict(ordered[-1], handedness="right")
    else:
        record = ordered[0]
        if side_source == "screen":
            side = "left" if record["x_center"] < width * 0.5 else "right"
        else:
            chirality = record.get("handedness")
            side = chirality if chirality in bounding_boxes else ("left" if record["x_center"] < width * 0.5 else "right")
        bounding_boxes[side]["hand"] = dict(record, handedness=side)

    return bounding_boxes


def bbox_iou(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0

    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else float(intersection / union)


def bbox_center_and_extent(bbox):
    bbox = np.asarray(bbox, dtype=np.float32)
    center = np.asarray([0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])], dtype=np.float32)
    extent = max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]), 1.0)
    return center, extent


def square_bbox_from_xyxy(xyxy, image_width, image_height, padding_factor):
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    side = max(x2 - x1, y2 - y1, 1.0) * (1.0 + 2.0 * padding_factor)
    half = 0.5 * side
    return np.asarray(
        [
            max(0.0, cx - half),
            max(0.0, cy - half),
            min(float(image_width - 1), cx + half),
            min(float(image_height - 1), cy + half),
        ],
        dtype=np.float32,
    )


def clipped_bbox_with_margin(bbox, image_shape, margin=0):
    image_height, image_width = image_shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox]
    x1 = int(max(0, np.floor(x1 - margin)))
    y1 = int(max(0, np.floor(y1 - margin)))
    x2 = int(min(image_width, np.ceil(x2 + margin)))
    y2 = int(min(image_height, np.ceil(y2 + margin)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def colored_marker_mask(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    yellow = (hue >= 18) & (hue <= 36) & (saturation >= 105) & (value >= 135)
    green_to_magenta = (hue > 36) & (hue <= 175) & (saturation >= 70) & (value >= 100)
    return yellow | green_to_magenta


def strict_colored_marker_mask(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    yellow = (hue >= 18) & (hue <= 36) & (saturation >= 130) & (value >= 155)
    green_to_magenta = (hue > 36) & (hue <= 175) & (saturation >= 95) & (value >= 125)
    return yellow | green_to_magenta


def clean_model_input_rgb(rgb_image, bounding_boxes, cleanup_mode="none"):
    if cleanup_mode == "none":
        return rgb_image, None
    if cleanup_mode != "inpaint-markers":
        raise ValueError(f"Unknown EgoForce input cleanup mode: {cleanup_mode}")

    loose_mask = colored_marker_mask(rgb_image)
    strict_mask = strict_colored_marker_mask(rgb_image)
    limited_mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)

    for side in ("left", "right"):
        record = bounding_boxes.get(side, {}).get("hand")
        if record is None:
            continue

        clipped = clipped_bbox_with_margin(
            record["bbox"],
            rgb_image.shape,
            margin=MARKER_CLEANUP_BOX_MARGIN,
        )
        if clipped is None:
            continue

        x1, y1, x2, y2 = clipped
        roi_mask = loose_mask[y1:y2, x1:x2]
        if roi_mask.size == 0:
            continue

        if float(np.count_nonzero(roi_mask)) / float(roi_mask.size) > MARKER_CLEANUP_MAX_ROI_FRACTION:
            roi_mask = strict_mask[y1:y2, x1:x2]
        limited_mask[y1:y2, x1:x2] = np.maximum(
            limited_mask[y1:y2, x1:x2],
            roi_mask.astype(np.uint8) * 255,
        )

    if not np.any(limited_mask):
        return rgb_image, limited_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    limited_mask = cv2.dilate(limited_mask, kernel, iterations=1)
    cleaned_bgr = cv2.inpaint(
        cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR),
        limited_mask,
        5,
        cv2.INPAINT_TELEA,
    )
    return cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB), limited_mask


def yolo_screen_bounding_boxes(inference, rgb_image):
    torch, _, _, _ = ensure_egoforce_runtime()
    detector_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    yolo_kwargs = dict(
        verbose=False,
        half=inference.device.type == "cuda",
        device=inference.device,
        conf=0.25,
        iou=0.50,
    )
    with torch.no_grad():
        if getattr(inference, "yolo_use_track", False):
            yolo_res = inference.hand_detector.track(
                detector_image,
                persist=getattr(inference, "yolo_track_persist", False),
                **yolo_kwargs,
            )[0]
        else:
            yolo_res = inference.hand_detector.predict(detector_image, **yolo_kwargs)[0]

    boxes = yolo_res.boxes
    if boxes is None or len(boxes) == 0:
        return {"left": {}, "right": {}}

    xyxy = boxes.xyxy.cpu().numpy().astype(np.float32, copy=False)
    confs = boxes.conf.cpu().numpy().astype(np.float32, copy=False)
    classes = boxes.cls.cpu().numpy().astype(np.int64, copy=False)
    image_height, image_width = rgb_image.shape[:2]

    records = []
    for bbox_xyxy, confidence, cls_idx in zip(xyxy, confs, classes):
        bbox = square_bbox_from_xyxy(bbox_xyxy, image_width, image_height, YOLO_CROP_PADDING_FACTOR)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue

        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])
        # The YOLO hand model's handedness labels are unreliable on these clips,
        # so use screen order for side assignment and keep keypoints bbox-derived.
        keypoints = np.asarray(
            [
                [cx, bbox[3]],
                [cx, cy],
                [cx, bbox[1]],
            ],
            dtype=np.float32,
        )
        records.append(
            {
                "bbox": bbox,
                "keypoint": keypoints,
                "conf": np.full((3,), float(confidence), dtype=np.float32),
                "score": float(confidence),
                "x_center": float(cx),
                "handedness": "left" if int(cls_idx) == 0 else "right",
                "source": "yolo-screen",
            }
        )

    records = sorted(records, key=lambda record: record["score"], reverse=True)[:2]
    return assign_arkit_hand_records(records, image_width, side_source="screen")


def arkit_record_matches_yolo(arkit_record, yolo_record):
    arkit_bbox = np.asarray(arkit_record["bbox"], dtype=np.float32)
    yolo_bbox = np.asarray(yolo_record["bbox"], dtype=np.float32)
    if bbox_iou(arkit_bbox, yolo_bbox) >= HYBRID_MIN_ARKIT_YOLO_IOU:
        return True

    arkit_center, arkit_extent = bbox_center_and_extent(arkit_bbox)
    yolo_center, yolo_extent = bbox_center_and_extent(yolo_bbox)
    center_distance = float(np.linalg.norm(arkit_center - yolo_center))
    max_allowed = max(
        HYBRID_MIN_CENTER_DISTANCE_PIXELS,
        HYBRID_MAX_CENTER_DISTANCE_FACTOR * min(arkit_extent, yolo_extent),
    )
    return center_distance <= max_allowed


def hybrid_arkit_yolo_bounding_boxes(inference, rgb_image, hand_pose_row, arkit_side_source="screen"):
    arkit_boxes = arkit_bounding_boxes(hand_pose_row, rgb_image.shape, side_source=arkit_side_source)
    yolo_boxes = yolo_screen_bounding_boxes(inference, rgb_image)
    output = {"left": {}, "right": {}}

    for side in ("left", "right"):
        arkit_record = arkit_boxes.get(side, {}).get("hand")
        yolo_record = yolo_boxes.get(side, {}).get("hand")
        if arkit_record is not None:
            output[side]["hand"] = arkit_record
        elif yolo_record is not None:
            record = dict(yolo_record)
            record["source"] = "hybrid-yolo"
            output[side]["hand"] = record
        elif arkit_record is not None:
            output[side]["hand"] = arkit_record

    return output


def arkit_bounding_boxes(hand_pose_row, image_shape, side_source="screen"):
    width = image_shape[1]
    records = arkit_hand_records(hand_pose_row, image_shape)
    return assign_arkit_hand_records(records, width, side_source=side_source)


def add_synthetic_arm_records(bounding_boxes, image_shape):
    image_height, image_width = image_shape[:2]
    output = {side: dict(records) for side, records in bounding_boxes.items()}
    for side in ("left", "right"):
        hand = output.get(side, {}).get("hand")
        if hand is None:
            continue

        hand_bbox = np.asarray(hand["bbox"], dtype=np.float32)
        hand_keypoints = np.asarray(hand.get("keypoint", []), dtype=np.float32)
        if len(hand_keypoints):
            wrist = hand_keypoints[0]
        else:
            wrist = np.asarray(
                [0.5 * (hand_bbox[0] + hand_bbox[2]), 0.5 * (hand_bbox[1] + hand_bbox[3])],
                dtype=np.float32,
            )

        hand_size = max(float(hand_bbox[2] - hand_bbox[0]), float(hand_bbox[3] - hand_bbox[1]), 50.0)
        x1 = max(0.0, float(wrist[0] - SYNTHETIC_ARM_HALF_WIDTH_FACTOR * hand_size))
        x2 = min(float(image_width - 1), float(wrist[0] + SYNTHETIC_ARM_HALF_WIDTH_FACTOR * hand_size))
        y1 = max(0.0, float(wrist[1] - SYNTHETIC_ARM_TOP_FACTOR * hand_size))
        y2 = min(float(image_height - 1), float(wrist[1] + SYNTHETIC_ARM_LENGTH_FACTOR * hand_size))
        if y2 <= y1 or x2 <= x1:
            continue

        arm_keypoints = np.asarray(
            [
                [wrist[0], min(float(image_height - 1), wrist[1] + SYNTHETIC_ARM_LENGTH_FACTOR * hand_size)],
                [wrist[0], min(float(image_height - 1), wrist[1] + 0.8 * hand_size)],
                wrist,
            ],
            dtype=np.float32,
        )
        output[side]["arm"] = {
            "bbox": np.asarray([x1, y1, x2, y2], dtype=np.float32),
            "keypoint": arm_keypoints,
            "score": 0.5,
            "x_center": float(0.5 * (x1 + x2)),
            "handedness": side,
            "source": "synthetic-arkit",
        }
    return output


def read_depth_confidence_maps(archive, archive_root, arkit_row):
    if arkit_row is None or not arkit_row.get("has_depth"):
        return None, None

    depth_path = archive_relative_path(archive_root, arkit_row["depth_path"])
    confidence_path = archive_relative_path(archive_root, arkit_row["confidence_path"])
    depth_width, depth_height = [int(value) for value in arkit_row["depth_resolution"]]

    depth = np.frombuffer(archive.read(depth_path), dtype=np.float16).astype(np.float32)
    depth = depth.reshape(depth_height, depth_width)

    confidence_bytes = np.frombuffer(archive.read(confidence_path), dtype=np.uint8)
    confidence = cv2.imdecode(confidence_bytes, cv2.IMREAD_UNCHANGED)
    return depth, confidence


def sample_depth_at_uv(depth, confidence, uv, image_shape):
    if depth is None:
        return None

    image_height, image_width = image_shape[:2]
    depth_height, depth_width = depth.shape[:2]
    u, v = [float(value) for value in uv]
    if not np.isfinite([u, v]).all():
        return None

    x = int(round(u * depth_width / max(float(image_width), 1.0)))
    y = int(round(v * depth_height / max(float(image_height), 1.0)))
    if x < 0 or y < 0 or x >= depth_width or y >= depth_height:
        return None

    x1 = max(0, x - LIDAR_DEPTH_SAMPLE_RADIUS)
    x2 = min(depth_width, x + LIDAR_DEPTH_SAMPLE_RADIUS + 1)
    y1 = max(0, y - LIDAR_DEPTH_SAMPLE_RADIUS)
    y2 = min(depth_height, y + LIDAR_DEPTH_SAMPLE_RADIUS + 1)

    patch = depth[y1:y2, x1:x2]
    valid = np.isfinite(patch) & (patch > 0.05) & (patch < 5.0)
    if confidence is not None:
        valid &= confidence[y1:y2, x1:x2] >= LIDAR_DEPTH_CONFIDENCE_THRESHOLD

    values = patch[valid]
    if values.size == 0:
        return None
    return float(np.percentile(values, LIDAR_DEPTH_PERCENTILE))


def camera_xyz_from_uv_z(camera_model, uv, z):
    f = camera_model.f.detach().cpu().numpy() if hasattr(camera_model.f, "detach") else np.asarray(camera_model.f)
    c = camera_model.c.detach().cpu().numpy() if hasattr(camera_model.c, "detach") else np.asarray(camera_model.c)
    uv = np.asarray(uv, dtype=np.float32)
    return np.asarray(
        [
            (uv[0] - c[0]) / f[0] * z,
            (uv[1] - c[1]) / f[1] * z,
            z,
        ],
        dtype=np.float32,
    )


def get_pred_translation(pred_transl, hdx):
    translation = np.asarray(pred_transl)
    if translation.ndim == 3:
        return translation[hdx, 0].astype(np.float32, copy=False)
    return translation[hdx].astype(np.float32, copy=False)


def set_pred_translation(pred_transl, hdx, value):
    if pred_transl.ndim == 3:
        pred_transl[hdx, 0] = value
    else:
        pred_transl[hdx] = value


def apply_lidar_depth_anchor(outs, bounding_boxes, depth, confidence, camera_model, image_shape):
    if depth is None or bounding_boxes is None or "pred_transl" not in outs:
        return outs

    anchors = {}
    for hdx, side in enumerate(("left", "right")):
        record = bounding_boxes.get(side, {}).get("hand")
        if record is None:
            continue

        keypoints = np.asarray(record.get("keypoint", []), dtype=np.float32)
        confidences = np.asarray(record.get("conf", np.ones(len(keypoints))), dtype=np.float32)
        if len(keypoints) < len(HAND_JOINT_ORDER):
            continue

        old_translation = get_pred_translation(outs["pred_transl"], hdx).copy()
        relative_joints = np.asarray(outs["pred_j3d"][hdx], dtype=np.float32) - old_translation[None, :]

        deltas = []
        sampled_depths = []
        for jdx, uv in enumerate(keypoints[: len(HAND_JOINT_ORDER)]):
            if jdx >= len(relative_joints):
                break
            if jdx < len(confidences) and confidences[jdx] < ARKIT_CROP_CONFIDENCE_THRESHOLD:
                continue

            depth_z = sample_depth_at_uv(depth, confidence, uv, image_shape)
            if depth_z is None:
                continue

            target_xyz = camera_xyz_from_uv_z(camera_model, uv, depth_z)
            deltas.append(target_xyz - relative_joints[jdx])
            sampled_depths.append(depth_z)

        if len(deltas) < LIDAR_MIN_ANCHOR_POINTS:
            continue

        new_translation = np.median(np.asarray(deltas, dtype=np.float32), axis=0)
        if not np.isfinite(new_translation).all() or new_translation[2] <= 0.05:
            continue

        delta = new_translation - old_translation
        if np.linalg.norm(delta) > LIDAR_MAX_ANCHOR_TRANSLATION_DELTA:
            continue
        for key in ("pred_j3d", "pred_vertices", "pred_arm_j3d", "pred_arm_vertices"):
            if key in outs:
                outs[key][hdx] = outs[key][hdx] + delta

        set_pred_translation(outs["pred_transl"], hdx, new_translation)
        if "pred_j2d" in outs:
            outs["pred_j2d"][hdx] = camera_model.camera_to_uv(outs["pred_j3d"][hdx])
        if "pred_arm_j2d" in outs and "pred_arm_j3d" in outs:
            outs["pred_arm_j2d"][hdx] = camera_model.camera_to_uv(outs["pred_arm_j3d"][hdx])

        anchors[side] = {
            "points": len(deltas),
            "depth_median": float(np.median(sampled_depths)),
            "old_translation": old_translation.tolist(),
            "new_translation": new_translation.tolist(),
        }

    outs["lidar_depth_anchor"] = anchors
    return outs


def projected_points_are_unstable(points, bbox, image_shape):
    points = np.asarray(points, dtype=np.float32)
    finite = np.isfinite(points).all(axis=1)
    if finite.sum() < 6:
        return True

    visible_points = points[finite]
    bbox = np.asarray(bbox, dtype=np.float32)
    box_width = max(float(bbox[2] - bbox[0]), 1.0)
    box_height = max(float(bbox[3] - bbox[1]), 1.0)
    box_extent = max(box_width, box_height)
    box_diag = float(np.hypot(box_width, box_height))
    box_center = np.array([0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])], dtype=np.float32)

    point_center = visible_points.mean(axis=0)
    point_extent = max(
        float(visible_points[:, 0].max() - visible_points[:, 0].min()),
        float(visible_points[:, 1].max() - visible_points[:, 1].min()),
    )
    center_distance = float(np.linalg.norm(point_center - box_center))
    image_height, image_width = image_shape[:2]

    if point_extent > max(3.0 * box_extent, 1200.0):
        return True
    if center_distance > max(2.0 * box_diag, 1000.0):
        return True
    if np.abs(visible_points).max() > 4.0 * max(image_width, image_height):
        return True
    return False


def select_egoforce_overlay_points(outs, hdx, side, bounding_boxes, image_shape, overlay_space):
    camera_points = np.asarray(outs["pred_j2d"][hdx], dtype=np.float32)
    direct_points = np.asarray(outs.get("pred_hand_j2d_direct", outs["pred_j2d"])[hdx], dtype=np.float32)

    if overlay_space == "camera":
        return camera_points
    if overlay_space == "direct-2d":
        return direct_points

    record = None if bounding_boxes is None else bounding_boxes.get(side, {}).get("hand")
    if record is not None and projected_points_are_unstable(camera_points, record["bbox"], image_shape):
        return direct_points
    return camera_points


def draw_egoforce_hand_overlay(rgb_image, outs, bounding_boxes=None, overlay_space="auto"):
    overlay = rgb_image.copy()
    hand_weights = np.asarray(outs.get("pred_hand_kpt_w", []), dtype=np.float32)
    for hdx, side in enumerate(("left", "right")):
        color = EGOFORCE_HAND_COLORS[side]
        side_boxes = {} if bounding_boxes is None else bounding_boxes.get(side, {})

        if "hand" in side_boxes:
            weights = hand_weights[hdx] if hand_weights.ndim >= 2 and hdx < len(hand_weights) else None
            weight_sum = None if weights is None else float(np.nansum(weights))
            if weight_sum is None or weight_sum >= EGOFORCE_MIN_HAND_WEIGHT_SUM:
                points = select_egoforce_overlay_points(outs, hdx, side, bounding_boxes, rgb_image.shape, overlay_space)
                points_by_name = {}
                for jdx, joint_name in enumerate(HAND_JOINT_ORDER):
                    if jdx >= len(points) or not np.isfinite(points[jdx]).all():
                        continue
                    confidence = 1.0 if weights is None or jdx >= len(weights) else float(weights[jdx])
                    points_by_name[joint_name] = np.array([points[jdx, 0], points[jdx, 1], confidence], dtype=np.float32)

                if sum(point[2] >= EGOFORCE_DRAW_CONFIDENCE_THRESHOLD for point in points_by_name.values()) >= EGOFORCE_MIN_DRAW_JOINTS:
                    draw_hand_skeleton(
                        overlay,
                        points_by_name,
                        color,
                        confidence_threshold=EGOFORCE_DRAW_CONFIDENCE_THRESHOLD,
                    )

        for box_name in ("hand", "arm"):
            record = side_boxes.get(box_name)
            if record is None:
                continue
            x1, y1, x2, y2 = np.round(record["bbox"]).astype(int)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

    return cv2.addWeighted(overlay, 0.85, rgb_image, 0.15, 0)


def run_egoforce_outputs(
    inference,
    rgb_image,
    hand_pose_row=None,
    crop_source="yolo",
    arkit_side_source="screen",
    arm_source="none",
    input_cleanup="none",
):
    torch, _, infer_fn, cfg = ensure_egoforce_runtime()
    if crop_source == "arkit":
        bounding_boxes = arkit_bounding_boxes(hand_pose_row, rgb_image.shape, side_source=arkit_side_source)
    elif crop_source == "arkit-yolo-hybrid":
        bounding_boxes = hybrid_arkit_yolo_bounding_boxes(
            inference,
            rgb_image,
            hand_pose_row,
            arkit_side_source=arkit_side_source,
        )
    elif crop_source == "yolo":
        bounding_boxes = yolo_screen_bounding_boxes(inference, rgb_image)
    elif crop_source == "original":
        bounding_boxes = inference.detect_bounding_boxes(rgb_image)
    else:
        raise ValueError(f"Unknown EgoForce crop source: {crop_source}")

    if arm_source == "drop":
        for side in ("left", "right"):
            bounding_boxes.get(side, {}).pop("arm", None)
    elif arm_source == "synthetic-arkit":
        bounding_boxes = add_synthetic_arm_records(bounding_boxes, rgb_image.shape)
    elif arm_source != "none":
        raise ValueError(f"Unknown EgoForce arm source: {arm_source}")

    model_rgb, cleanup_mask = clean_model_input_rgb(rgb_image, bounding_boxes, input_cleanup)

    left_data = inference.left_dataset.transform(model_rgb, bounding_boxes["left"])
    right_data = inference.right_dataset.transform(model_rgb, bounding_boxes["right"])
    with torch.no_grad():
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            outs = infer_fn(inference, cfg, inference.model, inference.limb_model, left_data, right_data, inference.device)
    if cleanup_mask is not None:
        outs["input_cleanup"] = {
            "mode": input_cleanup,
            "masked_pixels": int(np.count_nonzero(cleanup_mask)),
        }
    return outs, bounding_boxes


def process_pair(
    pair_dir,
    inference,
    output_dir,
    start_seconds,
    seconds,
    max_frames,
    include_arm_mesh,
    viz_mode,
    egoforce_crop_source,
    egoforce_arm_source,
    arkit_side_source,
    egoforce_overlay_space,
    egoforce_depth_anchor,
    egoforce_input_cleanup,
):
    pair_dir = Path(pair_dir)
    video_path = pair_dir / "video.mp4"
    arkit_zip_path = pair_dir / "arkit.zip"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    writer = None
    processed = 0
    arkit_archive = None
    try:
        fps = get_capture_fps(capture)
        start_time_seconds = max(0.0, float(start_seconds))

        arkit_frames, hand_pose_rows = read_arkit_metadata(arkit_zip_path)
        arkit_archive = zipfile.ZipFile(arkit_zip_path)
        arkit_archive_root = archive_root_from_zip(arkit_archive)
        arkit_start_timestamp = float(arkit_frames[0].get("timestamp", 0.0))
        selected_rows = arkit_rows_for_time_window(
            arkit_frames,
            arkit_start_timestamp,
            start_time_seconds,
            seconds,
            max_frames=max_frames,
        )
        if not selected_rows:
            raise ValueError(f"No ARKit frames found in requested time window for {arkit_zip_path}")

        row_fps = fps_from_arkit_rows(selected_rows, fps)
        camera_row = selected_rows[0]
        start_frame = max(0, int(camera_row.get("frame_index", round(start_time_seconds * fps))))

        if inference is not None:
            inference.reset_runtime_state()
            inference.set_camera_model(camera_model_from_arkit_row(camera_row), undistort_inp=False)
            inference.set_kalman_filter_frequency(row_fps)

        crop_tag = (
            f"_{egoforce_crop_source}_crops"
            if viz_mode in {"egoforce-overlay", "comparison-overlay", "comparison-split"}
            else ""
        )
        output_path = output_dir / f"{pair_dir.name}_{viz_mode.replace('-', '_')}{crop_tag}.mp4"
        print(
            f"{pair_dir.name}: video_fps={fps:.3f}, row_fps={row_fps:.3f}, start_time={start_time_seconds:.3f}s, "
            f"start_frame_est={start_frame}, frame_limit={len(selected_rows)}, "
            f"arkit_frame={camera_row.get('frame_index')}, "
            f"arkit_dt={float(camera_row.get('timestamp', arkit_start_timestamp)) - (arkit_start_timestamp + start_time_seconds):.4f}s, "
            f"viz_mode={viz_mode}, egoforce_crop_source={egoforce_crop_source}, "
            f"egoforce_arm_source={egoforce_arm_source}, "
            f"egoforce_depth_anchor={egoforce_depth_anchor}, "
            f"egoforce_input_cleanup={egoforce_input_cleanup}",
            flush=True,
        )

        frame_reader = IndexedVideoFrameReader(capture)
        for arkit_row in selected_rows:
            video_frame_index = int(arkit_row.get("frame_index", start_frame + processed))
            bgr_image = frame_reader.read(video_frame_index)
            if bgr_image is None:
                continue
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            video_time_seconds = float(arkit_row.get("timestamp", arkit_start_timestamp)) - arkit_start_timestamp
            target_timestamp = arkit_start_timestamp + video_time_seconds
            if inference is not None:
                inference.set_camera_model(camera_model_from_arkit_row(arkit_row), undistort_inp=False)
            hand_pose_row = row_nearest_frame_index(hand_pose_rows, video_frame_index)
            if egoforce_depth_anchor == "lidar":
                depth, confidence = read_depth_confidence_maps(arkit_archive, arkit_archive_root, arkit_row)
            else:
                depth, confidence = None, None

            if viz_mode == "render":
                from demo_utils import compose_output_frame

                render_image, tp_image = inference.run(
                    rgb_image.copy(),
                    inference.device,
                    include_arm_mesh=include_arm_mesh,
                )
                output_rgb = compose_output_frame(rgb_image, render_image, tp_image)
                output_rgb = clamp_stacked_output_frame_size(output_rgb)
            elif viz_mode == "egoforce-overlay":
                outs, bounding_boxes = run_egoforce_outputs(
                    inference,
                    rgb_image.copy(),
                    hand_pose_row=hand_pose_row,
                    crop_source=egoforce_crop_source,
                    arkit_side_source=arkit_side_source,
                    arm_source=egoforce_arm_source,
                    input_cleanup=egoforce_input_cleanup,
                )
                if egoforce_depth_anchor == "lidar":
                    apply_lidar_depth_anchor(
                        outs,
                        bounding_boxes,
                        depth,
                        confidence,
                        inference.camera_model,
                        rgb_image.shape,
                    )
                output_rgb = draw_egoforce_hand_overlay(
                    rgb_image,
                    outs,
                    bounding_boxes,
                    overlay_space=egoforce_overlay_space,
                )
            elif viz_mode == "arkit-overlay":
                output_rgb = draw_apple_hand_overlay(rgb_image, hand_pose_row)
            elif viz_mode == "comparison-overlay":
                outs, bounding_boxes = run_egoforce_outputs(
                    inference,
                    rgb_image.copy(),
                    hand_pose_row=hand_pose_row,
                    crop_source=egoforce_crop_source,
                    arkit_side_source=arkit_side_source,
                    arm_source=egoforce_arm_source,
                    input_cleanup=egoforce_input_cleanup,
                )
                if egoforce_depth_anchor == "lidar":
                    apply_lidar_depth_anchor(
                        outs,
                        bounding_boxes,
                        depth,
                        confidence,
                        inference.camera_model,
                        rgb_image.shape,
                    )
                output_rgb = draw_egoforce_hand_overlay(
                    rgb_image,
                    outs,
                    bounding_boxes,
                    overlay_space=egoforce_overlay_space,
                )
                output_rgb = draw_apple_hand_overlay(output_rgb, hand_pose_row)
            elif viz_mode == "comparison-split":
                outs, bounding_boxes = run_egoforce_outputs(
                    inference,
                    rgb_image.copy(),
                    hand_pose_row=hand_pose_row,
                    crop_source=egoforce_crop_source,
                    arkit_side_source=arkit_side_source,
                    arm_source=egoforce_arm_source,
                    input_cleanup=egoforce_input_cleanup,
                )
                if egoforce_depth_anchor == "lidar":
                    apply_lidar_depth_anchor(
                        outs,
                        bounding_boxes,
                        depth,
                        confidence,
                        inference.camera_model,
                        rgb_image.shape,
                    )
                egoforce_rgb = draw_egoforce_hand_overlay(
                    rgb_image,
                    outs,
                    bounding_boxes,
                    overlay_space=egoforce_overlay_space,
                )
                arkit_rgb = draw_apple_hand_overlay(rgb_image, hand_pose_row)
                combined_rgb = draw_apple_hand_overlay(egoforce_rgb, hand_pose_row)
                output_rgb = compose_comparison_split(rgb_image, egoforce_rgb, arkit_rgb, combined_rgb)
            else:
                raise ValueError(f"Unknown viz mode: {viz_mode}")

            if writer is None:
                height, width = output_rgb.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    row_fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create output video: {output_path}")

            writer.write(cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))
            processed += 1
            if processed == 1 or processed % max(1, int(round(row_fps))) == 0:
                print(f"{pair_dir.name}: processed {processed} frames", flush=True)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if arkit_archive is not None:
            arkit_archive.close()

    if processed == 0:
        raise RuntimeError(f"No frames processed for {pair_dir}")

    transcode_browser_mp4(output_path)
    print(f"{pair_dir.name}: wrote {output_path} ({processed} frames)", flush=True)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run EgoForce on Visora MP4 + ARKit ZIP pairs.")
    parser.add_argument("--dataset-root", default=str(ROOT_DIR / "visora-arkit-v1"))
    parser.add_argument("--pair-id", action="append", help="Pair directory name to run. Can be repeated.")
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "_DATA" / "visora_runs"))
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=5.0, help="Seconds to process per pair. Use -1 for full video.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--viz-mode",
        choices=("arkit-overlay", "egoforce-overlay", "comparison-overlay", "comparison-split", "render"),
        default="egoforce-overlay",
        help="Visualization to write. render is the original EgoForce synthetic scene.",
    )
    parser.add_argument("--include-arm-mesh", action="store_true")
    parser.add_argument(
        "--egoforce-crop-source",
        choices=("arkit", "yolo", "arkit-yolo-hybrid", "original"),
        default="original",
        help="Source used to crop hands before EgoForce inference.",
    )
    parser.add_argument(
        "--egoforce-arm-source",
        choices=("none", "drop", "synthetic-arkit"),
        default="none",
        help="Forearm crop handling: keep original detector arms, drop arms, or synthesize arms from ARKit hand crops.",
    )
    parser.add_argument(
        "--arkit-side-source",
        choices=("screen", "chirality"),
        default="screen",
        help="How to assign ARKit hand crops to EgoForce left/right slots.",
    )
    parser.add_argument(
        "--egoforce-overlay-space",
        choices=("auto", "camera", "direct-2d"),
        default="auto",
        help="Use camera-space MANO reprojection, direct 2D head output, or auto fallback for unstable projections.",
    )
    parser.add_argument(
        "--egoforce-depth-anchor",
        choices=("none", "lidar"),
        default="none",
        help="Optionally anchor EgoForce camera-space translation with ARKit LiDAR depth.",
    )
    parser.add_argument(
        "--egoforce-input-cleanup",
        choices=("none", "inpaint-markers"),
        default="none",
        help="Preprocess the RGB passed to EgoForce crops while keeping detection/visualization on the original frame.",
    )
    parser.add_argument("--enable-trt", action="store_true", help="Allow TensorRT compile if torch_tensorrt is installed.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.enable_trt:
        os.environ.pop("EGOFORCE_DISABLE_TRT", None)

    dataset_root = Path(args.dataset_root)
    if args.pair_id:
        pair_dirs = [dataset_root / pair_id for pair_id in args.pair_id]
    else:
        pair_dirs = list(iter_pair_dirs(dataset_root))

    if not pair_dirs:
        raise SystemExit(f"No Visora pairs found under {dataset_root}")

    seconds = None if args.seconds is not None and args.seconds < 0 else args.seconds
    inference = None
    if args.viz_mode in {"egoforce-overlay", "comparison-overlay", "comparison-split", "render"}:
        _, Inference, _, _ = ensure_egoforce_runtime()
        inference = Inference()

    outputs = []
    for pair_dir in pair_dirs:
        outputs.append(
            process_pair(
                pair_dir,
                inference,
                args.output_dir,
                args.start_seconds,
                seconds,
                args.max_frames,
                args.include_arm_mesh,
                args.viz_mode,
                args.egoforce_crop_source,
                args.egoforce_arm_source,
                args.arkit_side_source,
                args.egoforce_overlay_space,
                args.egoforce_depth_anchor,
                args.egoforce_input_cleanup,
            )
        )

    print("Outputs:")
    for output_path in outputs:
        print(output_path)


if __name__ == "__main__":
    main()
