import torch
import torch.nn as nn
import cv2
import numpy as np
import os

from pathlib import Path
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont
try:
    from mmcv.cnn.utils.fuse_conv_bn import fuse_conv_bn
except ImportError:
    from torch.nn.utils.fusion import fuse_conv_bn_eval as fuse_conv_bn


def optimize_mmdet_model_for_inference(det_model: nn.Module, warmup_shape=(1, 3, 640, 640)):
    """
    Prepare an mmdet RTMDet model for fast single-GPU FP16 inference.

    Steps:
      1. Convert SyncBN → BN  (required before fusion on single GPU)
      2. Re-parameterize Rep-style blocks (CSPNeXt deploy mode)
      3. Fuse Conv+BN everywhere
      4. Remove Dropout / no-op layers
      5. Freeze all parameters (no grad overhead)
      6. Switch to channels-last memory format (faster on Ampere+)
      7. Run a warm-up forward pass so CUDA kernels are cached
    """
    def syncbn_to_bn(module):
        for name, child in module.named_children():
            if isinstance(child, nn.SyncBatchNorm):
                bn = nn.BatchNorm2d(
                    child.num_features,
                    eps=child.eps,
                    momentum=child.momentum,
                    affine=child.affine,
                    track_running_stats=child.track_running_stats,
                )
                if child.affine:
                    bn.weight = child.weight
                    bn.bias = child.bias
                bn.running_mean = child.running_mean
                bn.running_var = child.running_var
                bn.num_batches_tracked = child.num_batches_tracked
                setattr(module, name, bn)
            else:
                syncbn_to_bn(child)

    def try_switch_to_deploy(module):
        for m in module.modules():
            if hasattr(m, 'switch_to_deploy') and callable(m.switch_to_deploy):
                try:
                    m.switch_to_deploy()
                except Exception:
                    pass

    def fuse_conv_bn_in_place(module):
        for name, child in module.named_children():
            conv_attr = getattr(child, 'conv', None)
            bn_attr = getattr(child, 'bn', None)
            if isinstance(conv_attr, nn.Conv2d) and isinstance(bn_attr, nn.BatchNorm2d):
                bn_attr.eval()
                try:
                    child.conv = fuse_conv_bn(conv_attr, bn_attr)
                    child.bn = nn.Identity()
                except Exception:
                    pass

            if isinstance(child, nn.Sequential):
                i = 0
                while i < len(child) - 1:
                    if isinstance(child[i], nn.Conv2d) and isinstance(child[i + 1], nn.BatchNorm2d):
                        child[i + 1].eval()
                        try:
                            child[i] = fuse_conv_bn(child[i], child[i + 1])
                            child[i + 1] = nn.Identity()
                        except Exception:
                            pass
                        i += 2
                    else:
                        i += 1

            fuse_conv_bn_in_place(child)

    def remove_noops(module):
        for name, child in module.named_children():
            if isinstance(child, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                setattr(module, name, nn.Identity())
            else:
                remove_noops(child)

    det_model.eval()
    syncbn_to_bn(det_model)
    try_switch_to_deploy(det_model)
    fuse_conv_bn_in_place(det_model)
    remove_noops(det_model)

    for p in det_model.parameters():
        p.requires_grad_(False)

    det_model.to(memory_format=torch.channels_last)

    if warmup_shape is not None:
        device = next(det_model.parameters()).device
        dtype = next(det_model.parameters()).dtype
        dummy = torch.zeros(warmup_shape, device=device, dtype=dtype)
        dummy = dummy.to(memory_format=torch.channels_last)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(dtype == torch.float16), dtype=torch.float16):
            try:
                det_model(dummy)
            except Exception:
                pass
        del dummy
        torch.cuda.empty_cache()

    return det_model




def init_tracking_defaults(self):
    self.yolo_track_cfg = dict(persist=True, conf=0.25, iou=0.50, tracker='bytetrack.yaml')
    # EgoForce is frame-level; stale detector history can freeze bad crops in long clips.
    self.yolo_use_track = False
    self.yolo_track_persist = False
    self.yolo_temporal_assignment = False
    self.arm_attach_iou = 0.20     # when selecting arm from current detections (pre-fallback)
    self.hand_stable_iou = 1.01
    self.arm_stable_iou = 1.01
    self.hand_bbox_ema_alpha = 0.0
    self.arm_bbox_ema_alpha = 0.0
    self.hand_bbox_smooth_iou = self.hand_stable_iou
    self.arm_bbox_smooth_iou = self.arm_stable_iou
    # state for fallback
    self.prev_boxes = {'left': {'hand': None, 'arm': None},
                    'right': {'hand': None, 'arm': None}}



def compile_to_tensorrt(model, device):
    x1, x2, x3, x4 = torch.rand([2, 1, 3, 224, 224]), torch.rand([2, 1, 3, 6, 2]), torch.rand([2, 1, 3, 224, 224]), torch.rand([2, 1, 3, 6, 2])
    x1, x2, x3, x4 = x1.to(device), x2.to(device), x3.to(device), x4.to(device)

    with torch.inference_mode():
        model = model.to(device).half()
        x1, x2, x3, x4 = x1.half(), x2.half(), x3.half(), x4.half()
        traced_model = torch.jit.trace(model, (x1, x2, x3, x4), strict=False)

    if os.environ.get("EGOFORCE_DISABLE_TRT", "").lower() in {"1", "true", "yes"}:
        return traced_model

    backend_kwargs = {
        "enabled_precisions": {torch.half},
        "min_block_size": 2,
        "torch_executed_ops": {"torch.ops.aten.sub.Tensor"},
        "optimization_level": 5,
        "use_python_runtime": False,
    }

    try:
        model = torch.compile(traced_model, backend="torch_tensorrt", options=backend_kwargs, dynamic=False,)
        with torch.no_grad():
            model(x1, x2, x3, x4) # compiled on first run
    except Exception as exc:
        print(f"TensorRT compile unavailable, using TorchScript model: {exc}")
        model = traced_model

    return model


def create_square_bbox(keypoints, image_width, image_height, padding_factor):
    keypoints = keypoints[:, :2]

    if keypoints.size == 0:
        # Return full image if no keypoints are present
        return (0, 0, image_width, image_height)

    # Center keypoints
    mean = np.mean(keypoints, axis=0)
    centered = keypoints - mean

    # PCA
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, order]

    # Rotation matrix to align with principal components
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    rotation_matrix = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle),  np.cos(-angle)]
    ])
    rotated = centered @ rotation_matrix.T

    # Bounding rectangle in rotated space
    x_min, y_min = np.min(rotated, axis=0)
    x_max, y_max = np.max(rotated, axis=0)
    width = x_max - x_min
    height = y_max - y_min

    # Side length of the square
    side_length = max(width, height)

    padding = side_length * padding_factor
    side_length += 2 * padding

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    # Define square in rotated space
    x_min_padded = x_center - side_length / 2
    y_min_padded = y_center - side_length / 2

    # Corners of the square in rotated space
    corners_rotated = np.array([
        [x_min_padded, y_min_padded],
        [x_min_padded + side_length, y_min_padded],
        [x_min_padded + side_length, y_min_padded + side_length],
        [x_min_padded, y_min_padded + side_length]
    ])

    # Rotate corners back to original orientation
    rotation_matrix_inv = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    corners = corners_rotated @ rotation_matrix_inv.T + mean

    # Get axis-aligned bounding box from rotated square
    x_coords = corners[:, 0]
    y_coords = corners[:, 1]

    x1 = np.min(x_coords)
    y1 = np.min(y_coords)
    x2 = np.max(x_coords)
    y2 = np.max(y_coords)

    # Ensure the bounding box is within image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_width, x2)
    y2 = min(image_height, y2)

    # Convert to integers
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    return (x1, y1, x2, y2)


def compute_bbox(j2d, img_w, img_h, type='hand'):
    valid_joints = []
    for x, y in j2d.astype(int):
        invalid = False
        if x < 0 or x >= img_w or y < 0 or y >= img_h:
            invalid = True  # Out of bounds

        valid_joints.append(not invalid)

    if type == 'hand':
        valid_joints = sum(valid_joints) / len(valid_joints)

        if valid_joints < 0.5:
            return False, np.array([-1, -1, -1, -1])
        
        x1, y1, x2, y2 = create_square_bbox(j2d, img_w, img_h, padding_factor=0.05)
    else:
        valid_joints = sum(valid_joints)
        if not valid_joints:
            return False, np.array([-1, -1, -1, -1])

        x1, y1, x2, y2 = create_square_bbox(j2d, img_w, img_h, padding_factor=0)

    return True, np.array([x1, y1, x2, y2])


def iou_xyxy_one_to_many(a_box: np.ndarray, b_boxes: np.ndarray) -> np.ndarray:
    """
    a_box:   (4,)  [x1,y1,x2,y2]
    b_boxes: (M,4)
    returns: (M,) IoUs
    """
    ax1, ay1, ax2, ay2 = a_box
    bx1, by1, bx2, by2 = b_boxes.T  # views

    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)

    iw = np.clip(ix2 - ix1, 0.0, None)
    ih = np.clip(iy2 - iy1, 0.0, None)
    inter = iw * ih

    area_a = np.clip(ax2 - ax1, 0.0, None) * np.clip(ay2 - ay1, 0.0, None)
    area_b = np.clip(bx2 - bx1, 0.0, None) * np.clip(by2 - by1, 0.0, None)

    denom = area_a + area_b - inter + 1e-9
    return inter / denom


def compute_bbox_iou(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(iou_xyxy_one_to_many(a, b[None, :])[0])



def get_j2d_from_kpt2d(cfg, meta, kpt2d, pred_type='hand'):
    device = kpt2d.device

    K_T = meta[f'K_{pred_type}'].to(device)
    crop_size = meta[f'{pred_type}_crop_size'].to(device)
    hand_bbox = meta[f'{pred_type}_bbox'].to(device)

    crop_w, crop_h = crop_size[:, 0], crop_size[:, 1]
    inp_w, inp_h = cfg.POSE_3D.IMAGE_SIZE 
    
    j2d_out = torch.zeros_like(kpt2d, device=device)

    j2d_out[..., 0] = kpt2d[..., 0] / inp_w * crop_w.unsqueeze(1) 
    j2d_out[..., 1] = kpt2d[..., 1] / inp_h * crop_h.unsqueeze(1)

    j2d_out = j2d_out + hand_bbox[:, :2].unsqueeze(1)  
    
    return j2d_out


def undistort_keypoints(keypoints, K_new, distortion_model):
    """
    Remap keypoints from the distorted sub-image coordinate system to the undistorted view.
    
    keypoints: Nx2 array of keypoint coordinates computed using the linear K_new projection.
    K_new: The new intrinsics used for the output image.
    K_sub: The adjusted sub-image intrinsic matrix (K_orig adjusted for the ROI).
    distortion_model: An instance of OVR624Distortion initialized for the sub-image dimensions.
    
    Returns an array of keypoints aligned with the undistorted image.
    """
    # Convert keypoints to normalized coordinates (as used by K_new)
    _lib = torch if isinstance(keypoints, torch.Tensor) else np

    K_new = K_new.unsqueeze(1) if len(K_new.shape) == 3 else K_new

    keypoints_norm = _lib.empty_like(keypoints)
    keypoints_norm[..., 0] = (keypoints[..., 0] - K_new[..., 0, 2]) / K_new[..., 0, 0]
    keypoints_norm[..., 1] = (keypoints[..., 1] - K_new[..., 1, 2]) / K_new[..., 1, 1]
    
    # Apply the inverse distortion mapping to get the "ideal" undistorted normalized coordinates.
    # (Note: Your distortion model is non-linear so this step is key for alignment)
    keypoints_undist_norm = distortion_model.inverse_evaluate(keypoints_norm)
    
    # Reproject the undistorted normalized coordinates with K_new, (or any new desired intrinsics).
    keypoints_undist = _lib.empty_like(keypoints)
    keypoints_undist[..., 0] = keypoints_undist_norm[..., 0] * K_new[..., 0, 0] + K_new[..., 0, 2]
    keypoints_undist[..., 1] = keypoints_undist_norm[..., 1] * K_new[..., 1, 1] + K_new[..., 1, 2]
    
    return keypoints_undist


def resize_to_height(image, target_height):
    if image.shape[0] == target_height:
        return image

    scale = float(target_height) / float(max(image.shape[0], 1))
    target_width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


TITLE_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
]
TITLE_TEXT_FILL = (245, 241, 232, 255)
TITLE_SHADOW_FILL = (0, 0, 0, 180)
TITLE_BG_FILL = (12, 16, 24, 170)
TITLE_BORDER_FILL = (255, 255, 255, 42)


@lru_cache(maxsize=16)
def load_title_fonts(title_size):
    title_size = max(18, int(title_size))
    superscript_size = max(12, int(round(title_size * 0.58)))

    for font_path in TITLE_FONT_CANDIDATES:
        if Path(font_path).exists():
            return (
                ImageFont.truetype(font_path, size=title_size),
                ImageFont.truetype(font_path, size=superscript_size),
            )

    return ImageFont.load_default(), ImageFont.load_default()


def build_panel_title_segments(panel_key, title_font, superscript_font):
    title_size = getattr(title_font, "size", 24)
    superscript_rise = max(4, int(round(title_size * 0.34)))

    if panel_key == "input":
        return [("Input", title_font, 0)]
    if panel_key == "ego":
        return [("Ego View", title_font, 0)]
    if panel_key == "third_person":
        return [
            ("3", title_font, 0),
            ("rd", superscript_font, superscript_rise),
            (" Person View", title_font, 0),
        ]

    raise ValueError(f"Unknown panel title key: {panel_key}")


def measure_title_segments(draw, segments, stroke_width):
    measured_segments = []
    total_width = 0
    min_top = 0
    max_bottom = 0

    for text, font, rise in segments:
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
        width = bbox[2] - bbox[0]
        top = bbox[1] - rise
        bottom = bbox[3] - rise
        min_top = min(min_top, top)
        max_bottom = max(max_bottom, bottom)
        total_width += width
        measured_segments.append((text, font, rise, bbox, width))

    return measured_segments, total_width, max_bottom - min_top, min_top


def add_panel_title(image, panel_key):
    image = np.ascontiguousarray(image.astype(np.uint8))
    panel_height, panel_width = image.shape[:2]
    pil_image = Image.fromarray(image, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    title_size = max(
        16,
        int(round(min(
            int(round(panel_height * 0.065)),
            int(round(panel_width * 0.09)),
        ) * 0.75)),
    )
    title_font, superscript_font = load_title_fonts(title_size)
    stroke_width = max(1, int(round(title_size * 0.06)))
    segments = build_panel_title_segments(panel_key, title_font, superscript_font)
    measured_segments, text_width, text_height, min_top = measure_title_segments(
        draw,
        segments,
        stroke_width,
    )

    pad_x = max(18, int(round(title_size * 0.85)))
    pad_y = max(12, int(round(title_size * 0.42)))
    capsule_width = text_width + (2 * pad_x)
    capsule_height = text_height + (2 * pad_y)
    radius = max(14, int(round(capsule_height * 0.48)))
    x0 = max(12, int(round((panel_width - capsule_width) / 2.0)))
    y0 = max(12, int(round(panel_height * 0.035)))
    x1 = min(panel_width - 12, x0 + capsule_width)
    y1 = min(panel_height - 12, y0 + capsule_height)

    if x1 <= x0 or y1 <= y0:
        return image

    border_width = max(1, int(round(title_size * 0.05)))
    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=radius,
        fill=TITLE_BG_FILL,
        outline=TITLE_BORDER_FILL,
        width=border_width,
    )

    cursor_x = x0 + pad_x
    base_y = y0 + pad_y - min_top
    shadow_offset = max(1, int(round(title_size * 0.06)))
    stroke_fill = (0, 0, 0, 120)
    for text, font, rise, bbox, width in measured_segments:
        text_x = cursor_x - bbox[0]
        text_y = base_y - rise - bbox[1]
        draw.text(
            (text_x + shadow_offset, text_y + shadow_offset),
            text,
            font=font,
            fill=TITLE_SHADOW_FILL,
        )
        draw.text(
            (text_x, text_y),
            text,
            font=font,
            fill=TITLE_TEXT_FILL,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        cursor_x += width

    return np.ascontiguousarray(Image.alpha_composite(pil_image, overlay).convert("RGB"))


def compose_output_frame(input_rgb_image, render_image, tp_image):
    if render_image.shape[:2] != tp_image.shape[:2]:
        tp_image = cv2.resize(
            tp_image,
            (render_image.shape[1], render_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    input_panel = resize_to_height(input_rgb_image, render_image.shape[0])
    input_panel = add_panel_title(input_panel, "input")
    ego_panel = add_panel_title(render_image, "ego")
    third_person_panel = add_panel_title(tp_image, "third_person")
    return np.ascontiguousarray(np.hstack([input_panel, ego_panel, third_person_panel]))


def brighten_rgb(
    image_rgb: np.ndarray,
    sigma: float = 31.0,
    max_gain: float = 1.8,
    shadow_boost: float = 0.7,
    denoise: bool = False,
) -> np.ndarray:
    """
    Edge-friendly local brightening based on a smooth illumination estimate.
    Brightens shadows more than highlights, with capped gain.

    Args:
        image_rgb: uint8 RGB image
        sigma: Gaussian sigma for illumination map
        max_gain: maximum multiplicative gain
        shadow_boost: strength of shadow lifting in [0, 1.5] roughly
        denoise: apply light denoise before enhancement

    Returns:
        uint8 RGB image
    """
    assert image_rgb.dtype == np.uint8

    img = image_rgb
    if denoise:
        img = cv2.bilateralFilter(img, d=5, sigmaColor=20, sigmaSpace=20)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)

    # Normalize luminance to [0,1]
    l01 = l / 255.0

    # Smooth illumination estimate
    illum = cv2.GaussianBlur(l01, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)

    # More boost where local illumination is dark
    # gain ~= 1 in bright areas, rises in shadow areas
    gain = 1.0 + shadow_boost * (1.0 - illum)

    # Cap gain to avoid noisy whitening
    gain = np.clip(gain, 1.0, max_gain)

    # Apply gain to luminance only
    l_new = np.clip(l01 * gain, 0.0, 1.0)

    # Mild highlight compression to avoid washed-out look
    l_new = np.power(l_new, 0.95)

    l_out = (l_new * 255.0).astype(np.uint8)
    lab_out = cv2.merge((l_out, a.astype(np.uint8), b.astype(np.uint8)))
    out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)
    return out
