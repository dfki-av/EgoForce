import os
import sys  

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import torch
try:
    import torch_tensorrt
except ImportError:
    torch_tensorrt = None

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
if torch_tensorrt is not None:
    torch_tensorrt.runtime.set_multi_device_safe_mode(True)
    torch_tensorrt.runtime.set_cudagraphs_mode(True)

import collections
import numpy as np  
import time
import cv2

try:
    from mmdet.apis import DetInferencer
except ImportError:
    DetInferencer = None
from ultralytics import YOLO
from camera_models import OVR624CameraModel
from models import HALO, LimbModel
from core import KalmanFilterCV3D, compute_camera_space_mesh, get_limb
from settings import config as cfg
from demo_hand_arm_loader import DemoHandArmLoader
from demo_utils import *
try:
    from renderer import Renderer
except Exception as exc:
    Renderer = None
    RENDERER_IMPORT_ERROR = exc
else:
    RENDERER_IMPORT_ERROR = None


UNDISTORT_INP = True


def _bbox_center_xy(bbox):
    bbox = np.asarray(bbox, dtype=np.float32)
    return np.asarray([0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])], dtype=np.float32)


def _bbox_is_valid(bbox):
    bbox = np.asarray(bbox, dtype=np.float32)
    return bool(np.isfinite(bbox).all() and bbox[2] > bbox[0] and bbox[3] > bbox[1])


def _smooth_record_bbox(record, prev_record, alpha):
    record = dict(record)
    current_bbox = np.asarray(record["bbox"], dtype=np.float32)
    previous_bbox = np.asarray(prev_record["bbox"], dtype=np.float32)
    record["detected_bbox"] = current_bbox.copy()
    record["bbox"] = ((1.0 - float(alpha)) * previous_bbox + float(alpha) * current_bbox).astype(np.float32)
    record["x_center"] = float(0.5 * (record["bbox"][0] + record["bbox"][2]))
    return record


def _assign_yolo_hand_candidates(hand_candidates, prev_boxes, image_width, use_temporal_assignment=True):
    hand_candidates = [rec for rec in hand_candidates if _bbox_is_valid(rec["bbox"])]
    if not hand_candidates:
        return {}

    selected = {}
    prev_left = prev_boxes.get("left", {}).get("hand")
    prev_right = prev_boxes.get("right", {}).get("hand")

    if len(hand_candidates) >= 2:
        if prev_left is not None and prev_right is not None:
            if use_temporal_assignment:
                left_prev_center = _bbox_center_xy(prev_left["bbox"])
                right_prev_center = _bbox_center_xy(prev_right["bbox"])
                best_pair = None
                best_cost = float("inf")
                for ldx, left_rec in enumerate(hand_candidates):
                    for rdx, right_rec in enumerate(hand_candidates):
                        if ldx == rdx:
                            continue
                        cost = float(
                            np.linalg.norm(_bbox_center_xy(left_rec["bbox"]) - left_prev_center)
                            + np.linalg.norm(_bbox_center_xy(right_rec["bbox"]) - right_prev_center)
                        )
                        if cost < best_cost:
                            best_cost = cost
                            best_pair = (left_rec, right_rec)
                if best_pair is not None and best_cost < 0.75 * float(image_width):
                    selected["left"], selected["right"] = best_pair

        if not selected:
            ordered = sorted(hand_candidates, key=lambda rec: float(rec["x_center"]))
            left_rec, right_rec = ordered[0], ordered[-1]
            if abs(float(right_rec["x_center"]) - float(left_rec["x_center"])) > 0.08 * float(image_width):
                selected["left"] = left_rec
                selected["right"] = right_rec

    else:
        rec = hand_candidates[0]
        center = _bbox_center_xy(rec["bbox"])
        distances = {}
        if prev_left is not None and _bbox_is_valid(prev_left["bbox"]):
            distances["left"] = float(np.linalg.norm(center - _bbox_center_xy(prev_left["bbox"])))
        if prev_right is not None and _bbox_is_valid(prev_right["bbox"]):
            distances["right"] = float(np.linalg.norm(center - _bbox_center_xy(prev_right["bbox"])))
        if use_temporal_assignment and distances and min(distances.values()) < 0.30 * float(image_width):
            selected[min(distances, key=distances.get)] = rec
        else:
            selected["left" if float(rec["x_center"]) < 0.5 * float(image_width) else "right"] = rec

    output = {}
    for side, rec in selected.items():
        assigned = dict(rec)
        assigned["class_handedness"] = rec.get("handedness")
        assigned["handedness"] = side
        assigned["source"] = "yolo-pose-temporal-screen"
        output[side] = assigned
    return output


def infer(self, config, model, limb_model, left_data, right_data, device):
    data = collections.defaultdict(list)
    meta = collections.defaultdict(list)
    pred_hand_type = []
    idx = 0
    for h_data, h_meta in [[left_data[0], left_data[1]], [right_data[0], right_data[1]]]:
        data['hand_crop'].append(h_data['hand_crop'])
        data['hand_sparse_kpe'].append(h_data['hand_sparse_kpe'])

        data['arm_crop'].append(h_data['arm_crop'])
        data['arm_sparse_kpe'].append(h_data['arm_sparse_kpe'])

        data['visible_arm'].append(h_data['visible_arm'])
        data['visible_hand'].append(h_data['visible_hand'])

        for k, v in h_meta.items():
            meta[k].append(v)

        pred_hand_type.append(idx)
        idx += 1

    for k, v in meta.items():
        meta[k] = torch.stack(v, dim=0)

    pred_hand_type = torch.tensor(pred_hand_type).unsqueeze(1)

    hand_crop = torch.stack(data['hand_crop'], dim=0).unsqueeze(1)
    hand_sparse_kpe = torch.stack(data['hand_sparse_kpe'], dim=0).unsqueeze(1)
    arm_crop = torch.stack(data['arm_crop'], dim=0).unsqueeze(1)
    arm_sparse_kpe = torch.stack(data['arm_sparse_kpe'], dim=0).unsqueeze(1)
    visible_arm = torch.stack(data['visible_arm'], dim=0).unsqueeze(1)
    visible_hand = torch.stack(data['visible_hand'], dim=0).unsqueeze(1)
    
    hand_crop = hand_crop.to(device) 
    hand_sparse_kpe = hand_sparse_kpe.to(device) 
    arm_crop = arm_crop.to(device) 
    arm_sparse_kpe = arm_sparse_kpe.to(device) 
    visible_arm = visible_arm.to(device) 
    visible_hand = visible_hand.to(device) 

    c = time.time() 
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        outputs = model(hand_crop, hand_sparse_kpe, arm_crop, arm_sparse_kpe)

    print('Model fps:', 1.0 / (time.time() - c), ' time taken: (ms):', (time.time() - c) * 1000)

    pred_betas = outputs['betas'].float()
    pred_global_orient = outputs['global_orient'].float()   
    pred_hand_pose = outputs['hand_pose'].float()
    pred_kpts_2d = outputs['hand_kpts_2d'].squeeze(1).float()
    pred_arm_kpts_2d = outputs['arm_kpts_2d'].squeeze(1).float()
    pred_hand_kpt_w = outputs['hand_kpt_w'].squeeze(1).float()
    pred_arm_kpt_w = outputs['arm_kpt_w'].squeeze(1).float()
    pred_hand_kpt_w = pred_hand_kpt_w * visible_hand.squeeze(1).unsqueeze(-1)
    pred_arm_kpt_w = pred_arm_kpt_w * visible_arm.squeeze(1).unsqueeze(-1)

    pred_arm_shape = outputs['arm_shape'].float()
    pred_arm_R = outputs['arm_R'].float()

    if not self.undistort_inp: # if input is not undistorted, undistort the output 2D keypoints
        pred_kpts_2d = undistort_keypoints(pred_kpts_2d, meta['K_hand'].to(device), self.camera_model.distortion_model)
        pred_arm_kpts_2d = undistort_keypoints(pred_arm_kpts_2d, meta['K_arm'].to(device), self.camera_model.distortion_model)

    zT = torch.zeros(pred_global_orient.shape[0], pred_global_orient.shape[1], 3).to(pred_global_orient.device)
    limb_output = get_limb(config, limb_model, pred_global_orient, pred_betas, pred_hand_pose, zT, pred_hand_type, pred_arm_shape, pred_arm_R)
    pred_vertices = limb_output.hand.vertices
    pred_j3d = limb_output.hand.joints
    pred_arm_vertices = limb_output.arm.vertices
    pred_arm_j3d = limb_output.arm.joints

    c = time.time() 

    limb_output.hand.crop_j2d = pred_kpts_2d
    limb_output.arm.crop_j2d = pred_arm_kpts_2d
    limb_output.hand.confidence = pred_hand_kpt_w
    limb_output.arm.confidence = pred_arm_kpt_w
    cs_limb_output = compute_camera_space_mesh(config, meta, limb_output)
    pred_vertices = cs_limb_output.hand.vertices
    pred_j3d = cs_limb_output.hand.joints
    pred_arm_vertices = cs_limb_output.arm.vertices
    pred_arm_j3d = cs_limb_output.arm.joints
    pred_transl = cs_limb_output.transl

    if self.enable_kalman_filter and self.left_kalman_filter is not None and self.right_kalman_filter is not None:
        filters = [self.left_kalman_filter, self.right_kalman_filter]
        for hdx, kalman_filter in enumerate(filters):
            measured_transl = pred_transl[hdx, 0]
            filtered_transl = kalman_filter.step(measured_transl, visible_hand[hdx]).view(1, 3)
            delta = filtered_transl - measured_transl.view(1, 3)

            pred_vertices[hdx] = pred_vertices[hdx] + delta
            pred_j3d[hdx] = pred_j3d[hdx] + delta
            pred_arm_vertices[hdx] = pred_arm_vertices[hdx] + delta
            pred_arm_j3d[hdx] = pred_arm_j3d[hdx] + delta
            pred_transl[hdx, 0] = filtered_transl.squeeze(0)

    print('RSS fps:', 1.0 / (time.time() - c), ' time taken: (ms):', (time.time() - c) * 1000)

    hand_crop = hand_crop.squeeze(1).permute(0, 2, 3, 1)
    arm_crop = arm_crop.squeeze(1).permute(0, 2, 3, 1)
    hand_crop = hand_crop.cpu().numpy() * 255
    arm_crop = arm_crop.cpu().numpy() * 255
    hand_crop = hand_crop.astype(np.uint8)
    arm_crop = arm_crop.astype(np.uint8)

    pred_hand_j2d_direct = get_j2d_from_kpt2d(config, meta, pred_kpts_2d, pred_type='hand').cpu().numpy()
    pred_transl = pred_transl.cpu().numpy()
    pred_j2d = self.camera_model.camera_to_uv(pred_j3d.cpu().numpy())
    pred_arm_j2d_proj = self.camera_model.camera_to_uv(pred_arm_j3d.cpu().numpy())

    pred_arm_j2d = get_j2d_from_kpt2d(config, meta, pred_arm_kpts_2d, pred_type='arm').cpu().numpy()
    pred_arm_j2d[:, [0, -1]] = pred_arm_j2d_proj[:, [0, -1]]

    visible_arm = visible_arm.squeeze().bool()
    visible_arm = visible_arm.cpu().numpy()

    pred_arm_j2d[~visible_arm] = pred_arm_j2d_proj[~visible_arm]

    pred_j3d = pred_j3d.cpu().numpy()         
    pred_arm_j3d = pred_arm_j3d.cpu().numpy()         

    pred_vertices = pred_vertices.cpu().numpy() 
    pred_arm_vertices = pred_arm_vertices.cpu().numpy() 

    visible_hand = visible_hand.cpu().numpy()

    pred_hand_kpt_w = pred_hand_kpt_w.cpu().numpy()
    pred_arm_kpt_w = pred_arm_kpt_w.cpu().numpy()

    pred_hand_type = pred_hand_type.squeeze(0).cpu().numpy()
    
    return {
        'hand_crop': hand_crop,
        'arm_crop': arm_crop,
        'pred_j3d': pred_j3d,
        'pred_j2d': pred_j2d,
        'pred_hand_j2d_direct': pred_hand_j2d_direct,
        'pred_vertices': pred_vertices,
        'pred_arm_j3d': pred_arm_j3d,
        'pred_arm_j2d': pred_arm_j2d,
        'pred_arm_vertices': pred_arm_vertices,
        'pred_transl': pred_transl,
        'pred_hand_kpt_w': pred_hand_kpt_w,
        'pred_arm_kpt_w': pred_arm_kpt_w,
    }

 
class Inference:
    def __init__(self, camera_model=None, undistort_inp=UNDISTORT_INP):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.undistort_inp = bool(undistort_inp)
        self.enable_kalman_filter = True
        self.kalman_filter_kwargs = {
            'q_pos': 0.001,
            'q_vel': 1e-05,
            'r_meas': 0.001,
        }
        self.kalman_filter_freq = 30.0

        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        self.box_inferencer = None
        if DetInferencer is not None and os.path.exists(cfg.DETECTION.HAND_ARM_PATH):
            try:
                self.box_inferencer = DetInferencer(f'{ROOT_DIR}/demo/rtmdet_tiny_8xb32-300e_combined_cutmix.py', weights=cfg.DETECTION.HAND_ARM_PATH, device=self.device)

                det_model = self.box_inferencer.model.eval().half()
                det_model = optimize_mmdet_model_for_inference(det_model)

                det_model = torch.compile(
                    det_model,
                    backend="inductor",
                    mode="max-autotune",
                    dynamic=False,
                    fullgraph=False
                ).half()
                self.box_inferencer.model = det_model
            except Exception as exc:
                print(f"MMDetection forearm detector unavailable, continuing with hand-only detections: {exc}")
                self.box_inferencer = None
        else:
            print("MMDetection forearm detector unavailable, continuing with hand-only detections.")
        self.hand_detector = YOLO(cfg.DETECTION.HAND_PATH, task="pose")

        self.classes = ['left_forearm', 'right_forearm', 'left_hand', 'right_hand']

        init_tracking_defaults(self)

        model = HALO(cfg)
        model.load_state_dict(torch.load(cfg.POSE_3D.CHECKPOINT_PATH, map_location=self.device), strict=True)
        model.eval()

        self.model = compile_to_tensorrt(model, self.device)


        self.limb_model = LimbModel(cfg, device=self.device, use_pose_pca=False, n_components=5)

        self.camera_model = None
        self.left_dataset = None
        self.right_dataset = None
        self.left_kalman_filter = None
        self.right_kalman_filter = None
        if camera_model is not None:
            self.set_camera_model(camera_model, undistort_inp=self.undistort_inp)
        self.set_kalman_filter_frequency(self.kalman_filter_freq)

        self.renderer = None
        self.frame_index = 0

        self.stream_det = torch.cuda.Stream() if torch.cuda.is_available() and self.box_inferencer is not None else None
        self.stream_yolo = torch.cuda.Stream() if torch.cuda.is_available() else None

    def set_camera_model(self, camera_model, undistort_inp=None):
        if camera_model is None:
            raise ValueError("camera_model must be provided before inference can run.")

        if undistort_inp is not None:
            self.undistort_inp = bool(undistort_inp)

        self.camera_model = camera_model.clone() if hasattr(camera_model, 'clone') else camera_model
        left_camera = self.camera_model.clone() if hasattr(self.camera_model, 'clone') else self.camera_model
        right_camera = self.camera_model.clone() if hasattr(self.camera_model, 'clone') else self.camera_model

        self.left_dataset = DemoHandArmLoader(
            cfg,
            left_camera,
            undistort_inp=self.undistort_inp,
            return_complete_image=False,
            hand_type='left',
        )
        self.right_dataset = DemoHandArmLoader(
            cfg,
            right_camera,
            undistort_inp=self.undistort_inp,
            return_complete_image=False,
            hand_type='right',
        )
        self.renderer = None
        return self.camera_model

    def set_kalman_filter_frequency(self, freq):
        self.kalman_filter_freq = float(max(freq, 1e-6))
        if not self.enable_kalman_filter:
            self.left_kalman_filter = None
            self.right_kalman_filter = None
            return

        self.left_kalman_filter = KalmanFilterCV3D(
            freq=self.kalman_filter_freq,
            **self.kalman_filter_kwargs,
        ).to(self.device)
        self.right_kalman_filter = KalmanFilterCV3D(
            freq=self.kalman_filter_freq,
            **self.kalman_filter_kwargs,
        ).to(self.device)

    def reset_runtime_state(self):
        init_tracking_defaults(self)
        self.renderer = None
        self.frame_index = 0

        if self.left_kalman_filter is not None:
            self.left_kalman_filter.reset_state()
        if self.right_kalman_filter is not None:
            self.right_kalman_filter.reset_state()


    def detect_bounding_boxes(self, rgb_image):
        t0 = time.time()

        det_out = None
        detector_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        yolo_kwargs = dict(
            verbose=False,
            half=self.device.type == "cuda",
            device=self.device,
            conf=0.25,
            iou=0.50,
        )

        if self.box_inferencer is not None and self.stream_det is not None and self.stream_yolo is not None:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                with torch.cuda.stream(self.stream_det):
                    det_out = self.box_inferencer(detector_image)
                with torch.cuda.stream(self.stream_yolo):
                    if getattr(self, "yolo_use_track", False):
                        yolo_res = self.hand_detector.track(
                            detector_image,
                            persist=getattr(self, "yolo_track_persist", False),
                            **yolo_kwargs,
                        )[0]
                    else:
                        yolo_res = self.hand_detector.predict(detector_image, **yolo_kwargs)[0]

            torch.cuda.current_stream().wait_stream(self.stream_det)
            torch.cuda.current_stream().wait_stream(self.stream_yolo)
            torch.cuda.synchronize()
        else:
            with torch.no_grad():
                if getattr(self, "yolo_use_track", False):
                    yolo_res = self.hand_detector.track(
                        detector_image,
                        persist=getattr(self, "yolo_track_persist", False),
                        **yolo_kwargs,
                    )[0]
                else:
                    yolo_res = self.hand_detector.predict(detector_image, **yolo_kwargs)[0]

        temp_left_hand, temp_left_arm = [], []
        temp_right_hand, temp_right_arm = [], []

        if det_out is not None:
            # === parse detector A (mmdeploy) ===
            det_preds = det_out['predictions'][0]
            labels_np   = np.asarray(det_preds['labels'],   dtype=np.int64)
            scores_np   = np.asarray(det_preds['scores'],   dtype=np.float32)
            bboxes_np   = np.asarray(det_preds['bboxes'],   dtype=np.float32)
            keypoints_np= np.asarray(det_preds['keypoints'])  # dtype as given

            classes = self.classes
            for idx, (lbl_idx, score) in enumerate(zip(labels_np, scores_np)):
                if score < 0.3:
                    continue
                label_str = classes[int(lbl_idx)]
                bbox      = bboxes_np[idx]
                kpt       = keypoints_np[idx]

                if 'hand' in label_str:
                    rec = {
                        'bbox': bbox,
                        'keypoint': kpt,
                        'score': float(score),
                        'x_center': float(0.5 * (bbox[0] + bbox[2])),
                        'handedness': 'left' if 'left' in label_str else ('right' if 'right' in label_str else None),
                    }
                    if rec['handedness'] == 'left':
                        temp_left_hand.append(rec)
                    elif rec['handedness'] == 'right':
                        temp_right_hand.append(rec)

                elif 'forearm' in label_str:
                    rec = {
                        'bbox': bbox,
                        'keypoint': kpt,
                        'score': float(score),
                        'x_center': float(0.5 * (bbox[0] + bbox[2])),
                        'handedness': 'left' if 'left' in label_str else ('right' if 'right' in label_str else None),
                    }
                    if rec['handedness'] == 'left':
                        temp_left_arm.append(rec)
                    elif rec['handedness'] == 'right':
                        temp_right_arm.append(rec)

        temp = {
            'left':  {'hand': temp_left_hand,  'arm': temp_left_arm},
            'right': {'hand': temp_right_hand, 'arm': temp_right_arm},
        }

        # === parse YOLO (batch all CPU copies) ===
        boxes = yolo_res.boxes
        kpts  = yolo_res.keypoints

        if boxes is None or kpts is None or len(boxes) == 0:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            confs = np.zeros((0,), dtype=np.float32)
            clses = np.zeros((0,), dtype=np.int64)
            kpxy = np.zeros((0, 3, 2), dtype=np.float32)
            kpc = np.zeros((0, 3), dtype=np.float32)
        else:
            xyxy  = boxes.xyxy.cpu().numpy().astype(np.float32, copy=False) if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy, dtype=np.float32)
            confs = boxes.conf.cpu().numpy().astype(np.float32, copy=False)  if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf, dtype=np.float32)
            clses = boxes.cls.cpu().numpy().astype(np.int64,   copy=False)   if hasattr(boxes.cls, "cpu")  else np.asarray(boxes.cls,  dtype=np.int64)

            kpxy  = kpts.xy.cpu().numpy().astype(np.float32, copy=False)     if hasattr(kpts.xy, "cpu")    else np.asarray(kpts.xy,    dtype=np.float32)
            kpc   = kpts.conf.cpu().numpy().astype(np.float32, copy=False)    if hasattr(kpts.conf, "cpu")  else np.asarray(kpts.conf,  dtype=np.float32)

        B = xyxy.shape[0]
        H, W = rgb_image.shape[:2]

        bounding_boxes = {'left': {}, 'right': {}}
        prev_boxes = self.prev_boxes
        hand_candidates = []

        # Fast per-detection loop (small B; keep simple)
        for b in range(B):
            bbox = xyxy[b]
            conf = float(confs[b])
            cls  = int(clses[b])
            kp_xy   = kpxy[b]
            kp_conf = kpc[b]

            handedness = 'left' if cls == 0 else 'right'

            # reuse your function (assumed available)
            hand_visible, hb = compute_bbox(kp_xy, W, H, type='hand')
            if not hand_visible:
                continue
            hand_candidates.append({
                'bbox': hb,
                'raw_bbox': bbox,
                'keypoint': kp_xy,
                'conf': kp_conf,
                'score': conf,
                'x_center': float(0.5 * (hb[0] + hb[2])),
                'handedness': handedness
            })

        for side, hand_record in _assign_yolo_hand_candidates(
            hand_candidates,
            prev_boxes,
            W,
            use_temporal_assignment=getattr(self, "yolo_temporal_assignment", False),
        ).items():
            bounding_boxes[side]['hand'] = hand_record

        iou_thr = 0.0
        hand_stable_iou = self.hand_stable_iou
        arm_stable_iou  = self.arm_stable_iou
        hand_bbox_ema_alpha = getattr(self, "hand_bbox_ema_alpha", 0.35)
        arm_bbox_ema_alpha = getattr(self, "arm_bbox_ema_alpha", 0.35)
        hand_bbox_smooth_iou = getattr(self, "hand_bbox_smooth_iou", hand_stable_iou)
        arm_bbox_smooth_iou = getattr(self, "arm_bbox_smooth_iou", arm_stable_iou)

        # === choose best arm per side via vectorized IoU ===
        for side in ('left', 'right'):
            hand = bounding_boxes[side].get('hand')
            if hand is None:
                continue

            arm_cands = temp[side]['arm']
            if not arm_cands:
                continue

            hand_box = np.asarray(hand['bbox'], dtype=np.float32)

            # stack arm boxes into (M,4); keep list for metadata
            arm_boxes = np.asarray([c['bbox'] for c in arm_cands], dtype=np.float32)
            ious = iou_xyxy_one_to_many(hand_box, arm_boxes)
            best_idx = int(ious.argmax())
            best_iou = float(ious[best_idx])
            if best_iou <= iou_thr:
                continue

            best = arm_cands[best_idx]

            # stability: prefer previous arm if similar (IoU >= arm_stable_iou)
            prev_arm = prev_boxes[side].get('arm')
            if prev_arm is not None:
                if compute_bbox_iou(best['bbox'], prev_arm['bbox']) >= arm_bbox_smooth_iou:
                    best = _smooth_record_bbox(best, prev_arm, arm_bbox_ema_alpha)

            bounding_boxes[side]['arm'] = best

        # === reuse previous boxes when stable (same logic) ===
        for side in ('left', 'right'):
            curr_hand = bounding_boxes[side].get('hand')
            if curr_hand is None:
                continue

            prev_hand = prev_boxes[side].get('hand')
            prev_arm  = prev_boxes[side].get('arm')

            if prev_hand is None:
                continue

            cur_prev_hand_iou = compute_bbox_iou(curr_hand['bbox'], prev_hand['bbox'])

            if cur_prev_hand_iou >= hand_bbox_smooth_iou:
                bounding_boxes[side]['hand'] = _smooth_record_bbox(curr_hand, prev_hand, hand_bbox_ema_alpha)

            if ('arm' not in bounding_boxes[side]) and (prev_arm is not None):
                if cur_prev_hand_iou >= hand_stable_iou:
                    pb = np.asarray(prev_arm['bbox'], dtype=np.float32).copy()
                    bounding_boxes[side]['arm'] = {
                        'bbox': pb,
                        'keypoint': prev_arm.get('keypoint', None),
                        'score': float(prev_arm.get('score', 0.0)),
                        'x_center': float(0.5 * (pb[0] + pb[2])),
                        'handedness': side
                    }

        # === update history ===
        for side in ('left', 'right'):
            prev_boxes[side]['hand'] = bounding_boxes[side].get('hand')
            prev_boxes[side]['arm']  = bounding_boxes[side].get('arm')

        return bounding_boxes

    def run(self, rgb_image, device, include_arm_mesh=False):
        if self.camera_model is None or self.left_dataset is None or self.right_dataset is None:
            raise RuntimeError("Inference camera model is not initialized. Call set_camera_model(...) first.")

        start_time = time.time()
        c = time.time()    
        bounding_boxes = self.detect_bounding_boxes(rgb_image)
        print('Detection fps: ', 1 / (time.time() - c), ' time taken: (ms) ', (time.time() - c)*1000)

        c = time.time()
        left_data = self.left_dataset.transform(rgb_image, bounding_boxes['left'])
        right_data = self.right_dataset.transform(rgb_image, bounding_boxes['right'])
        print('Crop fps: ', 1 / (time.time() - c), ' time taken: (ms) ', (time.time() - c)*1000)

        c = time.time() 
        with torch.no_grad():
            outs = infer(self, cfg, self.model, self.limb_model, left_data, right_data, device)
        print('Inference fps: ', 1 / (time.time() - c), ' time taken: (ms) ', (time.time() - c)*1000)
        print('Total fps: ', 1 / (time.time() - start_time), ' time taken: (ms) ', (time.time() - start_time)*1000)

        c = time.time()
        if Renderer is None:
            if self.frame_index == 0:
                print(f"Renderer unavailable, returning blank render views: {RENDERER_IMPORT_ERROR}")
            self.frame_index += 1
            blank = np.full_like(rgb_image, 30)
            return blank, blank

        if self.renderer is None:
            meta = left_data[1]
            try:
                self.renderer = Renderer(meta)
            except Exception as exc:
                print(f"Renderer initialization failed, returning blank render views: {exc}")
                self.frame_index += 1
                blank = np.full_like(rgb_image, 30)
                return blank, blank
        
        render_image, tp_image = self.renderer.render(
            outs,
            self.limb_model,
            rgb_image,
            include_arm_mesh=include_arm_mesh,
        )
        print('Visualization fps: ', 1 / (time.time() - c), ' time taken: (ms) ', (time.time() - c)*1000)

        return render_image, tp_image
