import os; import sys; 
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
PROJECT_DIR = os.path.dirname(ROOT_DIR)


import torch
import os
import numpy as np
import cv2
import random

from collections import OrderedDict
from torch.utils.data import Dataset
from utils.metrics import compute_3d_errors_batch, compute_acceleration_error
from utils.augmentation import get_aug_config, color_augmentation, bbox_augmentation, undistort_local_patch, undistort_keypoints, HandColorAugmentation, CutMix
from mmdet.apis import DetInferencer


def safe_log(x, eps=1e-6):
    return np.log(np.maximum(x, eps))


def perform_det_inference(inferencer, rgb):
    classes = ['forearm', 'forearm', 'left_hand', 'right_hand']
    
    results = inferencer(rgb)
    results = results['predictions']

    arms_preds = []

    result = results[0]

    labels = result['labels']
    scores = result['scores']
    bboxes = result['bboxes']
    keypoints = result['keypoints']

    for idx, bbox in enumerate(bboxes):
        label = labels[idx]
        score = scores[idx]
        keypoint = keypoints[idx]
        cls = classes[label]

        if score < 0.3:
            continue

        if cls == 'forearm':
            arms_preds.append({
                'label': label,
                'score': score,
                'bbox': bbox,
                'keypoints': keypoint
            })

    return arms_preds


def get_arm_bboxes(inferencer, rgb):
    arm_annos = perform_det_inference(inferencer, rgb)

    mH, mW = rgb.shape[:2]
    center_x = mW / 2

    if not len(arm_annos): return None, None

    # Sort all annotations by the midpoint x
    sorted_annos = sorted(
        arm_annos,
        key=lambda a: a['keypoints'][1][0]
    )

    left_anno, right_anno = None, None
    if len(sorted_annos) >= 2:  
        left_anno  = sorted_annos[0]
        right_anno = sorted_annos[-1]

    if len(sorted_annos) == 1:
        anno = sorted_annos[0]
        mid_x = anno['keypoints'][1][0]
        if mid_x < center_x:
            left_anno = anno
        else:
            right_anno = anno

    return left_anno, right_anno


class Arm3DDatasetwitArmDetector(Dataset):
    def __init__(self, cfg, target_datset, undistort_inp=True, return_complete_image=False, hand_type=None):
        self.cfg = cfg
        self.split = target_datset.split

        assert self.split in ['train', 'val', 'test'], "Invalid split. Must be one of ['train', 'val', 'test']"

        self.is_train = self.split == 'train'
        self.target_dataset = target_datset

        self.num_joints = cfg.NUM_JOINTS_PER_HAND
        self.heatmap_size = np.array(cfg.POSE_3D.HEATMAP_SIZE)
        self.target_type = cfg.POSE_3D.TARGET_TYPE
        self.sigma = cfg.POSE_3D.SIGMA
        self.image_size = cfg.POSE_3D.IMAGE_SIZE
        self.undistort_inp = undistort_inp
        self.return_complete_image = return_complete_image  
        self.hand_type = hand_type

        self.length = len(self.target_dataset)

        self.hard_color_augmentation = HandColorAugmentation(prob=0.5,)
        self.cut_mix = CutMix(prob=0.5,)
        
        weights = f'{PROJECT_DIR}/arm_detection/mmdet_test/work_dirs/rtmdet_tiny_8xb32-300e_combined_cutmix/working_weights_with_color_aug/epoch_500.pth'
        self.inferencer = DetInferencer(weights=weights, device='cpu')

    def __len__(self):
        return self.length

    def prepare_2D_heatmaps(self, inp, j2d, vis_j2d):
        inp_h, inp_w = inp.shape[:2]
        target_width, target_height = self.image_size

        invalid_j2d = (j2d[:, 0] < 0) + (j2d[:, 1] < 0) + (j2d[:, 0] >= inp_w) + (j2d[:, 1] >= inp_h)
        valid_j2d = ~invalid_j2d[:, None]

        if self.is_train is False:
            if vis_j2d.mean() > 0:
                vis_j2d = np.ones_like(vis_j2d)
            else:
                vis_j2d = np.zeros_like(vis_j2d)

        vis_j2d = vis_j2d * valid_j2d

        sx = target_width / inp_w
        sy = target_height / inp_h

        j2d = j2d.copy()
        j2d[:, 0] *= sx
        j2d[:, 1] *= sy

        target, vis_j2d = self.generate_target(j2d, vis_j2d)

        return target, vis_j2d

    def xyz_to_spherical(self, xyz):
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        r = torch.sqrt(x**2 + y**2 + z**2)
        mask_zero_r = torch.isclose(r, torch.zeros_like(r))
        phi = torch.zeros_like(r)

        valid_indices = ~mask_zero_r
        z_valid = z[valid_indices]
        r_valid = r[valid_indices]
        phi[valid_indices] = torch.acos(torch.clamp(z_valid / r_valid, -1.0, 1.0))

        theta = torch.atan2(y, x)

        theta = theta.unsqueeze(-1)
        phi = phi.unsqueeze(-1)
        r = r.unsqueeze(-1)
        valid_indices = valid_indices.unsqueeze(-1)

        return theta, phi, r, valid_indices

    def get_hand_crop(self, image, hand_bbox, visible_hand):
        if not visible_hand: return np.zeros((224, 224, 3))
        
        x1, y1, x2, y2 = hand_bbox
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(image.shape[1], x2))
        y2 = int(min(image.shape[0], y2))

        return image[y1:y2, x1:x2]

    def crop_j2d(self, j2d, hand_bbox):
        x1, y1, _, _ = hand_bbox 
        j2d_cropped = j2d - np.array([x1, y1])
        return j2d_cropped
    
    def refine_segmentation(self, hand_mask_crop, j2d):
        hand_mask_crop = hand_mask_crop.astype(np.float32)
        h, w = hand_mask_crop.shape[:2]

        j2d = j2d.astype(int)
        valid_mask = (j2d[:, 0] >= 0) & (j2d[:, 0] < w) & (j2d[:, 1] >= 0) & (j2d[:, 1] < h)
        j2d = j2d[valid_mask]  # Filter valid indices

        # Count the number of times each channel is activated at joint positions
        y, x = j2d[:, 1], j2d[:, 0]
        label_0 = np.sum(hand_mask_crop[y, x, 1])
        label_1 = np.sum(hand_mask_crop[y, x, 2])

        if label_0 < label_1:
            hand_mask_crop[:, :, 1] = 0
        else:
            hand_mask_crop[:, :, 2] = 0

        # Convert to binary mask
        return ((hand_mask_crop[..., 1] + hand_mask_crop[..., 2]) > 0).astype(np.float32)

    def sin_encode(self, angles, L=4):
        """
        angles: shape (N,) of angles (e.g., r or phi).
        L: number of frequency components.

        Returns shape (N*L*2), i.e. each angle -> [sin(angle), cos(angle), sin(2*angle), ...]
        """
        N = angles.shape[0]  # e.g., 5 for your 5 corner/center points
        # Frequencies: 1, 2, 4, 8, ... 2^(L-1)
        freqs = 2 ** np.arange(L)  # shape (L,)

        # Expand angles so we can broadcast: (N,1) * (1,L) => (N,L)
        angles_expanded = angles[:, np.newaxis] * freqs[np.newaxis, :]
        sin_part = np.sin(angles_expanded).reshape(-1) # shape (N*L,)
        cos_part = np.cos(angles_expanded).reshape(-1) # shape (N*L,)

        # Concatenate [sin, cos] => shape (N * L * 2)
        return np.concatenate([sin_part, cos_part], axis=-1)

    def get_crop_direction(self, K, hand_bbox):
        # Compute hand center
        x1, y1, x2, y2 = hand_bbox
        u_h, v_h = [(x1 + x2) / 2, (y1 + y2) / 2]

        # Compute ray direction of hand center
        d_h = np.linalg.inv(K) @ np.array([u_h, v_h, 1.0])
        d_h_mag = np.linalg.norm(d_h)

        z_axis = d_h / d_h_mag

        return np.concatenate([z_axis, [d_h_mag]], axis=0)

    def encode_intrinsics_old(self, K, hand_bbox):
        x1, y1, x2, y2 = hand_bbox
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        return np.array([safe_log(fx), safe_log(fy), 
                         cx - x1, cy - y1, 
                         safe_log(x2 - x1), safe_log(y2 - y1)])

    def encode_intrinsics(self, K, image_size, uv_undist):
        x1, y1 = uv_undist[0]
        x2, y2 = uv_undist[1]

        crop_w = max(1.0, x2 - x1)  
        crop_h = max(1.0, y2 - y1)

        out_w, out_h = self.image_size

        u_c, v_c = uv_undist[-1]

        p_x = (K[0, 2] - u_c) / crop_w
        p_y = (K[1, 2] - v_c) / crop_h

        org_w, org_h = image_size

        r_w = crop_w / org_w
        r_h = crop_h / org_h

        fx, fy = K[0, 0], K[1, 1]
        halfFOV_x = np.arctan( (org_w/2) / fx)
        halfFOV_y = np.arctan( (org_h/2) / fy)

        return np.array([
            p_x,
            p_y,
            safe_log(r_w),
            safe_log(r_h),
            halfFOV_x,
            halfFOV_y,
        ], dtype=np.float32)

    def create_sparse_kpe(self, camera_model, bbox, img_size):
        x1, y1, x2, y2 = bbox
        p1 = [x1, y1]
        p2 = [x2, y2]
        p3 = [x1, y2]
        p4 = [x2, y1]
        p5 = [(x1 + x2) / 2, (y1 + y2) / 2]

        sparse_kpts = np.array([p1, p2, p3, p4, p5, p5])
        sparse_kpe, uv_undist = camera_model.to_intrinsics_keypoint_encoding(sparse_kpts, return_undistorted=True)

        norm_kpts = sparse_kpts.copy()
        norm_kpts[:, 0] /= img_size[0] # Normalize x
        norm_kpts[:, 1] /= img_size[1] # Normalize y

        crop_dir = np.zeros_like(sparse_kpts)
        crop_dir[:6, 0] = self.encode_intrinsics(camera_model.get_K(), img_size, uv_undist)
        
        return np.stack([norm_kpts, sparse_kpe, crop_dir], axis=0)        

    def create_dense_kpe(self, camera_model, bbox, img_size):
        x1, y1, x2, y2 = bbox
        crop_w = max(1.0, x2 - x1)
        crop_h = max(1.0, y2 - y1)

        grid_x = np.linspace(x1, x2, int(crop_w), endpoint=False)
        grid_y = np.linspace(y1, y2, int(crop_h), endpoint=False)

        grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)  # shape: (crop_h, crop_w)

        dense_grid = np.stack([grid_xx, grid_yy], axis=-1)  # shape: (crop_h, crop_w, 2)

        dense_kpe = camera_model.to_intrinsics_keypoint_encoding(dense_grid)

        out_w, out_h = self.image_size
        sx = out_w / crop_w          
        sy = out_h / crop_h        

        K = camera_model.get_K()

        fx_prime = K[0, 0] * sx
        fy_prime = K[1, 1] * sy
        cx_prime = (K[0, 2] - x1) * sx         
        cy_prime = (K[1, 2] - y1) * sy

        return np.array([
            safe_log(fx_prime),
            safe_log(fy_prime),
            cx_prime,
            cy_prime,
            safe_log(crop_w),
            safe_log(crop_h),
        ], dtype=np.float32)




        # norm_kpts = sparse_kpts.copy()
        # norm_kpts[:, 0] /= img_size[0] # Normalize x
        # norm_kpts[:, 1] /= img_size[1] # Normalize y

        # crop_dir = np.zeros_like(sparse_kpts)
        # crop_dir[:6, 0] = self.encode_intrinsics(, bbox)
        
        # return np.stack([norm_kpts, sparse_kpe, crop_dir], axis=0)        


    def create_full_kpe(self, camera_model, j2d, bbox, img_size):
        x1, y1, x2, y2 = bbox
        p1 = [x1, y1]
        p2 = [x2, y2]
        p3 = [x1, y2]
        p4 = [x2, y1]
        p5 = [(x1 + x2) / 2, (y1 + y2) / 2]

        sparse_kpts = np.array([p1, p2, p3, p4, p5] + j2d.tolist())
        sparse_kpe, uv_undist = camera_model.to_intrinsics_keypoint_encoding(sparse_kpts, return_undistorted=True)

        norm_kpts = sparse_kpts.copy()
        norm_kpts[:, 0] /= img_size[0] # Normalize x
        norm_kpts[:, 1] /= img_size[1] # Normalize y

        crop_dir = np.zeros_like(sparse_kpts)
        crop_dir[:6, 0] = self.encode_intrinsics(camera_model.get_K(), img_size, uv_undist)
        
        return np.stack([norm_kpts, sparse_kpe, crop_dir], axis=0)        

    def process_crop(self, image, mask, bbox, j2d, visible, camera_model):
        bbox = np.int32(bbox)
        
        crop_w = bbox[2] - bbox[0]
        crop_h = bbox[3] - bbox[1]

        K = camera_model.get_K()

        if crop_h <= 0 or crop_w <= 0:
            crop_w = crop_h = 1
            visible = 0
            image_crop = np.zeros((224, 224, 3))
            mask_crop = np.zeros((224, 224))
        else:
            j2d_crop = self.crop_j2d(j2d, bbox)

            if self.undistort_inp:                    
                image_crop, K_T = undistort_local_patch(image, bbox, K, camera_model.distortion_model, out_size=(crop_w, crop_h))
                j2d = undistort_keypoints(j2d_crop, K_T, camera_model.distortion_model)

                mask_crop, K_T = undistort_local_patch(mask, bbox, K, camera_model.distortion_model, out_size=(crop_w, crop_h))

                K = K_T
            else:
                image_crop = self.get_hand_crop(image, bbox, visible)

                j2d = j2d_crop.copy()
                mask_crop = self.get_hand_crop(mask, bbox, visible)

            mask_crop = self.refine_segmentation(mask_crop, j2d)

            image_crop = cv2.resize(image_crop, (self.image_size[0], self.image_size[1]))
            mask_crop = cv2.resize(mask_crop, (self.image_size[0], self.image_size[1]))

            j2d[:, 0] = j2d[:, 0] * self.image_size[0] / crop_w
            j2d[:, 1] = j2d[:, 1] * self.image_size[1] / crop_h

        crop_size = (crop_w, crop_h)

        return image_crop, mask_crop, j2d, crop_size, visible, K

    def __getitem__(self, index):
        override_hand_type = self.hand_type
        requested_hand_type = None
        if isinstance(index, tuple):
            index, requested_hand_type = index

        if override_hand_type is None:
            hand_type = requested_hand_type
        else:
            hand_type = override_hand_type


        data = self.target_dataset[index]
        camera_model = data['extras']['camera_model']
        camera_params = data['camera_params']
        
        sequence_name = data['extras']['sequence_name']
        annotation_key = data['extras']['annotation_key']
        dataset = data['extras']['dataset']

        image = data['extras']['rgb_np']
        org_img_h, org_img_w = image.shape[:2]
        org_img_size = (org_img_w, org_img_h)
        hand_mask = data['extras']['hand_mask_np'] 
        arm_mask = data['extras']['arm_mask_np'] 
        hand_params = data['hand_params']
        arm_params = data['arm_params']

        if hand_type is None:
            valid_hand_types = []
            for hand_type in ['left', 'right']:
                if hand_type not in hand_params: continue
                if hand_params[hand_type]['visible_hand'] == 0: continue

                valid_hand_types.append(hand_type)

            if len(valid_hand_types) == 0:
                hand_type = 'right' # Default to right hand           
            else:
                hand_type = random.choice(valid_hand_types)

        hand_param = hand_params[hand_type]
        arm_param = arm_params[hand_type]

        visible_hand = hand_param['visible_hand']
        global_orient = hand_param['camera_global_orient']
        betas = hand_param['camera_betas']
        hand_pose = hand_param['hand_pose']
        hand_j3d = hand_param['camera_j3D']
        hand_j2d = hand_param['camera_j2D']
        transl = hand_param['camera_transl']
        hand_bbox = hand_param['camera_hand_box']

        hand_type = 0 if hand_type == 'left' else 1

        visible_arm = arm_param['visible_arm']
        arm_j3d = arm_param['camera_j3D']
        arm_j2d = arm_param['camera_j2D']
        arm_T = arm_param['camera_T']
        arm_R = arm_param['camera_R']
        arm_shape = arm_param['camera_shape']
        arm_bbox = arm_param['camera_arm_box']
        valid_arm = (arm_j3d != 0).any() or (arm_shape != 0).any()

        left_anno, right_anno = get_arm_bboxes(self.inferencer, image)

        if hand_type == 'right':
            arm_anno = right_anno
        else:
            arm_anno = left_anno

        if arm_anno is not None:
            arm_bbox = np.array(arm_anno['bbox'])
            arm_j2d = np.array(arm_anno['keypoints'])
            valid_arm = visible_arm = 1
            

        # if np.isnan(global_orient).any() or np.isinf(global_orient).any():
        #     print("global_orient is not finite", annotation_key, global_orient)
        # if np.isnan(betas).any() or np.isinf(betas).any():
        #     print("betas is not finite", annotation_key, betas)
        # if np.isnan(hand_pose).any() or np.isinf(hand_pose).any():
        #     print("hand_pose is not finite", annotation_key, hand_pose)
        # if np.isnan(transl).any() or np.isinf(transl).any():
        #     print("transl is not finite", annotation_key, transl)
        # if np.isnan(hand_j3d).any() or np.isinf(hand_j3d).any():
        #     print("hand_j3d is not finite", annotation_key, hand_j3d)
        # if np.isnan(hand_j2d).any() or np.isinf(hand_j2d).any():
        #     print("hand_j2d is not finite", annotation_key, hand_j2d)
 
        # if np.isnan(arm_j3d).any() or np.isinf(arm_j3d).any():
        #     print("arm_j3d is not finite", annotation_key, arm_j3d)
        # if np.isnan(arm_j3d).any() or np.isinf(arm_j3d).any():
        #     print("arm_j3d is not finite", annotation_key, arm_j3d)
        # if np.isnan(arm_T).any() or np.isinf(arm_T).any():
        #     print("arm_T is not finite", annotation_key, arm_T)
        # if np.isnan(arm_R).any() or np.isinf(arm_R).any():
        #     print("arm_R is not finite", annotation_key, arm_R)

        hand_j2d = np.nan_to_num(hand_j2d, nan=-1, posinf=-1, neginf=-1)
        arm_j2d = np.nan_to_num(arm_j2d, nan=-1, posinf=-1, neginf=-1)

        aug_params = None
        if self.is_train:
            aug_params = get_aug_config()
            scale, rot, color_scale, do_flip = aug_params

            hand_bbox = bbox_augmentation(hand_bbox, org_img_size, scale, rot)
            arm_bbox = bbox_augmentation(arm_bbox, org_img_size, scale, rot)

        hand_sparse_kpe = self.create_sparse_kpe(camera_model, hand_bbox, org_img_size)
        arm_sparse_kpe = self.create_full_kpe(camera_model, arm_j2d[1:], arm_bbox, org_img_size)

        hand_crop, hand_mask_crop, hand_j2d, hand_crop_size, visible_hand, K_hand = self.process_crop(image, hand_mask, hand_bbox, hand_j2d, visible_hand, camera_model)
        try:
            arm_crop, arm_mask_crop, arm_j2d, arm_crop_size, visible_arm, K_arm = self.process_crop(image, arm_mask, arm_bbox, arm_j2d, visible_arm, camera_model)
        except Exception as e:
            arm_crop = np.zeros((*self.image_size, 3))
            arm_mask_crop = np.zeros(self.image_size)
            arm_j2d = np.zeros((3, 2))
            arm_crop_size = (0, 0)
            visible_arm = 0
            K_arm = camera_model.get_K()
        
        if self.is_train:
            scale, rot, color_scale, do_flip = aug_params

            if visible_hand:
                hand_crop = color_augmentation(hand_crop, color_scale)
                hand_crop = self.hard_color_augmentation(hand_crop)
                hand_crop = self.cut_mix(hand_crop, image)

            if visible_arm:
                arm_crop = color_augmentation(arm_crop, color_scale)
                arm_crop = self.hard_color_augmentation(arm_crop)
                arm_crop = self.cut_mix(arm_crop, image)

        hand_pose_2d, hand_vis_2d = self.generate_sa_simdr(hand_j2d, visible_hand, num_joints=21)
        arm_pose_2d, arm_vis_2d = self.generate_sa_simdr(arm_j2d, visible_arm, num_joints=3)

        hand_hms, _ = self.generate_hms2d(hand_j2d, visible_hand, num_joints=21)
        arm_hms, _ = self.generate_hms2d(arm_j2d, visible_arm, num_joints=3)

        hand_crop = torch.tensor(hand_crop).permute(2, 0, 1).float() / 255.0
        hand_mask_crop = torch.tensor(hand_mask_crop).float() 
        hand_j2d = torch.tensor(hand_j2d).float()
        hand_j3d = torch.tensor(hand_j3d).float()
        hand_sparse_kpe = torch.tensor(hand_sparse_kpe).float()
        hand_type = torch.tensor(hand_type).long()

        hand_pose = torch.tensor(hand_pose).float()
        global_orient = torch.tensor(global_orient).float()
        betas = torch.tensor(betas).float()
        transl = torch.tensor(transl).float()
        visible_hand = torch.tensor(visible_hand).float()
        hand_bbox = torch.tensor(hand_bbox).float()
        hand_crop_size = torch.tensor(hand_crop_size).float()
        
        arm_crop = torch.tensor(arm_crop).permute(2, 0, 1).float() / 255.0
        arm_mask_crop = torch.tensor(arm_mask_crop).float()
        arm_j2d = torch.tensor(arm_j2d).float()
        arm_j3d = torch.tensor(arm_j3d).float()
        arm_sparse_kpe = torch.tensor(arm_sparse_kpe).float()
        visible_arm = torch.tensor(visible_arm).float()
        valid_arm = torch.tensor(valid_arm).float()
        arm_T = torch.tensor(arm_T).float()
        arm_R = torch.tensor(arm_R).float()
        arm_shape = torch.tensor(arm_shape).float()
        arm_bbox = torch.tensor(arm_bbox).float()
        arm_crop_size = torch.tensor(arm_crop_size).float()

        hand_pose_2d = torch.tensor(hand_pose_2d).float()
        hand_vis_2d = torch.tensor(hand_vis_2d).float()
        arm_pose_2d = torch.tensor(arm_pose_2d).float() 
        arm_vis_2d = torch.tensor(arm_vis_2d).float()

        K_hand = torch.tensor(K_hand).float()
        K_arm = torch.tensor(K_arm).float()

        hand_hms = torch.tensor(hand_hms).float()
        arm_hms = torch.tensor(arm_hms).float()
        
        samplekey = f'{dataset}@{data["extras"]["index"]}@{annotation_key}'
        samplekey = torch.nn.functional.pad(torch.tensor([ord(c) for c in samplekey], dtype=torch.int64), (0, 100 - len(samplekey)), value=0)[:100]

        data = {
            'hand_crop': hand_crop,
            'hand_mask': hand_mask_crop,
            'hand_j2d': hand_j2d,
            'hand_j3d': hand_j3d,
            'hand_sparse_kpe': hand_sparse_kpe,
            'hand_type': hand_type,
            'hand_pose': hand_pose.view(15, -1),
            'global_orient': global_orient,
            'betas': betas,
            'transl': transl,
            'visible_hand': visible_hand,

            'arm_crop': arm_crop,
            'arm_mask': arm_mask_crop,
            'arm_j2d': arm_j2d,
            'arm_j3d': arm_j3d,
            'arm_sparse_kpe': arm_sparse_kpe,
            'visible_arm': visible_arm,
            'valid_arm': valid_arm,
            'arm_T': arm_T,
            'arm_R': arm_R,
            'arm_shape': arm_shape, 

            'hand_pose_2d': hand_pose_2d,
            'hand_vis_2d': hand_vis_2d,
            'arm_pose_2d': arm_pose_2d,
            'arm_vis_2d': arm_vis_2d,

            'hand_hms': hand_hms,
            'arm_hms': arm_hms,
        }

        meta = {
            "camera_type": torch.tensor(camera_params['camera_type']).float(),
            "focal_length": torch.tensor(camera_params['focal_length']).float(),
            "principal_point": torch.tensor(camera_params['principal_point']).float(),
            "projection_params": torch.tensor(camera_params['projection_params']).float(),
            "camera_type": torch.tensor(camera_params['camera_type']).float(),
            "org_img_size": torch.tensor([org_img_w, org_img_h]),
            "hand_bbox": hand_bbox,
            "hand_crop_size": hand_crop_size,
            "arm_bbox": arm_bbox,
            "arm_crop_size": arm_crop_size,
            "is_undistorted": torch.tensor(self.undistort_inp).bool(),
            "K_hand": K_hand,
            "K_arm": K_arm,
            "samplekey":samplekey,
        }

        if self.return_complete_image:
            meta['image'] = torch.tensor(image)

        return data, meta

    def generate_sa_simdr(self, joints, hand_vis, num_joints, simdr_split_ratio=1.0):
        vis = np.ones((num_joints, 1), dtype=np.float32) * hand_vis

        sigma = 4
        image_size = self.image_size
        
        J = joints.shape[0]
        W = int(image_size[0] * simdr_split_ratio)
        H = int(image_size[1] * simdr_split_ratio)

        # scaled joint centers
        mus_x = joints[:, 0] * simdr_split_ratio
        mus_y = joints[:, 1] * simdr_split_ratio

        # 1D coordinate vectors
        xs = np.arange(W, dtype=np.float32)
        ys = np.arange(H, dtype=np.float32)

        # compute outer (J×W) and (J×H) distance matrices
        dx2 = (xs[None, :] - mus_x[:, None]) ** 2  # shape (J, W)
        dy2 = (ys[None, :] - mus_y[:, None]) ** 2  # shape (J, H)

        # raw Gaussian (before truncation or normalization)
        gauss_x = np.exp(-dx2 / (2 * sigma**2))
        gauss_y = np.exp(-dy2 / (2 * sigma**2))

        # truncate outside ±3σ window for efficiency
        radius = 3 * sigma
        mask_x = (np.abs(xs[None, :] - mus_x[:, None]) <= radius)
        mask_y = (np.abs(ys[None, :] - mus_y[:, None]) <= radius)
        gauss_x *= mask_x
        gauss_y *= mask_y

        # normalize so ∑ gauss = 1 for each joint
        norm_x = gauss_x.sum(axis=1, keepdims=True) + 1e-8
        norm_y = gauss_y.sum(axis=1, keepdims=True) + 1e-8
        gauss_x /= norm_x
        gauss_y /= norm_y

        # apply visibility: zero out rows for invisible joints
        vis = vis.reshape(J, 1).astype(np.float32)
        gauss_x *= vis
        gauss_y *= vis

        # final targets: shape (J, 2, W_or_H)
        targets = np.stack([gauss_x, gauss_y], axis=1)

        # weight is just your visibility mask, but you could also 
        # set to zero if center falls outside image:
        tmp_size = sigma * 3  * simdr_split_ratio

        in_bounds = (
            ((tmp_size + mus_x) >= 0) & ((mus_x - tmp_size) < W) &
            ((tmp_size + mus_y) >= 0) & ((mus_y - tmp_size) < H)
        ).astype(np.float32).reshape(J, 1)
        target_weight = vis * in_bounds

        return targets, target_weight

    def adjust_target_weight(self, joint, target_weight, tmp_size):
        # feat_stride = self.image_size / self.heatmap_size
        mu_x = joint[0]
        mu_y = joint[1]
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= (self.image_size[0]) or ul[1] >= self.image_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight = 0

        return target_weight


    def generate_hms2d(self, joints, hand_vis, num_joints):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((num_joints, 1), dtype=np.float32) * hand_vis

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            feat_stride = self.image_size / self.heatmap_size

            
            for joint_id in range(num_joints):
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight

    def generate_d_hms_target(self, d3d, joints_vis):
        '''
        Generates 1D Gaussian heatmaps for the distance of joints.
        '''
        self.heatmap_length_1d = 1000
        self.sigma_z = 1
        
        # Initialize target and weight arrays
        target_1d = np.zeros((self.num_joints, self.heatmap_length_1d), dtype=np.float32)
        target_weight_1d = np.ones((self.num_joints, 1), dtype=np.float32)
        
        tmp_size_z = int(self.sigma_z * 3)
        sigma_z_sq_2 = 2 * (self.sigma_z ** 2)
        
        for joint_id in range(self.num_joints):
            z = d3d[joint_id, 0] * 1000 # Convert to mm
            v_z = joints_vis[joint_id, 0]

            if v_z < 0.5:
                target_weight_1d[joint_id] = 0
                continue

            mu_z = int(z + 0.5)

            # Define the range of the Gaussian
            ul_z = mu_z - tmp_size_z
            br_z = mu_z + tmp_size_z + 1

            # Check if the Gaussian is out of bounds
            if ul_z >= self.heatmap_length_1d or ul_z < 0 or br_z < 0 or br_z > self.heatmap_length_1d:
                target_weight_1d[joint_id] = 0
                continue

            # Generate Gaussian
            size_z = 2 * tmp_size_z + 1
            z_range = np.arange(0, size_z, 1, np.float32)
            z_center = size_z // 2
            g_z = np.exp(- ((z_range - z_center) ** 2) / sigma_z_sq_2)

            # Usable Gaussian range
            g_z_start = max(0, -ul_z)
            g_z_end = min(br_z, self.heatmap_length_1d) - ul_z

            # Image (heatmap) range
            img_z_start = max(0, ul_z)
            img_z_end = min(br_z, self.heatmap_length_1d)

            # Assign Gaussian to the target heatmap
            target_1d[joint_id, img_z_start:img_z_end] = g_z[g_z_start:g_z_end]

        return target_1d, target_weight_1d

    def evaluate_joints(cls, cfg, all_gt_j3ds, all_preds_j3d, all_vis_j3d, root_joint):
        errors, error_rr, errors_pa = compute_3d_errors_batch(all_gt_j3ds, all_preds_j3d, all_vis_j3d, root_joint)
        accc_error = compute_acceleration_error(all_gt_j3ds, all_preds_j3d, all_vis_j3d)

        N_JOINTS = errors.shape[0]

        MPJPE = np.mean(errors)
        RR_MPJPE = np.mean(error_rr) 
        PAMPJPE = np.mean(errors_pa)
        MACC = np.mean(accc_error)

        name_values = []

        for i in range(N_JOINTS):
            name_values.append((f'Joint_{i}_MPJPE', errors[i]))
        
        for i in range(N_JOINTS):
            name_values.append((f'Joint_{i}_RR_MPJPE', error_rr[i]))

        for i in range(N_JOINTS):
            name_values.append((f'Joint_{i}_PAMPJPE', errors_pa[i]))

        for i in range(N_JOINTS):
            name_values.append((f'Joint_{i}_ACCER', accc_error[i]))

        name_values.append(('MPJPE', MPJPE))
        name_values.append(('RR_MPJPE', RR_MPJPE))
        name_values.append(('PAMPJPE', PAMPJPE))
        name_values.append(('MEAN_ACC', MACC))


        # heatmap_sequence = ["Head", # 0
        #                     "Neck", # 1
        #                     "Right_shoulder", # 2 
        #                     "Right_elbow", # 3
        #                     "Right_wrist", # 4
        #                     "Left_shoulder", # 5
        #                     "Left_elbow", # 6
        #                     "Left_wrist", # 7
        #                     "Right_hip", # 8
        #                     "Right_knee", # 9
        #                     "Right_ankle", # 10
        #                     "Right_foot", # 11
        #                     "Left_hip", # 12 
        #                     "Left_knee", # 13
        #                     "Left_ankle", #14
        #                     "Left_foot"] # 15

        # for i, joint_name in enumerate(heatmap_sequence):
        #     name_values.append((f'{joint_name}_MPJPE', errors[i]))
        # name_values.append(('MPJPE', MPJPE))

        # for i, joint_name in enumerate(heatmap_sequence):
        #     name_values.append((f'{joint_name}_PAMPJPE', errors_pa[i]))
        # name_values.append(('PAMPJPE', PAMPJPE))

        name_values = OrderedDict(name_values)

        return name_values, MPJPE


def main():
    from settings import config as cfg
    from datasets import HOT3DLoader 

    train_dataset = DistanceDataset(cfg, HOT3DLoader(cfg.DATASET.ROOT, get_camera=True, split='train'))

    for i in range(len(train_dataset)):
        data, meta = train_dataset[i]
        print(data.keys())
        print(meta.keys())
        break


if __name__ == '__main__':
    main()