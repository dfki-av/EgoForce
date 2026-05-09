import os
import sys
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from utils.augmentation import get_aug_config, color_augmentation, bbox_augmentation, undistort_local_patch, undistort_keypoints


def safe_log(x, eps=1e-6):
    return np.log(np.maximum(x, eps))


class DemoHandArmLoader(Dataset):
    def __init__(self, cfg, camera_model, undistort_inp=True, return_complete_image=False, hand_type=None):
        self.cfg = cfg

        self.camera_model = camera_model

        self.num_joints = cfg.NUM_JOINTS_PER_HAND
        self.heatmap_size = np.array(cfg.POSE_3D.HEATMAP_SIZE)
        self.target_type = cfg.POSE_3D.TARGET_TYPE
        self.sigma = cfg.POSE_3D.SIGMA
        self.image_size = cfg.POSE_3D.IMAGE_SIZE
        self.undistort_inp = undistort_inp
        self.return_complete_image = return_complete_image  
        self.hand_type = hand_type

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
        crop_dir[:6, 0] = self.encode_intrinsics(camera_model.get_K(), bbox, img_size, uv_undist)
        
        return np.stack([norm_kpts, sparse_kpe, crop_dir], axis=0)        

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

    def process_crop(self, image, bbox, j2d, visible, camera_model):        
        crop_w = bbox[2] - bbox[0]
        crop_h = bbox[3] - bbox[1]

        K = camera_model.get_K()

        if crop_h <= 0 or crop_w <= 0:
            crop_w = crop_h = 1
            visible = 0
            image_crop = np.zeros((224, 224, 3))
        else:
            j2d_crop = self.crop_j2d(j2d, bbox)

            if self.undistort_inp:                    
                image_crop, K_T = undistort_local_patch(image, bbox, K, camera_model.distortion_model, out_size=(crop_w, crop_h))
                j2d = undistort_keypoints(j2d_crop, K_T, camera_model.distortion_model)
            else:
                image_crop = self.get_hand_crop(image, bbox, visible)

                j2d = j2d_crop.copy()

            image_crop = cv2.resize(image_crop, (self.image_size[0], self.image_size[1]))

            j2d[:, 0] = j2d[:, 0] * self.image_size[0] / crop_w
            j2d[:, 1] = j2d[:, 1] * self.image_size[1] / crop_h


        x1, y1, x2, y2 = bbox

        K_sub = K.copy()
        K_sub[0, 2] -= x1
        K_sub[1, 2] -= y1


        crop_size = (crop_w, crop_h)

        return image_crop, j2d, crop_size, visible, K_sub

    def transform(self, image, bounding_box):
        camera_model = self.camera_model

        org_img_h, org_img_w = image.shape[:2]
        org_img_size = (org_img_w, org_img_h)

        hand_type = 0 if self.hand_type == 'left' else 1

        if 'hand' in bounding_box:
            visible_hand = 1
            hand_j2d = np.array(bounding_box['hand']['keypoint'], dtype=np.int32)
            hand_bbox = np.array(bounding_box['hand']['bbox'], dtype=np.int32)
        else:
            visible_hand = 0
            hand_j2d = np.zeros((3, 2), dtype=np.int32)
            hand_bbox = np.zeros((4,), dtype=np.int32)
            
        if 'arm' in bounding_box:
            visible_arm = valid_arm = 1
            arm_j2d = np.array(bounding_box['arm']['keypoint'], dtype=np.int32)
            arm_bbox = np.array(bounding_box['arm']['bbox'], dtype=np.int32)
        else:
            visible_arm = valid_arm = 0        
            arm_j2d = np.zeros((3, 2), dtype=np.int32)
            arm_bbox = np.zeros((4,), dtype=np.int32)

        hand_wrist_kpe = hand_sparse_kpe = self.create_sparse_kpe(camera_model, hand_bbox, org_img_size)
        arm_sparse_kpe = arm_full_kpe = self.create_sparse_kpe(camera_model, arm_bbox, org_img_size)


        hand_crop, hand_j2d, hand_crop_size, visible_hand, K_hand = self.process_crop(image, hand_bbox, hand_j2d, visible_hand, camera_model)
        try:
            arm_crop, arm_j2d, arm_crop_size, visible_arm, K_arm = self.process_crop(image, arm_bbox, arm_j2d, visible_arm, camera_model)
        except Exception as e:
            arm_crop = np.zeros((*self.image_size, 3))
            arm_mask_crop = np.zeros(self.image_size)
            arm_j2d = np.zeros((3, 2))
            arm_crop_size = (0, 0)
            visible_arm = 0
            K_arm = camera_model.get_K()

        hand_crop = torch.tensor(hand_crop).permute(2, 0, 1).float() / 255.0
        hand_sparse_kpe = torch.tensor(hand_sparse_kpe).float()
        hand_type = torch.tensor(hand_type).long()
        visible_hand = torch.tensor(visible_hand).float()
        hand_bbox = torch.tensor(hand_bbox).float()
        hand_crop_size = torch.tensor(hand_crop_size).float()

        arm_crop = torch.tensor(arm_crop).permute(2, 0, 1).float() / 255.0
        arm_j2d = torch.tensor(arm_j2d).float()
        arm_sparse_kpe = torch.tensor(arm_sparse_kpe).float()
        visible_arm = torch.tensor(visible_arm).float()
        valid_arm = torch.tensor(valid_arm).float()
        arm_bbox = torch.tensor(arm_bbox).float()
        arm_crop_size = torch.tensor(arm_crop_size).float()
        
        K_hand = torch.tensor(K_hand).float()
        K_arm = torch.tensor(K_arm).float()

        data = {
            'hand_crop': hand_crop,
            'hand_j2d': hand_j2d,
            'hand_sparse_kpe': hand_sparse_kpe,
            'hand_type': hand_type,
            'visible_hand': visible_hand,

            'arm_crop': arm_crop,
            'arm_j2d': arm_j2d,
            'arm_sparse_kpe': arm_sparse_kpe,
            'visible_arm': visible_arm,
            'valid_arm': valid_arm,
        }   

        camera_type = camera_model.TYPE_ID
        focal_length = camera_model.f
        principal_point = camera_model.c
        projection_params = camera_model.params

        projection_params = np.concatenate([np.array([focal_length[0]]), principal_point, projection_params], axis=0)
        
        meta = {
            "camera_type": torch.tensor(camera_type).float(),
            "focal_length": torch.tensor(focal_length).float(),
            "principal_point": torch.tensor(principal_point).float(),
            "projection_params": torch.tensor(projection_params).float(),
            "org_img_size": torch.tensor([org_img_w, org_img_h]),
            "hand_bbox": hand_bbox,
            "hand_crop_size": hand_crop_size,
            "arm_bbox": arm_bbox,
            "arm_crop_size": arm_crop_size,
            "is_undistorted": torch.tensor(self.undistort_inp).bool(),
            "K_hand": K_hand,
            "K_arm": K_arm,
        }

        if self.return_complete_image:
            meta['image'] = torch.tensor(image)

        return data, meta
