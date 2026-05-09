import tarfile
import json
import torch
import asyncio
import os
import sys
import numpy as np
import h5py
import cv2
import pickle
import random
import aiofiles
from async_lru import alru_cache

from tqdm import tqdm
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from datapipes.base_pipeline import BasePipelineCreator
from datapipes.decoders.image_decoder import ImageDecoder


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from camera_models import PinholeCameraModel
    
    
class FreiHANDLoader(Dataset):
    IMG_WIDTH = 224
    IMG_HEIGHT = 224

    def __init__(self, data_root, split, get_camera=False, filter_only_hands=True, **kwargs) -> None:
        self.config = kwargs['config']

        self.split = split

        assert self.split in ['val', 'test']

        data_root = f'{data_root}/FreiHAND_Eval'
    
        self.anno_root = f"{data_root}/annotation"
        self.data_root = f"{data_root}/rgb"
        self.arm_root = f"{data_root}/armpreds"

        self.sample_keys = os.listdir(self.data_root)
        self.sample_keys = [k[:-4] for k in self.sample_keys if k.endswith('.jpg')]

        self.get_camera = get_camera
        self.is_rot6d = self.config.POSE_3D.ROT_6D
        self.rot_dim = 6 if self.is_rot6d else 3

        print(f"Loading FreiHAND dataset from {data_root}, {len(self.sample_keys)} samples")

        self.n_samples = len(self.sample_keys)

    def __len__(self):
        return self.n_samples
    
    def get_image(self, sample_name):
        sample_name = sample_name + '.jpg'
        rgb = cv2.imread(os.path.join(self.data_root, sample_name))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb

    def __getitem__(self, index):
        sample_name = self.sample_keys[index]

        rgb = self.get_image(sample_name)
        height, width = rgb.shape[:2]

        valid_image_space = np.ones(rgb.shape[:2], dtype=np.uint8)

        anno_data = self.get_annotation(sample_name, width, height)

        camera_pose = anno_data["camera_pose"]
        camera_params = anno_data["camera_params"]
        hand_params = anno_data["hand_params"]
        arm_params = anno_data["arm_params"]

        hand_mask = np.zeros_like(rgb)
        arm_mask = np.zeros_like(rgb)
        
        data = dict()
        data['hand_params'] = hand_params
        data['arm_params'] = arm_params
        data['camera_params'] = camera_params
        data['camera_params']['camera_type'] = 0 # Pinhole
        data['extras'] = dict()
        data['extras']['rgb_np'] = rgb
        data['extras']['hand_mask_np'] = hand_mask
        data['extras']['arm_mask_np'] = arm_mask
        data['extras']['sequence_name'] = f'validation'
        data['extras']['valid_image_space'] = valid_image_space
        data['extras']['annotation_key'] = sample_name
        data['extras']['dataset'] = 'handco'
        data['extras']['index'] = index
        
        if self.get_camera:
            focal_length = camera_params['focal_length']
            principal_point = camera_params['principal_point']
            height, width = rgb.shape[:2]
            focal_length = np.array(focal_length)
            principal_point = np.array(principal_point)
 
            camera_model = PinholeCameraModel(focal_length, principal_point, width, height)
            
            data['extras']['camera_model'] = camera_model

        return data

    def get_annotation(self, sample_name, width, height):
        with open(f'{self.anno_root}/{sample_name}.json', 'r') as f:
            hand_data = json.load(f)
        
        with open(f'{self.arm_root}/{sample_name}.json', 'r') as f:
            arm_data = json.load(f)

        is_rot_6d = '_rot6d' if self.is_rot6d else ''   

        K = np.array(hand_data['K'])
        visible_hand = valid_hand = hand_data['visible']
        hand_j3d = np.array(hand_data['j3d'])
        hand_j2d = np.array(hand_data['j2d'])
        hand_bbox = np.array(hand_data['hand_bbox'])
        betas = np.array(hand_data['betas']).reshape(-1)
        global_orient = np.array(hand_data['global_orient' + is_rot_6d]).reshape(-1)
        hand_pose = np.array(hand_data['hand_pose' + is_rot_6d]).reshape((-1, 6) if self.is_rot6d else -1)
        transl = np.array(hand_data['transl']).reshape(-1)

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        if len(arm_data):
            arm_data = arm_data[0]
            arm_bbox = np.array(arm_data['bbox'])
            arm_keypoints_2d = np.array(arm_data['keypoints'])
            valid_arm = True
            visible_arm = True
        else:
            arm_bbox = np.array([0, 0, 0, 0])
            arm_keypoints_2d = np.zeros((3, 2))
            valid_arm = False
            visible_arm = False

        arm_keypoints_2d[:, 0] = np.clip(arm_keypoints_2d[:, 0], 0, width)
        arm_keypoints_2d[:, 1] = np.clip(arm_keypoints_2d[:, 1], 0, height)
        arm_bbox = np.clip(arm_bbox, 0, [width, height, width, height])


        hand_j2d[:, 0] = np.clip(hand_j2d[:, 0], 0, width)
        hand_j2d[:, 1] = np.clip(hand_j2d[:, 1], 0, height)
        hand_bbox = np.clip(hand_bbox, 0, [width, height, width, height])

        if fx == 0: # Invalid camera
            fx = fy = 1 

        padded_dist = np.zeros(15, dtype=np.float32)


        return {
            "key": sample_name,  # Decode string
            "camera_params": {
                "focal_length": (fx, fy), 
                "principal_point": (cx, cy), 
                "projection_params": padded_dist,
                "width": width, "height": height,
            },
            "camera_pose": np.eye(4, dtype=np.float32),

            "hand_params": {
                "left": {
                    "valid_hand": 0,
                    "visible_hand": 0,
                    "camera_global_orient": np.zeros(6 if is_rot_6d else 3, dtype=np.float32),
                    "hand_pose": np.zeros((15, 6) if is_rot_6d else 45, dtype=np.float32),
                    "camera_betas": np.zeros(10, dtype=np.float32),
                    "camera_transl": np.zeros(3, dtype=np.float32),
                    "camera_j3D": np.zeros((21, 3), dtype=np.float32),
                    "camera_j2D": np.zeros((21, 2), dtype=np.float32),
                    "camera_hand_box": np.array([0, 0, 0, 0]),
                },
                "right": {
                    "valid_hand": valid_hand,
                    "visible_hand": visible_hand,
                    "camera_global_orient": global_orient,
                    "hand_pose": hand_pose,
                    "camera_betas": betas,
                    "camera_transl": transl,
                    "camera_j3D": hand_j3d,
                    "camera_j2D": hand_j2d,
                    "camera_hand_box": hand_bbox,
                    }
            },
            "arm_params": {
                "left": {
                    "valid_arm": 0,
                    "visible_arm": 0,
                    "camera_shape": np.zeros(5, dtype=np.float32),
                    "camera_R": np.zeros(6 if is_rot_6d else 3, dtype=np.float32),
                    "camera_T": np.zeros(3, dtype=np.float32),
                    "camera_j3D": np.zeros((3, 3), dtype=np.float32),
                    "camera_j2D": np.zeros((3, 2), dtype=np.float32),
                    "camera_arm_box": np.array([0, 0, 0, 0]),
                },
                "right": {
                    "valid_arm": valid_arm,
                    "visible_arm": visible_arm,
                    "camera_shape": np.zeros(5, dtype=np.float32),
                    "camera_R": np.zeros(6 if is_rot_6d else 3, dtype=np.float32),
                    "camera_T": np.zeros(3, dtype=np.float32),
                    "camera_j3D": np.zeros((3, 3), dtype=np.float32),
                    "camera_j2D": arm_keypoints_2d,
                    "camera_arm_box": arm_bbox,
                }
            }
        }

