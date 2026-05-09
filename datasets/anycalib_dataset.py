import os; import sys; 
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)

import torch
import os
import numpy as np
import cv2
import random

from collections import OrderedDict
from torch.utils.data import Dataset
from camera_models import ToPinholeCamera, OVR624CameraModel, PinholeCameraModel
from camera_models import Rational8CameraModel


def safe_log(x, eps=1e-6):
    return np.log(np.maximum(x, eps))


class AnyCalibDatasetPin(Dataset):
    def __init__(self, target_datset):
        self.target_dataset = target_datset
        self.split = target_datset.split
        self.length = len(self.target_dataset)
        self.get_camera = target_datset.get_camera
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
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

        height, width = org_img_h, org_img_w

        if data['extras']['dataset'] == 'HOT3D':
            camera_params['focal_length'] = np.array([603.6017456054688, 603.6017456054688])
            camera_params['principal_point'] = np.array([705.95947265625, 703.4464111328125])

            camera_params['projection_params'][0] = 0.12783679366111755
            camera_params['projection_params'][1] = -0.017603283748030663
            camera_params['projection_params'][2] = 0
            camera_params['projection_params'][3] = 0
            camera_params['projection_params'][4] = -0.011532214470207691
            camera_params['projection_params'][5] = 0.002314441138878464
            camera_params['projection_params'][6] = 0.0
            camera_params['projection_params'][7] = 0.0
            
            
            data['camera_params'] = camera_params
            data['camera_params']['camera_type'] = 2 


            if self.get_camera:
                focal_length = camera_params['focal_length']
                principal_point = camera_params['principal_point'] 
                projection_params = camera_params['projection_params'][:8]

                # dist_coeffs: 8-length array [k1, k2, p1, p2, k3, k4, k5, k6]
                camera_model = Rational8CameraModel(focal_length, principal_point, projection_params, width, height)
                data['extras']['camera_model'] = camera_model

        return data
    


class AnyCalibDataset624(Dataset):
    def __init__(self, target_datset):
        self.target_dataset = target_datset
        self.split = target_datset.split
        self.length = len(self.target_dataset)
        self.get_camera = target_datset.get_camera
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
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

        height, width = org_img_h, org_img_w

        if data['extras']['dataset'] == 'HOT3D':
            camera_params['focal_length'] = np.array([603.6017456054688, 603.6017456054688])
            camera_params['principal_point'] = np.array([705.95947265625, 703.4464111328125])

            camera_params['projection_params'] = np.zeros(15)
            camera_params['projection_params'][0] = 603.6017456054688
            camera_params['projection_params'][1] = 705.95947265625
            camera_params['projection_params'][2] = 703.4464111328125

            camera_params['projection_params'][3] = 0.12783679366111755 # k1
            camera_params['projection_params'][4] = -0.017603283748030663 # k2
            camera_params['projection_params'][5] = -0.011532214470207691 # k3
            camera_params['projection_params'][6] = 0.002314441138878464 # k4

            camera_params['projection_params'][7] = 0.0 # k5
            camera_params['projection_params'][8] = 0.0 # k6
            camera_params['projection_params'][9] = 0.0 # p1
            camera_params['projection_params'][10] = 0.0 # p2
            camera_params['projection_params'][11] = 0.0 # s1
            camera_params['projection_params'][12] = 0.0 # s2
            camera_params['projection_params'][13] = 0.0 # s3
            camera_params['projection_params'][14] = 0.0 # s4

            
            data['camera_params'] = camera_params
            data['camera_params']['camera_type'] = 3 # 3 for fisheye624


            if self.get_camera:
                focal_length = camera_params['focal_length']
                principal_point = camera_params['principal_point'] 
                projection_params = camera_params['projection_params']

                camera_model = OVR624CameraModel(focal_length, principal_point, projection_params[3:], width, height)
                data['extras']['camera_model'] = camera_model

        return data