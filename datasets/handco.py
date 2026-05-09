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


def build_tar_index(tar_path):
    index = {}
    with tarfile.open(tar_path, 'r') as tar:
        for idx, tarinfo in enumerate(tqdm(tar, desc=f"Building tar index")):
            index[tarinfo.name] = (tar_path, tarinfo.offset_data, tarinfo.size)
    return index


def _row_to_dict(row):
    if isinstance(row, np.void):
        out = {}
        for name in row.dtype.names:  # tuple of field names
            v = row[name]
            # decode bytes scalars
            if isinstance(v, (bytes, np.bytes_)):
                v = v.decode('utf-8')
            # decode arrays of bytes (rare)
            if isinstance(v, np.ndarray) and v.dtype.kind == 'S':  # fixed-width bytes
                v = v.astype(str)
            out[name] = v
        return out
    # If ever indexing returns a group (different file layout), handle that too:
    if isinstance(row, h5py.Group):
        return {k: row[k][()] for k in row.keys()}
    # Fallback: already a dict or array
    return row


class AsyncSamplerLoader:
    def __init__(self, data_root):
        self.image_decoder = ImageDecoder('rgb8')
        self.data_root = data_root

    @alru_cache(maxsize=4)
    async def async_load(self, tar_info, file_type):
        if tar_info is None: return None

        tar_path, offset_data, data_size = tar_info

        path_splits = self.data_root.split('/')
        folder_index = tar_path.find(path_splits[-1])
        tar_path = f"{'/'.join(path_splits[:-1])}/{tar_path[folder_index:]}"

        tar_path = os.path.join(self.data_root, 'shards', tar_path)

        async with aiofiles.open(tar_path, 'rb') as file_obj:
            await file_obj.seek(offset_data)
            file_data = await file_obj.read(data_size)

        if 'png' in file_type or 'jpg' in file_type:
            # Offload image decoding to a thread if it's CPU-bound.
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(None, self.image_decoder, "file_path", file_data)
        elif 'json' in file_type:
            data = json.loads(file_data)
        elif 'txt' in file_type:
            data = file_data.decode('utf-8').split('\n')
        else:
            data = file_data

        return data

    async def async_load_all(self, files):
        tasks = [self.async_load(*file) for file in files]
        results = await asyncio.gather(*tasks)
        return results

    def load_all(self, files):
        return asyncio.run(self.async_load_all(files))

    def load(self, tar_info, file_type):
        # Simply wrap the async call for a single file.
        return asyncio.run(self.async_load(tar_info, file_type))

    
    
class HandCOLoader(Dataset):
    IMG_WIDTH = 224
    IMG_HEIGHT = 224

    def build_tar_index_map(self, data_root, cams):
        self.factory = BasePipelineCreator(data_root)

        tar_index_map = dict()
        tar_file_paths = self.factory.get_tar_files_for_subsets(self.split, component_groups=cams,)

        with ThreadPoolExecutor() as executor:
            for component_name, tar_paths in tar_file_paths.items():
                tar_index_map[component_name] = dict()

                for tar_index in executor.map(build_tar_index, tar_paths):
                    tar_index_map[component_name].update(tar_index)
                
        return tar_index_map

    def load_tar_index_map(self, data_root, anno_root, cams):
        cam_str = '_'.join(cams)  

        component_groups = cams + [c.replace('cam', 'hand_mask') for c in cams] + [c.replace('cam', 'fg_mask') for c in cams] + ['background']

        if os.path.exists(f'{anno_root}/{self.split}_{cam_str}_tar_index_map.pkl'):
            with open(f'{anno_root}/{self.split}_{cam_str}_tar_index_map.pkl', 'rb') as f:
                self.tar_index_map = pickle.load(f)
        else:
            self.tar_index_map = self.build_tar_index_map(data_root, component_groups)
            with open(f'{anno_root}/{self.split}_{cam_str}_tar_index_map.pkl', 'wb') as f:
                pickle.dump(self.tar_index_map, f)

    def __init__(self, data_root, split, get_camera=False, filter_only_hands=True, no_flip=False, **kwargs) -> None:
        print(f'Loading HandCO Dataset from: {data_root}')
        self.config = kwargs['config']

        if 'cam' in kwargs:
            cam = kwargs['cam']
            self.cams = ['cam%02d' % cam]
        else:
            self.cams = ['cam00']
            cam = 0

        self.data_root = f"{data_root}/tars"
        self.anno_root = data_root

        self.split = split

        assert self.split in ['train', 'val', 'test']

        self.annotations = h5py.File(f'{self.anno_root}/cam{cam}_hand_arm_annotations_v1.h5', "r")[split]
        
        self.load_tar_index_map(self.data_root, self.anno_root, self.cams)

        self.sample_loader = AsyncSamplerLoader(self.data_root)
        
        self.sample_keys = list(self.tar_index_map[self.cams[0]].keys())
 
        if not no_flip:  # no_flip=False keeps original behavior (augment with horizontal flips).
            self.sample_keys += [k + 'flipImage' for k in self.sample_keys]
            filter_path = 'filtered_hand_present_indices.json'
        else:
            filter_path = 'no_flip_filtered_hand_present_indices.json'

        self.background_keys = list(self.tar_index_map['background'].keys())

        with open(f'{self.anno_root}/{filter_path}', 'r') as f:
            self.filtered_indices = list(json.load(f).keys())

        if filter_only_hands:
            self.n_samples = len(self.filtered_indices)
        else:
            self.n_samples = len(self.sample_keys)

        print('Number of samples:', self.n_samples)
        self.get_camera = get_camera
        self.filter_only_hands = filter_only_hands

        self.is_rot6d = self.config.POSE_3D.ROT_6D
        self.rot_dim = 6 if self.is_rot6d else 3

    def __len__(self):
        return self.n_samples
    
    def get_image(self, seq, sample_name, cam_id, flip):
        rgb_key = f'{seq}.{sample_name}.cam{cam_id:02d}.jpg'
        hand_mask_key = f'{seq}.{sample_name}.hand_mask{cam_id:02d}.jpg'
        fg_mask_key = f'{seq}.{sample_name}.fg_mask{cam_id:02d}.jpg'
        background_key = random.choice(self.background_keys)

        rgb_info = self.tar_index_map[f'cam{cam_id:02d}'][rgb_key]        
        mask_info = self.tar_index_map[f'hand_mask{cam_id:02d}'][hand_mask_key]        

        
        if fg_mask_key in self.tar_index_map[f'fg_mask{cam_id:02d}']:
            fg_info = self.tar_index_map[f'fg_mask{cam_id:02d}'][fg_mask_key]        
        else:
            fg_info = None
            
        bg_info = self.tar_index_map['background'][background_key]        

        rgb, mask, fg, bg = self.sample_loader.load_all([
            (rgb_info, 'jpg'), 
            (mask_info, 'jpg'),
            (fg_info, 'jpg'), 
            (bg_info, 'jpg')
        ])

        if fg is not None:
            bg = cv2.resize(bg, rgb.shape[:2])
            tfg = (fg > 100) + (mask > 100) 
            nfg = ~tfg
            rgb = rgb.copy()  

            if fg.sum() > 0:
                rgb[nfg] = bg[nfg]
    
        if flip:
            rgb = cv2.flip(rgb, 1)
    
        return rgb

    def __getitem__(self, index):
        if self.filter_only_hands: index = int(self.filtered_indices[index])

        sample_name = self.sample_keys[index]

        annotation_key = sample_name
        seq, sample_name, component_name, ext = sample_name.split('.')
        cam_id = int(component_name[-2:])

        flip = False
        if 'flipImage' in ext:
            flip = True

        anno_data = self.get_annotation(index)
        anno_key = anno_data['key']        
        if len(anno_key) == 0:  # No annotation for this sample
            ...
        else: # Check if the keys match
            key = annotation_key
            assert key == anno_key, "Key mismatch: %s != %s" % (key, anno_key)

        camera_pose = anno_data["camera_pose"]
        camera_params = anno_data["camera_params"]
        hand_params = anno_data["hand_params"]
        arm_params = anno_data["arm_params"]

        rgb = self.get_image(seq, sample_name, cam_id, flip)
        valid_image_space = np.ones(rgb.shape[:2], dtype=np.uint8)

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
        data['extras']['sequence_name'] = f'{seq}_{flip}'
        data['extras']['valid_image_space'] = valid_image_space
        data['extras']['annotation_key'] = anno_key
        data['extras']['dataset'] = 'handco'
        data['extras']['index'] = index
        
        if self.get_camera:
            focal_length = camera_params['focal_length']
            principal_point = camera_params['principal_point']
            height, width = rgb.shape[:2]
            focal_length = np.array(focal_length)
            principal_point = np.array(principal_point)
 
            projection_params = camera_params['projection_params']
            camera_model = PinholeCameraModel(focal_length, principal_point, width, height)
            
            data['extras']['camera_model'] = camera_model

        return data

    def get_annotation(self, index):
        h5_data = self.annotations[index]
        data   = _row_to_dict(h5_data)

        data["right_hand_camera_j2D"][:, 0] = np.clip(data["right_hand_camera_j2D"][:, 0], 0, self.IMG_WIDTH)
        data["right_hand_camera_j2D"][:, 1] = np.clip(data["right_hand_camera_j2D"][:, 1], 0, self.IMG_HEIGHT)

        data["left_hand_camera_j2D"][:, 0] = np.clip(data["left_hand_camera_j2D"][:, 0], 0, self.IMG_WIDTH)
        data["left_hand_camera_j2D"][:, 1] = np.clip(data["left_hand_camera_j2D"][:, 1], 0, self.IMG_HEIGHT)

        data["left_hand_camera_hand_box"] = np.clip(data["left_hand_camera_hand_box"], 0, [self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_HEIGHT])
        data["right_hand_camera_hand_box"] = np.clip(data["right_hand_camera_hand_box"], 0, [self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_HEIGHT])

        data["right_arm_camera_j2D"][:, 0] = np.clip(data["right_arm_camera_j2D"][:, 0], 0, self.IMG_WIDTH)
        data["right_arm_camera_j2D"][:, 1] = np.clip(data["right_arm_camera_j2D"][:, 1], 0, self.IMG_HEIGHT)

        data["left_arm_camera_j2D"][:, 0] = np.clip(data["left_arm_camera_j2D"][:, 0], 0, self.IMG_WIDTH)
        data["left_arm_camera_j2D"][:, 1] = np.clip(data["left_arm_camera_j2D"][:, 1], 0, self.IMG_HEIGHT)

        data["left_arm_camera_hand_box"] = np.clip(data["left_arm_camera_hand_box"], 0, [self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_HEIGHT])
        data["right_arm_camera_hand_box"] = np.clip(data["right_arm_camera_hand_box"], 0, [self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_HEIGHT])

        if data["fx"] == 0: # Invalid camera
            data["fx"] = data["fy"] = 1 

        padded_dist = np.zeros(15, dtype=np.float32)
        padded_dist[:data["dist"].shape[0]] = data["dist"]

        is_rot_6d = '_rot6d' if self.is_rot6d else ''   


        data["left_hand_j3d_valid"] = (data["left_hand_camera_j3D"] != 0).any() and data["left_hand_valid"]
        data["right_hand_j3d_valid"] = (data["right_hand_camera_j3D"] != 0).any() and data["right_hand_valid"]
        data["left_hand_param_valid"] = data["left_hand_j3d_valid"]
        data["right_hand_param_valid"] = data["right_hand_j3d_valid"]

        data["left_arm_j3d_valid"] = (data["left_arm_camera_j3D"] != 0).any() and data["left_arm_valid"]
        data["right_arm_j3d_valid"] = (data["right_arm_camera_j3D"] != 0).any() and data["right_arm_valid"]

        data["left_arm_param_valid"] = data["left_arm_j3d_valid"]
        data["right_arm_param_valid"] = data["right_arm_j3d_valid"]

        
        return {
            "key": data["key"],  # Decode string
            "camera_params": {
                "focal_length": (data["fx"], data["fy"]), 
                "principal_point": (data["cx"], data["cy"]), 
                "projection_params": padded_dist,
                "width": data["width"], "height": data["height"],
            },
            "camera_pose": data["camera_pose"],

            "hand_params": {
                "left": {
                    "visible": data["left_hand_visible"] and data["left_hand_valid"],
                    "valid_j3d": data["left_hand_j3d_valid"],
                    "valid_param": data["left_hand_param_valid"],
                    
                    "camera_global_orient": data["left_hand_camera_global_orient" + is_rot_6d],
                    "hand_pose": data["left_hand_hand_pose" + is_rot_6d],
                    "camera_betas": data["left_hand_camera_betas"],
                    "camera_transl": data["left_hand_camera_transl"],
                    "camera_j3D": data["left_hand_camera_j3D"],
                    "camera_j2D": data["left_hand_camera_j2D"],
                    "camera_hand_box": data["left_hand_camera_hand_box"],
                },
                "right": {
                    "visible": data["right_hand_visible"] and data["right_hand_valid"],
                    "valid_j3d": data["right_hand_j3d_valid"],
                    "valid_param": data["right_hand_param_valid"],

                    "camera_global_orient": data["right_hand_camera_global_orient" + is_rot_6d],
                    "hand_pose": data["right_hand_hand_pose" + is_rot_6d],
                    "camera_betas": data["right_hand_camera_betas"],
                    "camera_transl": data["right_hand_camera_transl"],
                    "camera_j3D": data["right_hand_camera_j3D"],
                    "camera_j2D": data["right_hand_camera_j2D"],
                    "camera_hand_box": data["right_hand_camera_hand_box"],
                    }
            },
            "arm_params": {
                "left": {
                    "visible": data["left_arm_visible"] and data["left_arm_valid"],
                    "valid_j3d": data["left_arm_j3d_valid"],
                    "valid_param": data["left_arm_param_valid"],

                    "camera_shape": data["left_arm_pca_params"],
                    "camera_R": data["left_arm_camera_R_axis" + is_rot_6d],
                    "camera_T": data["left_arm_camera_T"],
                    "camera_j3D": data["left_arm_camera_j3D"],
                    "camera_j2D": data["left_arm_camera_j2D"],
                    "camera_arm_box": data["left_arm_camera_hand_box"],
                },
                "right": {
                    "visible": data["right_arm_visible"] and data["right_arm_valid"],
                    "valid_j3d": data["right_arm_j3d_valid"],
                    "valid_param": data["right_arm_param_valid"],


                    "camera_shape": data["right_arm_pca_params"],
                    "camera_R": data["right_arm_camera_R_axis" + is_rot_6d],
                    "camera_T": data["right_arm_camera_T"],
                    "camera_j3D": data["right_arm_camera_j3D"],
                    "camera_j2D": data["right_arm_camera_j2D"],
                    "camera_arm_box": data["right_arm_camera_hand_box"],
                }
            }
        }
