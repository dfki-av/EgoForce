import os
import numpy as np
import cv2
import platform
import torch
import tarfile
import json
import sys
import pickle
import asyncio
import aiofiles
import re
import h5py
from async_lru import alru_cache

from torch.utils.data import Dataset
from tqdm import tqdm, trange
from concurrent.futures import ThreadPoolExecutor

if platform.system() == 'Windows':
    PATH_OF_HOT3D_LIB = 'F:/Datasets/HOT3D/hot3d/hot3d'
else:
    PATH_OF_HOT3D_LIB = '/netscratch/millerdurai/Datasets/HOT3D/hot3d/hot3d'

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PATH_OF_HOT3D_LIB)
sys.path.append(ROOT_DIR)

from datapipes.base_pipeline import BasePipelineCreator
from datapipes.decoders.image_decoder import ImageDecoder
from contextlib import ExitStack
from camera_models import PinholeCameraModel


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


def build_tar_index(data_root, get_depth, cams, samples):
    shards_path = os.path.join(data_root, 'shards')
    shards = os.listdir(shards_path)

    annotation_tars_list = [tar for tar in shards if 'annotations.tar' in tar]
    rgb_tars_list = [tar for tar in shards if 'rgb.tar' in tar]
    depth_tars_list = [tar for tar in shards if 'depth.tar' in tar]

    samples = set([re.sub(r'cam\d+/', '', sam).replace('/rgb', '').replace('/', '_').replace('.png', '') for sam in samples])
    assert len(annotation_tars_list) == len(rgb_tars_list) == len(depth_tars_list)

    annotation_tars_dict = dict()
    rgb_tars_dict = dict()
    depth_tars_dict = dict()
    for cam in ['cam0', 'cam1', 'cam2', 'cam3', 'cam4']:
        annotation_tars_dict[cam] = sorted([tar for tar in annotation_tars_list if cam in tar])
        rgb_tars_dict[cam] = sorted([tar for tar in rgb_tars_list if cam in tar])
        depth_tars_dict[cam] = sorted([tar for tar in depth_tars_list if cam in tar])   

    def worker_fn(tar_index, cam):
        index = dict()
        with ExitStack() as stack:
            # Open all tar files
            rgb_tar = stack.enter_context(tarfile.open(os.path.join(shards_path, rgb_tars_dict[cam][tar_index]), 'r'))
            annotation_tar = stack.enter_context(tarfile.open(os.path.join(shards_path, annotation_tars_dict[cam][tar_index]), 'r'))
            annotation_tar = iter(annotation_tar)

            if get_depth:
                depth_tar = stack.enter_context(tarfile.open(os.path.join(shards_path, depth_tars_dict[cam][tar_index]), 'r'))
                depth_tar = iter(depth_tar)

            for rgbinfo in tqdm(rgb_tar, desc=f"Building tar index for {tar_index}/{len(rgb_tars_dict[cam])}"):
                cam_info = next(annotation_tar)
                if 'verb_label.txt' in cam_info.name:
                    cam_info = next(annotation_tar)
                if 'action_label' in cam_info.name:
                    cam_info = next(annotation_tar)
                
                hand_pose_info = next(annotation_tar)
                hand_pose_mano_info = next(annotation_tar)
                obj_pose_info = next(annotation_tar)
                obj_pose_rt_info = next(annotation_tar)

                assert (rgbinfo.name.replace('.rgb.png', '') == cam_info.name.replace('.cam_pose.txt', '') or 
                        rgbinfo.name.replace('.rgb.png', '') == hand_pose_info.name.replace('.hand_pose.txt', '') or 
                        rgbinfo.name.replace('.rgb.png', '') == hand_pose_mano_info.name.replace('.hand_pose_mano.txt', '') or
                        rgbinfo.name.replace('.rgb.png', '') == obj_pose_info.name.replace('.obj_pose.txt', '') or
                        rgbinfo.name.replace('.rgb.png', '') == obj_pose_rt_info.name.replace('.obj_pose_rt.txt', ''))

                if get_depth:
                    depthinfo = next(depth_tar)
                    assert rgbinfo.name.replace('.rgb.png', '') == depthinfo.name.replace('.depth.png', '')
                
                sample_name = re.sub(r'cam\d+_', '', rgbinfo.name.replace('.rgb.png', ''))

                if sample_name not in samples:
                    continue

                index[sample_name] = dict()
                index[sample_name]['rgb'] = (rgb_tars_dict[cam][tar_index], rgbinfo.offset_data, rgbinfo.size)

                index[sample_name]['cam_pose'] = (annotation_tars_dict[cam][tar_index], cam_info.offset_data, cam_info.size)
                index[sample_name]['hand_pose'] = (annotation_tars_dict[cam][tar_index], hand_pose_info.offset_data, hand_pose_info.size)
                index[sample_name]['hand_pose_mano'] = (annotation_tars_dict[cam][tar_index], hand_pose_mano_info.offset_data, hand_pose_mano_info.size)
                index[sample_name]['obj_pose'] = (annotation_tars_dict[cam][tar_index], obj_pose_info.offset_data, obj_pose_info.size)
                index[sample_name]['obj_pose_rt'] = (annotation_tars_dict[cam][tar_index], obj_pose_rt_info.offset_data, obj_pose_rt_info.size)

                if get_depth:
                    index[sample_name]['depth'] = (depth_tars_dict[cam][tar_index], depthinfo.offset_data, depthinfo.size)
                        
        return index

    cam_indices = dict()
    for cam in cams:
        cam_indices[cam] = dict()
        with ThreadPoolExecutor() as executor:
            results = executor.map(lambda tar_index: worker_fn(tar_index, cam), range(len(rgb_tars_dict[cam])))
        for index in results:
            cam_indices[cam].update(index)
            
    return cam_indices


class AsyncSamplerLoader:
    def __init__(self, data_root, anno_root):
        self.image_decoder = ImageDecoder('rgb8')
        self.data_root = data_root

    @alru_cache(maxsize=4)
    async def async_load(self, tar_info, file_type):
        tar_path, offset_data, data_size = tar_info
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
        tasks = [self.async_load(tar_info, file_type) for tar_info, file_type in files]
        results = await asyncio.gather(*tasks)
        return results

    def load_all(self, files):
        return asyncio.run(self.async_load_all(files))

    def load(self, tar_info, file_type):
        # Simply wrap the async call for a single file.
        return asyncio.run(self.async_load(tar_info, file_type))


class SamplerLoader:
    # cam4 is egocentric, 
    def __init__(self, data_root, anno_root):

        self.data_root = data_root
        self.tarfiles_dict = dict()
        self.image_decoder = ImageDecoder('rgb8') 

    def load(self, tar_info, type):
        tar_path, offset_data, data_size = tar_info
        tar_path = os.path.join(self.data_root, 'shards', tar_path)

        if tar_path not in self.tarfiles_dict:            
            file_obj = open(tar_path, 'rb')
            tar_size = os.path.getsize(tar_path)
            # mmap_obj = mmap.mmap(file_obj.fileno(), tar_size, access=mmap.ACCESS_READ)

            self.tarfiles_dict[tar_path] = {
                                            'file_obj': file_obj, 
                                            # 'mmap_obj': mmap_obj
                                            }
 
        tar_file_data = self.tarfiles_dict[tar_path]

        # file_data = tar_file_data['mmap_obj'][offset_data:offset_data + data_size]

        tar_file_data['file_obj'].seek(offset_data)
        file_data = tar_file_data['file_obj'].read(data_size)

        if 'png' in type or 'jpg' in type:
            data = self.image_decoder("file_path", file_data)
        elif 'json' in type:
            data = json.loads(file_data)
        elif 'txt' in type:
            data = file_data.decode('utf-8').split('\n')

        return data


def project_3d_to_2d(camera_params, kpt3d):
    fx, fy = camera_params['focal_length']
    cx, cy = camera_params['principal_point']
    
    # Extract X, Y, Z
    X, Y, Z = kpt3d[:, 0], kpt3d[:, 1], kpt3d[:, 2]
    
    # Avoid division by zero
    Z = np.where(Z == 0, 1e-6, Z)

    # Compute 2D projections
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy

    # Stack into (N, 2) array
    kpt2d = np.stack((u, v), axis=-1)

    return kpt2d


class H2OLoader(Dataset):
    IMG_WIDTH = 1280
    IMG_HEIGHT = 720
    def load_tar_index_map(self, data_root, anno_root, depth, cams):
        if os.path.exists(f'{anno_root}/{self.split}_tar_index_map.pkl'):
            with open(f'{anno_root}/{self.split}_tar_index_map.pkl', 'rb') as f:
                self.tar_index_map = pickle.load(f)
        else:
            self.tar_index_map = build_tar_index(data_root, depth, cams, self.samples)
            with open(f'{anno_root}/{self.split}_tar_index_map.pkl', 'wb') as f:
                pickle.dump(self.tar_index_map, f)

    def __init__(self, data_root, split, get_camera=False, **kwargs) -> None:
        self.data_root = data_root + '/wds'
        self.anno_root = data_root
        self.config = kwargs['config']

        if 'cam' in kwargs:
            cams = ['cam' + str(kwargs['cam'])]
        else:
            cams = ['cam4']

        self.split = split
        
        assert self.split in ['train', 'val', 'test']

        split_path = os.path.join(self.anno_root, 'splits', f'pose_{split}.txt')
        with open(split_path, 'r') as f:
            self.samples = [line.strip() for line in f.readlines()]

        depth = False
        
        self.load_tar_index_map(self.data_root, self.anno_root, depth, cams)

        self.sample_loader = AsyncSamplerLoader(self.data_root, self.anno_root)
    
        self.sample_keys = list(self.tar_index_map[cams[0]].keys())
        self.n_samples = len(self.sample_keys)

        self.cam = cams[0]

        self.annotations = h5py.File(f'{self.anno_root}/{self.cam}_annos.h5', "r")[split]

        try:
            self.hand_masks = h5py.File(f'{self.anno_root}/{self.cam}_hand_masks.h5', "r")[split]
        except Exception as e:
            print(e)
            self.hand_masks = None

        print(f'H2O - {split} - Number of samples: {self.n_samples}')

        self.load_cameras(self.anno_root)
        self.valid_image_space = np.ones((self.IMG_HEIGHT, self.IMG_WIDTH), dtype=np.uint8)
        self.get_camera = get_camera
        self.is_rot6d = self.config.POSE_3D.ROT_6D
        self.rot_dim = 6 if self.is_rot6d else 3

    def load_cameras(self, anno_root):  
        cam_dir = os.path.join(anno_root, 'camera_calibration')

        self.camera_params = dict()
        for cam_name in os.listdir(cam_dir):
            cam_path = os.path.join(cam_dir, cam_name)
            with open(cam_path, 'r') as f:
                cam_params = f.read().strip()
                
                cam_name = cam_name.replace('.intrinsics.txt', '')
                fx, fy, cx, cy, width, height = cam_params.split()

                self.camera_params[cam_name] = {
                    'focal_length': (float(fx), float(fy)),
                    'principal_point': (float(cx), float(cy)),
                    'projection_params': (0, ),
                    'width': int(width),
                    'height': int(height)
                }

    def __len__(self):
        return self.n_samples
    
    def get_only_sample_name_and_cam(self, index):
        sample_name = self.sample_keys[index]
    
        cam = self.cam
        session_name = '_'.join(sample_name.split('_')[:-1])
                
        camera_param = self.camera_params[f'{session_name}_{cam}']
        return sample_name, camera_param

    def __getitem__(self, index):
        sample_name = self.sample_keys[index]
    
        cam = self.cam
        session_name = '_'.join(sample_name.split('_')[:-1])
                
        item = self.tar_index_map[cam][sample_name]
       
        camera_param = self.camera_params[f'{session_name}_{cam}']

        rgb_info = item['rgb']
        
        rgb = self.sample_loader.load(rgb_info, 'png')
        anno_data = self.get_annotation(index)
        anno_key = anno_data['key']        
        if len(anno_key) == 0:  # No annotation for this sample
            ...
        else: # Check if the keys match
            assert sample_name == anno_key, "Key mismatch: %s != %s" % (sample_name, anno_key)

        camera_pose = anno_data["camera_pose"]
        camera_params = anno_data["camera_params"]
        hand_params = anno_data["hand_params"]
        arm_params = anno_data["arm_params"]

        if self.hand_masks is None:
            hand_mask = np.zeros_like(rgb)
        else:
            hand_mask = self.hand_masks[index][:, :, ::-1]
    
        arm_mask = np.zeros_like(rgb)

        data = dict()
        data['hand_params'] = hand_params
        data['arm_params'] = arm_params
        data['camera_params'] = camera_params
        data['camera_params']['camera_type'] = 0 # ideal pinhole camera
        data['extras'] = dict()
        data['extras']['rgb_np'] = rgb
        data['extras']['hand_mask_np'] = hand_mask
        data['extras']['arm_mask_np'] = arm_mask
        data['extras']['sequence_name'] = session_name
        data['extras']['valid_image_space'] = self.valid_image_space
        data['extras']['annotation_key'] = sample_name   
        data['extras']['dataset'] = 'H2O'
        data['extras']['index'] = index

        if self.get_camera:
            focal_length = camera_param['focal_length']
            principal_point = camera_param['principal_point']
            height, width = rgb.shape[:2]
            focal_length = np.array(focal_length)
            principal_point = np.array(principal_point)
            camera_model = PinholeCameraModel(focal_length, principal_point, width, height)

            # focal_length = np.sqrt(width**2 + height**2), np.sqrt(width**2 + height**2)
            # principal_point = (width / 2, height / 2)
            # camera_model = PinholeCameraModel(focal_length, principal_point, width, height)

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

        is_rot_6d = '_rot6d' if self.is_rot6d else ''   

        if data["fx"] == 0: # Invalid camera
            data["fx"] = data["fy"] = 1 

        if np.isnan(data["left_arm_camera_j3D"]).any() or np.isinf(data["left_arm_camera_j3D"]).any():
            data["left_arm_visible_hand"] = 0
            data["left_arm_pca_params"] = np.zeros_like(data["left_arm_pca_params"])
            data["left_arm_camera_R_axis" + is_rot_6d] = np.zeros_like(data["left_arm_camera_R_axis" + is_rot_6d])
            data["left_arm_camera_T"] = np.zeros_like(data["left_arm_camera_T"])
            data["left_arm_camera_j2D"] = np.zeros_like(data["left_arm_camera_j2D"])
            data["left_arm_camera_j3D"] = np.zeros_like(data["left_arm_camera_j3D"])
            data["left_arm_camera_hand_box"] = np.zeros_like(data["left_arm_camera_hand_box"])

        if np.isnan(data["right_arm_camera_j3D"]).any() or np.isinf(data["right_arm_camera_j3D"]).any():
            data["right_arm_visible_hand"] = 0
            data["right_arm_pca_params"] = np.zeros_like(data["right_arm_pca_params"])
            data["right_arm_camera_R_axis" + is_rot_6d] = np.zeros_like(data["right_arm_camera_R_axis" + is_rot_6d])
            data["right_arm_camera_T"] = np.zeros_like(data["right_arm_camera_T"])
            data["right_arm_camera_j2D"] = np.zeros_like(data["right_arm_camera_j2D"])
            data["right_arm_camera_j3D"] = np.zeros_like(data["right_arm_camera_j3D"])
            data["right_arm_camera_hand_box"] = np.zeros_like(data["right_arm_camera_hand_box"])

        data["left_hand_j3d_valid"] = (data["left_hand_camera_j3D"] != 0).any()
        data["right_hand_j3d_valid"] = (data["right_hand_camera_j3D"] != 0).any()
        data["left_hand_param_valid"] = data["left_hand_j3d_valid"]
        data["right_hand_param_valid"] = data["right_hand_j3d_valid"]

        data["left_arm_j3d_valid"] = (data["left_arm_camera_j3D"] != 0).any()
        data["right_arm_j3d_valid"] = (data["right_arm_camera_j3D"] != 0).any() 

        data["left_arm_param_valid"] = data["left_arm_j3d_valid"]
        data["right_arm_param_valid"] = data["right_arm_j3d_valid"]


        return {
            "key": data["key"],  # Decode string
            "camera_params": {
                "focal_length": (data["fx"], data["fy"]), 
                "principal_point": (data["cx"], data["cy"]), 
                "projection_params": np.zeros(15), 
                "width": data["width"], "height": data["height"],
            },
            "camera_pose": data["camera_pose"],
            "hand_params": {
                "left": {
                    "visible": data["left_hand_visible_hand"],
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
                    "visible": data["right_hand_visible_hand"],
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
                    "visible": data["left_arm_visible_hand"],
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
                    "visible": data["right_arm_visible_hand"],
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
