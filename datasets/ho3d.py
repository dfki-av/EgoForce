import tarfile
import json
import torch
import asyncio
import os
import sys
import numpy as np
import h5py
import pickle
import aiofiles
import cv2
from async_lru import alru_cache

from tqdm import tqdm
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from datapipes.base_pipeline import BasePipelineCreator
from datapipes.decoders.image_decoder import ImageDecoder

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from camera_models import PinholeCameraModel
from utils.rotations import axis_angle_to_rotation_6d_np


def build_tar_index(tar_path):
    index = {}
    with tarfile.open(tar_path, 'r') as tar:
        for idx, tarinfo in enumerate(tqdm(tar, desc=f"Building tar index")):
            index[tarinfo.name] = (tar_path, tarinfo.offset_data, tarinfo.size)
    return index



class AsyncSamplerLoader:
    def __init__(self, data_root):
        self.image_decoder = ImageDecoder('rgb8')
        self.data_root = data_root

    @alru_cache(maxsize=4)
    async def async_load(self, tar_info, file_type):
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
        elif 'pkl' in file_type:
            data = pickle.loads(file_data)
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



class HO3DV2Loader(Dataset):
    IMG_WIDTH = 640
    IMG_HEIGHT = 480

    def build_tar_index_map(self, data_root):
        self.factory = BasePipelineCreator(data_root)

        component_groups = ["anno", "depth", "rgb", "seg",]

        tar_index_map = dict()
        tar_file_paths = self.factory.get_tar_files_for_subsets(self.split, 
                                                                component_groups=component_groups,)

        with ThreadPoolExecutor() as executor:
            for component_name, tar_paths in tar_file_paths.items():
                tar_index_map[component_name] = dict()

                for tar_index in executor.map(build_tar_index, tar_paths):
                    tar_index_map[component_name].update(tar_index)
                
        return tar_index_map

    def load_evaluation_anno(self, anno_root):        
        with open(f'{anno_root}/evaluation.txt', 'r') as f:
            sample_keys = [line.strip().replace('/', '.') + '.rgb.png' for line in f.readlines()]

        eval_xyz = dict()
        with open(f'{anno_root}/evaluation_xyz.json', 'r') as f:
            for idx, xyz in enumerate(json.load(f)):
                eval_xyz[sample_keys[idx]] = np.array(xyz)

        return sample_keys, eval_xyz

    def load_tar_index_map(self, data_root, anno_root):
        if os.path.exists(f'{anno_root}/{self.split}_tar_index_map.pkl'):
            with open(f'{anno_root}/{self.split}_tar_index_map.pkl', 'rb') as f:
                self.tar_index_map = pickle.load(f)
        else:
            self.tar_index_map = self.build_tar_index_map(data_root)
            with open(f'{anno_root}/{self.split}_tar_index_map.pkl', 'wb') as f:
                pickle.dump(self.tar_index_map, f)

    def __init__(self, data_root, split, get_camera=False, filter_only_hands=True, **kwargs) -> None:
        print(f'Loading HO3DV2 Dataset from: {data_root}')
        self.config = kwargs['config']

        self.data_root = f"{data_root}/tars"
        self.anno_root = data_root

        self.split = split

        assert self.split in ['train', 'val', 'test']

        if self.split in ['val', 'test']: 
            self.split = 'evaluation'

        self.load_tar_index_map(self.data_root, self.anno_root)
        self.sample_loader = AsyncSamplerLoader(self.data_root)

        self.sample_keys = list(self.tar_index_map["rgb"].keys())

        if self.split == 'evaluation':
            self.sample_keys, self.eval_xyz = self.load_evaluation_anno(self.anno_root)

        self.n_samples = len(self.sample_keys)

        self.annotations = h5py.File(f'{self.anno_root}/hand_arm_annotations_v1.h5', "r")[self.split]

        print('Number of samples:', self.n_samples)
        self.get_camera = get_camera
        self.filter_only_hands = filter_only_hands

        self.is_ego = False
        self.is_rot6d = self.config.POSE_3D.ROT_6D
        self.rot_dim = 6 if self.is_rot6d else 3

    def __len__(self):
        return self.n_samples
    
    def get_data(self, seq, sample_name):
        rgb_key = f'{seq}.{sample_name}.rgb.png'
        rgb_info = self.tar_index_map[f'rgb'][rgb_key]

        anno_key = f'{seq}.{sample_name}.anno.pkl'
        anno_info = self.tar_index_map[f'anno'][anno_key]

        seg_key = f'{seq}.{sample_name}.seg.jpg'
        seg_info = self.tar_index_map[f'seg'][seg_key]  

        return self.sample_loader.load_all([
            [rgb_info, 'png'],
            [anno_info, 'pkl'],
            [seg_info, 'jpg']
        ])

    def __getitem__(self, index):
        sample_name_full = self.sample_keys[index]

        seq, sample_name, component_name, ext = sample_name_full.split('.')
        anno_id = int(sample_name)
        rgb, anno, seg = self.get_data(seq, sample_name)  
        seg = (seg > 10).astype(np.uint8)
        seg = cv2.resize(seg, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        valid_image_space = np.ones(rgb.shape[:2], dtype=np.uint8)
        right_hand_mask = seg[:, :, 0] 
        object_mask = seg[:, :, -1]

        hand_mask = np.zeros_like(seg)
        # hand_mask[:, :, 0] = left_hand_mask # not available
        hand_mask[:, :, 1] = right_hand_mask






        # def project_points(vertices, K):
        #     proj = (K @ vertices.T).T  # shape (N, 3)
        #     proj_2d = proj[:, :2] / np.expand_dims(proj[:, 2], axis=1)
        #     return proj_2d


        # anno_key = sample_name_full 

        # K = anno['camMat']
        # if self.split == 'train':
        #     handJoints3D = anno['handJoints3D']
        # else:
        #     handJoints3D = self.eval_xyz[anno_key]
        #     handBoundingBox = np.array(anno['handBoundingBox'])

        # j2d = project_points(handJoints3D, K)

        
        # hand_params = {
        #         "left": {
        #             "visible_hand": False,
        #             "camera_global_orient": np.zeros(1),
        #             "hand_pose": np.zeros(1),
        #             "camera_betas": np.zeros(1),
        #             "camera_transl": np.zeros(1),
        #             "camera_j3D": np.zeros(1),
        #             "camera_j2D": np.zeros(1),
        #             "camera_hand_box": np.zeros(1),
        #         },
        #         "right": {
        #             "visible_hand": True,
        #             "camera_global_orient": np.zeros(1),
        #             "hand_pose": np.zeros(1),
        #             "camera_betas": np.zeros(1),
        #             "camera_transl": np.zeros(1),
        #             "camera_j3D": handJoints3D,
        #             "camera_j2D": j2d,
        #             "camera_hand_box": np.zeros(1),
        #             }
        # }


        # data = dict()
        # data['hand_params'] = hand_params
        # data['seg'] = seg
        # data['anno'] = anno
        # data['K'] = K

        # if self.split != 'train':
        #     data['handBoundingBox'] = handBoundingBox
        #     data['handJoints3D'] = handJoints3D
        #     data['handJoints2D'] = j2d

        # # print(rgb.shape)
        # # print(hand_mask.shape)
        # # print(seg.shape)

        # data['extras'] = dict()
        # data['extras']['rgb_np'] = rgb
        # data['extras']['hand_mask_np'] = hand_mask
        # data['extras']['arm_mask_np'] = np.zeros_like(hand_mask)
        # data['extras']['sequence_name'] = seq
        # data['extras']['valid_image_space'] = valid_image_space
        # data['extras']['annotation_key'] = anno_key   


        # data['extras']['dataset'] = 'HO3D'
        # data['extras']['index'] = index

        
        # return data




        anno_data = self.get_annotation(index)
        anno_key = anno_data['key']        
        if len(anno_key) == 0:  # No annotation for this sample
            ...
        else:  # Check if the keys match
            assert sample_name_full == anno_key, "Key mismatch: %s != %s" % (sample_name_full, anno_key)

        data = dict()
        camera_pose = anno_data["camera_pose"]
        camera_params = anno_data["camera_params"]
        hand_params = anno_data["hand_params"]
        arm_params = anno_data["arm_params"]


        # if self.split == 'train':
        #     hand_beta = anno['handBeta']
        #     handpose = anno['handPose']
        #     hand_trans = anno['handTrans']
        #     handJoints3D = anno['handJoints3D']
        
        #     hand_params['right']['visible'] = 1
        #     hand_params['right']['valid_j3d'] = 1
        #     hand_params['right']['valid_param'] = 1

        #     hand_params['right']['camera_betas'] = hand_beta
        #     hand_params['right']['camera_transl'] = hand_trans
            
        #     global_orient = axis_angle_to_rotation_6d_np(handpose[:3]) if self.is_rot6d else handpose[:3]
        #     hand_pose = axis_angle_to_rotation_6d_np(handpose[3:].reshape(15, 3)) if self.is_rot6d else handpose[3:]

        #     hand_params['right']['camera_global_orient'] = global_orient
        #     hand_params['right']['hand_pose'] = hand_pose

        #     hand_params['right']["camera_j3D"][..., 0] *= -1




        data['extras'] = dict()
        data['extras']['rgb_np'] = rgb
        data['extras']['hand_mask_np'] = hand_mask
        data['extras']['arm_mask_np'] = np.zeros_like(hand_mask)
        data['extras']['sequence_name'] = seq
        data['extras']['valid_image_space'] = valid_image_space
        data['extras']['annotation_key'] = anno_key   
        data['extras']['dataset'] = 'HO3D'
        data['extras']['index'] = index

        data['hand_params'] = hand_params
        data['arm_params'] = arm_params
        data['camera_params'] = camera_params
        data['camera_params']['camera_type'] = 0 # ideal pinhole camera
        
        if self.get_camera:
            focal_length = camera_params['focal_length']
            principal_point = camera_params['principal_point']
            height, width = rgb.shape[:2]
            focal_length = np.array(focal_length)
            principal_point = np.array(principal_point)
            camera_model = PinholeCameraModel(focal_length, principal_point, width, height)
            
            data['extras']['camera_model'] = camera_model

        return data

    def get_annotation(self, index):
        data = self.annotations[index]

        if self.split == 'evaluation':
            data["left_hand_camera_hand_box"] = scale_bbox(data["left_hand_camera_hand_box"])
            data["right_hand_camera_hand_box"] = scale_bbox(data["right_hand_camera_hand_box"])

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
        
        return {
            "key": data["key"].decode("utf-8"),  # Decode string
            "camera_params": {
                "focal_length": (data["fx"], data["fy"]), 
                "principal_point": (data["cx"], data["cy"]), 
                "projection_params": padded_dist,
                "width": data["width"], "height": data["height"],
            },
            "camera_pose": data["camera_pose"],
            "hand_params": {
                "left": {
                    "visible": data["left_hand_visible"],
                    "valid_j3d": data["left_hand_j3d_valid"],
                    "valid_param": data["left_hand_param_valid"],
                    "camera_global_orient": data["left_hand_camera_global_orient" + is_rot_6d],
                    "hand_pose": data["left_hand_hand_pose" + is_rot_6d],
                    "camera_betas": data["left_hand_camera_betas"],
                    "camera_transl": data["left_hand_camera_transl"],
                    "camera_j3D": data["left_hand_camera_j3D"],
                    "camera_j2D": data["left_hand_camera_j2D"],
                    "camera_hand_box": data["left_hand_camera_hand_box"],
                    "occluded_jnt": data["left_hand_jnt_occ"],
                },
                "right": {
                    "visible": data["right_hand_visible"],
                    "valid_j3d": data["right_hand_j3d_valid"],
                    "valid_param": data["right_hand_param_valid"],
                    "camera_global_orient": data["right_hand_camera_global_orient" + is_rot_6d],
                    "hand_pose": data["right_hand_hand_pose" + is_rot_6d],
                    "camera_betas": data["right_hand_camera_betas"],
                    "camera_transl": data["right_hand_camera_transl"],
                    "camera_j3D": data["right_hand_camera_j3D"],
                    "camera_j2D": data["right_hand_camera_j2D"],
                    "camera_hand_box": data["right_hand_camera_hand_box"],
                    "occluded_jnt": data["right_hand_jnt_occ"],
                    }
            },
            "arm_params": {
                "left": {
                    "visible": data["left_arm_visible"],
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
                    "visible": data["right_arm_visible"],
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


def scale_bbox(bbox, scale_factor=1.1): 
    # Width and height
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    # Center
    cx = bbox[0] + w / 2
    cy = bbox[1] + h / 2

    # New width and height
    new_w = w * scale_factor
    new_h = h * scale_factor

    # New bbox
    scaled_bbox = np.array([
        cx - new_w / 2,
        cy - new_h / 2,
        cx + new_w / 2,
        cy + new_h / 2
    ], dtype=np.float32)

    return scaled_bbox
