import os
import numpy as np
import cv2
import pickle
import platform
import torch
import tarfile
import json
import sys
import collections
import asyncio
import aiofiles
from async_lru import alru_cache

from torch.utils.data import Dataset
from tqdm import tqdm, trange
from concurrent.futures import ThreadPoolExecutor

if platform.system() == 'Windows':
    PATH_OF_HOT3D_LIB = 'F:/Datasets/HOT3D/hot3d/hot3d'
else:
    PATH_OF_HOT3D_LIB = '/netscratch/millerdurai/Datasets/HOT3D/hot3d/hot3d'

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PATH_OF_HOT3D_LIB)
sys.path.append(ROOT_DIR)

from camera_models import (
    ToPinholeCamera,
    ToStereographicCamera,
    ToEquisolidCamera,
    ToEquirectangularCamera,
    OVR624CameraModel,
    PinholeCameraModel,
)
from utils.camera_utils import construct_world_to_eye_matrix, transform_points
from utils.rotations import axis_angle_to_rotation_6d

from datapipes.base_pipeline import BasePipelineCreator
from datapipes.decoders.image_decoder import ImageDecoder

    
def build_tar_index(tar_path):
    index = {}
    with tarfile.open(tar_path, 'r') as tar:
        for idx, tarinfo in enumerate(tqdm(tar, desc=f"Building tar index")):
        
            index[tarinfo.name] = (tar_path, tarinfo.offset_data, tarinfo.size)

    return index


class AsyncSamplerLoader:
    def __init__(self, *args, **kwargs):
        self.data_root = args[0]
        self.image_decoder = ImageDecoder('rgb8')

    @alru_cache(maxsize=4)
    async def async_load(self, tar_info, file_type):
        tar_path, offset_data, data_size = tar_info

        path_splits = self.data_root.split('/')
        folder_index = tar_path.find(path_splits[-1])
        tar_path = f"{'/'.join(path_splits[:-1])}/{tar_path[folder_index:]}"

        async with aiofiles.open(tar_path, 'rb') as file_obj:
            await file_obj.seek(offset_data)
            file_data = await file_obj.read(data_size)

        if 'png' in file_type or 'jpg' in file_type:
            # Offload image decoding to a thread if it's CPU-bound.
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(None, self.image_decoder, "file_path", file_data)
        elif 'json' in file_type:
            data = json.loads(file_data)
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tarfiles_dict = dict()
        self.image_decoder = ImageDecoder('rgb8') 

    def load(self, tar_info, type):
        tar_path, offset_data, data_size = tar_info

        with open(tar_path, 'rb') as file_obj:
            file_obj.seek(offset_data)
            file_data = file_obj.read(data_size)


        if 'png' in type or 'jpg' in type:
            data = self.image_decoder("file_path", file_data)
        elif 'json' in type:
            data = json.loads(file_data)

        return data



class HOT3DLoader(Dataset):        
    IMG_HEIGHT = 1408
    IMG_WIDTH = 1408

    def load_tar_index_map(self, data_root):
        if os.path.exists(f'{data_root}/{self.split}_tar_index_map.pkl'):
            with open(f'{data_root}/{self.split}_tar_index_map.pkl', 'rb') as f:
                self.tar_index_map = pickle.load(f)
        else:
            factory = BasePipelineCreator(data_root)

            tar_index_map = dict()
            tar_file_paths = factory.get_tar_files_for_subsets(self.split, 
                                                                component_groups=["rgb", 
                                                                                  "annotations",
                                                                                  "segmentation"],)

            with ThreadPoolExecutor() as executor:
                for component_name, tar_paths in tar_file_paths.items():
                    tar_index_map[component_name] = dict()

                    for tar_index in executor.map(build_tar_index, tar_paths):
                        tar_index_map[component_name].update(tar_index)

            self.tar_index_map = tar_index_map
            with open(f'{data_root}/{self.split}_tar_index_map.pkl', 'wb') as f:
                pickle.dump(tar_index_map, f)   

    def __init__(self, data_root, split, get_camera=False, filter_only_hands=True, conversion_mode='none', **kwargs) -> None:
        print(f'Loading HOT3D Dataset from: {data_root}')

        self.data_root = data_root
        self.split = split
        self.config = kwargs['config']

        assert self.split in ['train', 'val', 'test']

        self.load_tar_index_map(data_root)

        print('Loading hand params...')
        self.hand_params = np.load(f'{data_root}/hand_params.npy', allow_pickle=True).item()

        print('Loading hand bboxes...')
        self.hand_bboxes = np.load(f'{data_root}/hand_bboxes.npy', allow_pickle=True).item()

        self.sample_loader = AsyncSamplerLoader(data_root)        
        self.rgb_keys = list(self.tar_index_map['rgb'].keys())
        self.rgb_keys = self.rgb_keys[50:] # start from 50th sample as GT is bad
        
        self.get_camera = get_camera

        self.valid_image_space = cv2.imread(f'{ROOT_DIR}/datasets/hot3d_image_mask.png')
        self.valid_image_space = cv2.cvtColor(self.valid_image_space, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        self.is_rot6d = self.config.POSE_3D.ROT_6D
        self.rot_dim = 6 if self.is_rot6d else 3
        
        with open(f'{self.data_root}/filtered_hand_present_indices.json', 'r') as f:
            self.filtered_indices = list(json.load(f)[split].keys())

        if filter_only_hands:
            self.n_samples = len(self.filtered_indices)
        else:
            self.n_samples = len(self.rgb_keys)
        self.filter_only_hands = filter_only_hands

        print('Number of samples:', self.n_samples)
        
        occ_path = f'{data_root}/occlusion_data_{self.split}.npy'

        if os.path.exists(occ_path):
            print('Loading occlusion data...')
            self.occlusion_data = np.load(occ_path, allow_pickle=True).item()
        else:
            self.occlusion_data = None
    
        arm_annos_path = f'{data_root}/arm_params_{self.split}.npy'

        if os.path.exists(arm_annos_path):
            print('Loading arm annos...')  
            self.arm_annos = np.load(arm_annos_path, allow_pickle=True).item()
        else:
            self.arm_annos = collections.defaultdict(dict)

        if conversion_mode is None:
            conversion_mode = 'none'

        conversion_mode = str(conversion_mode).strip().lower()
        valid_modes = {'none', 'pinhole', 'equisolid', 'equirectangular', 'stereographic'}
        if conversion_mode not in valid_modes:
            raise ValueError(
                f'Invalid HOT3D conversion_mode={conversion_mode}. '
                'Expected one of: none, pinhole, equisolid, equirectangular, stereographic.'
            )

        self.convert_to_pinhole = conversion_mode == 'pinhole'
        self.convert_to_equisolid = conversion_mode == 'equisolid'
        self.convert_to_equirectangular = conversion_mode == 'equirectangular'
        self.convert_to_stereographic = conversion_mode == 'stereographic'

        enabled = [
            self.convert_to_pinhole,
            self.convert_to_equisolid,
            self.convert_to_equirectangular,
            self.convert_to_stereographic,
        ]
        if sum(enabled) > 1:
            raise ValueError(
                "Only one conversion mode can be enabled at once: "
                "CONVERT_TO_PINHOLE / CONVERT_TO_EQUISOLID / "
                "CONVERT_TO_EQUIRECTANGULAR / CONVERT_TO_STEREOGRAPHIC"
            )

        print('HOT3D Conversion mode:', conversion_mode)
        print('HOT3D Convert to pinhole:', self.convert_to_pinhole)
        print('HOT3D Convert to equisolid:', self.convert_to_equisolid)
        print('HOT3D Convert to equirectangular:', self.convert_to_equirectangular)
        print('HOT3D Convert to stereographic:', self.convert_to_stereographic)

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        if self.filter_only_hands: index = int(self.filtered_indices[index])

        tarinfo_name = self.rgb_keys[index]
        
        sequence_name = tarinfo_name.split('.')[0]
        rgb_info = self.tar_index_map['rgb'][tarinfo_name]
        
        hand_mask_key = tarinfo_name.replace('rgb', 'hand_mask').replace('.jpg', '.png')
        hand_mask_info = self.tar_index_map['segmentation'][hand_mask_key]
        annotation_key = tarinfo_name.replace('rgb', 'annotation').replace('.jpg', '.json')
        annotation_info = self.tar_index_map['annotations'][annotation_key]

        rgb, hand_mask, annotation = self.sample_loader.load_all([
            (rgb_info, 'jpg'),
            (hand_mask_info, 'png'),
            (annotation_info, 'json')
        ])

        camera_params = annotation['rgb_camera_params']
        arm_annos = self.arm_annos[annotation_key]
        hand_bboxes = self.hand_bboxes[annotation_key]
        hand_poses = self.hand_params[annotation_key]

        hand_params = dict()
        arm_params = dict()
        for hand_type in ['left', 'right']:
            if hand_type in annotation['rgb_hand_params']:
                hand_param = annotation['rgb_hand_params'][hand_type]
            
                if hand_type in hand_bboxes:
                    hand_bbox = hand_bboxes[hand_type]
                    visible_hand = 1
                else:
                    hand_bbox = [0, 0, 0, 0]
                    visible_hand = 0
                    
                hand_pose = hand_poses[hand_type]
                camera_j2D = np.array(hand_param['camera_j2D'])

                camera_j2D[:, 0] = np.clip(camera_j2D[:, 0], 0, rgb.shape[1] - 1)
                camera_j2D[:, 1] = np.clip(camera_j2D[:, 1], 0, rgb.shape[0] - 1)
                
                if self.is_rot6d:
                    hand_pose = np.array(hand_pose['hand_pose_rot6d'])
                    camera_global_orient = axis_angle_to_rotation_6d(torch.tensor(hand_param['camera_global_orient'], dtype=torch.float32)).numpy()   
                else:
                    hand_pose = np.array(hand_pose['hand_pose_aa'])
                    camera_global_orient = np.array(hand_param['camera_global_orient'])

                hand_param_valid = 1
                hand_j3d_valid = 1

                camera_transl = np.array(hand_param['camera_transl'])
                camera_j3D = np.array(hand_param['camera_j3D'])
                betas = np.array(hand_param['betas'])
            else:
                hand_bbox = [-1, -1, -1, -1]
                visible_hand = 0
                hand_param_valid = 0
                hand_j3d_valid = 0

                hand_pose = np.zeros((15, self.rot_dim))
                camera_global_orient = np.zeros(self.rot_dim)
                camera_transl = np.zeros((3))
                camera_j3D = np.zeros((21, 3))
                camera_j2D = np.zeros((21, 2))
                betas = np.zeros((10))

            occluded_j3D = np.zeros(21)


            if self.occlusion_data is not None:
                annotation_index = int(annotation_key.split('.')[1])
                if sequence_name in self.occlusion_data: 
                    occluded_j3D = self.occlusion_data[sequence_name][annotation_index][hand_type]

            hand_params[hand_type] = {
                'hand_pose': hand_pose,
                'camera_global_orient': camera_global_orient,

                'camera_transl': camera_transl,
                'camera_j3D': camera_j3D,
                'camera_j2D': camera_j2D,
                'camera_betas': betas,
                'camera_hand_box': hand_bbox,
                'occluded_jnt': occluded_j3D,


                'visible': visible_hand,
                'valid_param': hand_param_valid,
                'valid_j3d': hand_j3d_valid,
            }

            arm_anno = arm_annos[hand_type]
            
            if 'bbox' in arm_anno:
                arm_bbox = arm_anno['bbox']
                arm_j2d = arm_anno['keypoints']
                visible_arm = 1
            else:
                arm_bbox = [0, 0, 0, 0]
                arm_j2d = np.zeros((3, 2))
                visible_arm = 0

            arm_param_valid = 0
            arm_j3d_valid = 0

            arm_params[hand_type] = {
                'visible': visible_arm,
                'valid_param': arm_param_valid,
                'valid_j3d': arm_j3d_valid,
                'camera_j2D': arm_j2d,
                'camera_arm_box': arm_bbox,

                'camera_j3D': np.zeros((3, 3)),
                'camera_T': np.zeros((3)),
                'camera_R': np.zeros((self.rot_dim)),
                'camera_shape': np.zeros((5)),
            }

        data = dict()
        data['hand_params'] = hand_params
        data['arm_params'] = arm_params

        data['camera_params'] = camera_params['fisheye624_params']
        data['camera_params']['camera_type'] = 3 # 3 for fisheye624
        data['extras'] = dict()
        data['extras']['rgb_np'] = rgb
        data['extras']['hand_mask_np'] = hand_mask
        data['extras']['sequence_name'] = sequence_name
        data['extras']['valid_image_space'] = self.valid_image_space
        data['extras']['annotation_key'] = annotation_key   
        data['extras']['dataset'] = 'HOT3D'
        data['extras']['index'] = index

        data['extras']['arm_mask_np'] = np.zeros_like(rgb)

        if self.get_camera:
            focal_length = camera_params['fisheye624_params']['focal_length']
            principal_point = camera_params['fisheye624_params']['principal_point']
            projection_params = camera_params['fisheye624_params']['projection_params'][3:]
            height, width = rgb.shape[:2]
            focal_length = np.array(focal_length)
            principal_point = np.array(principal_point)
            projection_params = np.array(projection_params)
            camera_model = OVR624CameraModel(focal_length, principal_point, projection_params, width, height)
        
            data['extras']['camera_model'] = camera_model

        if self.convert_to_pinhole:
            data = convert_to_pinhole_fn(data)
        elif self.convert_to_equisolid:
            data = convert_to_equisolid_fn(data)
        elif self.convert_to_equirectangular:
            data = convert_to_equirectangular_fn(data)
        elif self.convert_to_stereographic:
            data = convert_to_stereographic_fn(data)

        return data


def _convert_data_with_converter(
    data,
    dst_camera,
    dst_focal_length,
    dst_principal_point,
    dst_camera_type,
):
    data['extras']['rgb_np'] = dst_camera(data['extras']['rgb_np'])
    data['extras']['hand_mask_np'] = dst_camera(data['extras']['hand_mask_np'])
    data['extras']['arm_mask_np'] = dst_camera(data['extras']['arm_mask_np'])
    data['extras']['valid_image_space'] = dst_camera(data['extras']['valid_image_space'])

    for hand_type in ['left', 'right']:
        data['hand_params'][hand_type]['camera_hand_box'] = dst_camera.transform_keypoints_2d(np.array(data['hand_params'][hand_type]['camera_hand_box']).reshape(-1, 2)).reshape(-1)
        data['hand_params'][hand_type]['camera_j2D'] = dst_camera.transform_keypoints_2d(np.array(data['hand_params'][hand_type]['camera_j2D']))

        data['arm_params'][hand_type]['camera_arm_box'] = dst_camera.transform_keypoints_2d(np.array(data['arm_params'][hand_type]['camera_arm_box']).reshape(-1, 2)).reshape(-1)
        data['arm_params'][hand_type]['camera_j2D'] = dst_camera.transform_keypoints_2d(np.array(data['arm_params'][hand_type]['camera_j2D']))

    data['camera_params']['camera_type'] = dst_camera_type
    data['camera_params']['focal_length'] = dst_focal_length
    data['camera_params']['principal_point'] = dst_principal_point
    data['camera_params']['projection_params'] = np.array([0, 0, 0, 0], dtype=np.float32)

    # Keep a simple camera model artifact with destination intrinsics for downstream callers.
    data['extras']['camera_model'] = PinholeCameraModel(
        dst_focal_length,
        dst_principal_point,
        1408,
        1408,
    )

    return data


def convert_to_pinhole_fn(data):
    src_camera = data['extras']['camera_model'] 
    dst_focal_length = np.array([480., 480.])
    dst_principal_point = np.array([1408 // 2, 1408 // 2])
    dst_camera = ToPinholeCamera(src_camera, dst_focal_length, dst_principal_point, [1408, 1408])

    return _convert_data_with_converter(
        data,
        dst_camera,
        dst_focal_length,
        dst_principal_point,
        dst_camera_type=0,
    )


def convert_to_equisolid_fn(data):
    src_camera = data['extras']['camera_model']
    dst_focal_length = np.array([480.0, 480.0], dtype=np.float32)
    dst_principal_point = np.array([1408 // 2, 1408 // 2], dtype=np.float32)
    dst_camera = ToEquisolidCamera(src_camera, dst_focal_length, dst_principal_point, [1408, 1408])

    return _convert_data_with_converter(
        data,
        dst_camera,
        dst_focal_length,
        dst_principal_point,
        dst_camera_type=5,
    )


def convert_to_equirectangular_fn(data):
    src_camera = data['extras']['camera_model']
    width = 1408
    height = 1408
    dst_focal_length = np.array([width / (2.0 * np.pi), height / np.pi], dtype=np.float32)
    dst_principal_point = np.array([width // 2, height // 2], dtype=np.float32)
    dst_camera = ToEquirectangularCamera(src_camera, dst_focal_length, dst_principal_point, [width, height])

    return _convert_data_with_converter(
        data,
        dst_camera,
        dst_focal_length,
        dst_principal_point,
        dst_camera_type=6,
    )


def convert_to_stereographic_fn(data):
    src_camera = data['extras']['camera_model']
    dst_focal_length = np.array([480.0, 480.0], dtype=np.float32)
    dst_principal_point = np.array([1408 // 2, 1408 // 2], dtype=np.float32)
    dst_camera = ToStereographicCamera(src_camera, dst_focal_length, dst_principal_point, [1408, 1408])

    return _convert_data_with_converter(
        data,
        dst_camera,
        dst_focal_length,
        dst_principal_point,
        dst_camera_type=7,
    )


def get_j2d_from_params(hand_params, camera_params, image):
    focal_length = camera_params['fisheye624_params']['focal_length']
    principal_point = camera_params['fisheye624_params']['principal_point']
    projection_params = camera_params['fisheye624_params']['projection_params'][3:]

    focal_length = np.array(focal_length)
    principal_point = np.array(principal_point)
    projection_params = np.array(projection_params)

    height, width = image.shape[:2] 
    camera_model = OVR624CameraModel(focal_length, principal_point, projection_params, width, height)

    w2c = camera_params["w2c"]
    
    j2Ds = []
    for hand_type, hand_param in hand_params.items():
        world_j3D = np.array(hand_param["world_j3D"])
        camera_j3D = transform_points(w2c, world_j3D)

        j2D = camera_model.camera_to_uv(camera_j3D)
        j2D = j2D.astype(np.int32)
        j2Ds.append(j2D)

    j2D = np.concatenate(j2Ds, axis=0)
    
    return j2D


