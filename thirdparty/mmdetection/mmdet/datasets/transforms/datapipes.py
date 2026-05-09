# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import json
import numpy as np
import pycocotools.mask as maskUtils
import os
import msgpack
import mmap
import lmdb
from mmcv.transforms import LoadImageFromFile

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import get_box_type
from mmdet.structures.bbox.box_type import autocast_box_type
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmdet.datasets.transforms import LoadAnnotations
from datapipes.decoders.image_decoder import ImageDecoder


@TRANSFORMS.register_module()
class LoadImageFromDataPipe(LoadImageFromFile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tarfiles_dict = dict()
        self.image_decoder = ImageDecoder('rgb8') 

    def transform(self, results: dict) -> dict:
        """Transform function to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        file_path = results['file_path']
        tar_path = results['tar_path']        
        offset_data, data_size = results['rgb_offset']

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

        img = self.image_decoder(file_path, file_data)

        if self.to_float32:
            img = img.astype(np.float32)

        results['img_path'] = file_path
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]

        return results


@TRANSFORMS.register_module()
class LoadAnnotationsFromLMDB(LoadAnnotations):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        lmdb_path = self.backend_args['lmdb_path']
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, create=False) 

    def read_arm_anno(self, key):
        with self.env.begin(write=False) as txn:
            value_bytes = txn.get(key.encode('utf-8'))  # Get the value for the given key
            if value_bytes is not None:
                value = msgpack.unpackb(value_bytes)  # Decode MessagePack data
                return value
            else:
                return None  # If the key doesn't exist

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """
        anno_key = results['anno_key']
        annos = self.read_arm_anno(anno_key)
    
        instances = list()
        for anno in annos:
            bbox = anno['bbox']
            bbox_type = anno['type']
            handedness = anno['handedness']
            kp_2d = anno['kp_2d']

            if bbox_type == 'forearm' and handedness == 0:
                bbox_label = 0
            elif bbox_type == 'forearm' and handedness == 1:
                bbox_label = 1
            elif bbox_type == 'hand' and handedness == 0:
                bbox_label = 2
            elif bbox_type == 'hand' and handedness == 1:
                bbox_label = 3
            
            instance = dict()
            instance['bbox'] = bbox
            instance['bbox_label'] = bbox_label
            instance['keypoints'] = kp_2d
            instance['ignore_flag'] = False

            instances.append(instance)

        if not len(instances):
            instances = [
                {
                    'bbox': [0, 0, 0, 0],
                    'bbox_label': 0,
                    'keypoints': [[0, 0, 0]],
                    'ignore_flag': True
                }
            ]

        results['instances'] = instances

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_mask:
            self._load_masks(results)
        if self.with_seg:
            self._load_seg_map(results)
        if self.with_keypoints:
            self._load_kps(results)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'with_keypoints={self.with_keypoints}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str


@TRANSFORMS.register_module()
class LoadAnnotationsFromJSON(LoadAnnotations):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.annos = dict()

        self.json_path = self.backend_args['json_path']
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as f:
                self.annos = json.load(f)
        else:
            print(f'File not found: {self.json_path}')
            

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """
        anno_key = results['anno_key']

        # import os, cv2
        # ROOT_DIR = '/netscratch/millerdurai/Datasets/COCO_KPTS/2017'
        # image_root = f'{ROOT_DIR}/coco_images'
        # image_path = os.path.join(image_root, f'{int(anno_key)}.jpg')
        # image = cv2.imread(image_path)
        # cv2.imwrite('/cmillerd/Projects/Aria/arm_detection/mmdet_test/image.jpg', image)
        # print(self.json_path, anno_key)
        # exit()

        annos = self.annos[anno_key]
    
        instances = list()
        for anno in annos:
            bbox = anno['bbox']
            bbox_type = anno['type']
            handedness = anno['handedness']
            kp_2d = anno['kp_2d']

            if bbox_type == 'forearm' and handedness == 0:
                bbox_label = 0
            elif bbox_type == 'forearm' and handedness == 1:
                bbox_label = 1
            elif bbox_type == 'hand' and handedness == 0:
                bbox_label = 2
            elif bbox_type == 'hand' and handedness == 1:
                bbox_label = 3
            
            instance = dict()
            instance['bbox'] = bbox
            instance['bbox_label'] = bbox_label
            instance['keypoints'] = kp_2d
            instance['ignore_flag'] = False

            instances.append(instance)

        if not len(instances):
            instances = [
                {
                    'bbox': [0, 0, 0, 0],
                    'bbox_label': 0,
                    'keypoints': [[0, 0, 0]],
                    'ignore_flag': True
                }
            ]

        results['instances'] = instances

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_mask:
            self._load_masks(results)
        if self.with_seg:
            self._load_seg_map(results)
        if self.with_keypoints:
            self._load_kps(results)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'with_keypoints={self.with_keypoints}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str


@TRANSFORMS.register_module()
class LoadAnnotationsFromJSONS(LoadAnnotations):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.json_paths = self.backend_args['json_paths']
        
        self.annos = dict()
        for json_path in self.json_paths:
            if os.path.exists(json_path):   
                with open(json_path, 'r') as f:
                    annos = json.load(f)
            else:
                annos = dict()
                        
                for key, value in annos.items():
                    assert key not in self.annos
                    self.annos[key] = value

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """
        anno_key = results['anno_key']

        # import os, cv2
        # ROOT_DIR = '/netscratch/millerdurai/Datasets/COCO_KPTS/2017'
        # image_root = f'{ROOT_DIR}/coco_images'
        # image_path = os.path.join(image_root, f'{int(anno_key)}.jpg')
        # image = cv2.imread(image_path)
        # cv2.imwrite('/cmillerd/Projects/Aria/arm_detection/mmdet_test/image.jpg', image)
        # print(self.json_path, anno_key)
        # exit()

        annos = self.annos[anno_key]
    
        instances = list()
        for anno in annos:
            bbox = anno['bbox']
            bbox_type = anno['type']
            handedness = anno['handedness']
            kp_2d = anno['kp_2d']

            if bbox_type == 'forearm' and handedness == 0:
                bbox_label = 0
            elif bbox_type == 'forearm' and handedness == 1:
                bbox_label = 1
            elif bbox_type == 'hand' and handedness == 0:
                bbox_label = 2
            elif bbox_type == 'hand' and handedness == 1:
                bbox_label = 3
            
            instance = dict()
            instance['bbox'] = bbox
            instance['bbox_label'] = bbox_label
            instance['keypoints'] = kp_2d
            instance['ignore_flag'] = False

            instances.append(instance)

        if not len(instances):
            instances = [
                {
                    'bbox': [0, 0, 0, 0],
                    'bbox_label': 0,
                    'keypoints': [[0, 0, 0]],
                    'ignore_flag': True
                }
            ]

        results['instances'] = instances

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_mask:
            self._load_masks(results)
        if self.with_seg:
            self._load_seg_map(results)
        if self.with_keypoints:
            self._load_kps(results)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'with_keypoints={self.with_keypoints}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str
