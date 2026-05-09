import os
import tarfile
import json
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from mmdet.datasets.base_det_dataset import BaseDetDataset
from mmdet.registry import DATASETS
from datapipes.base_pipeline import BasePipelineCreator

    
def build_tar_index(tar_path):
    index = {}
    with tarfile.open(tar_path, 'r') as tar:
        for tarinfo in tqdm(tar, desc=f"Building tar index"):
            key = f"{tar_path}{os.path.sep}{tarinfo.name}"
            index[key] = (tarinfo.offset_data, tarinfo.size)
    return index


@DATASETS.register_module()
class Arctic(BaseDetDataset):
    METAINFO = {
       'classes': ('left_forearm', 'right_forearm', 'left_hand', 'right_hand'),
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)]
    }

    def load_data_list(self):
        with open(self.ann_file, 'r') as f:
            self.data = json.load(f)    
        anno_keys = list(self.data.keys())

        subset = self.data_prefix['subset'].split(os.path.sep)[-1]
        data_root = self.data_root

        factory = BasePipelineCreator(data_root)
        
        tar_index_map = dict()
        tar_file_paths = factory.get_tar_files_for_subsets(subset, component_groups=["cam00"])

        # for key, tar_paths in tar_file_paths.items():
        #     for tar_path in tar_paths:
        #         tar_index_map.update(build_tar_index(tar_path))
        #         break

        with ThreadPoolExecutor() as executor:
            for key, tar_paths in tar_file_paths.items():
                for tar_index in executor.map(build_tar_index, tar_paths):
                    tar_index_map.update(tar_index)
                    
                    
        data_infos = []
        for file_path, file_offset in tqdm(tar_index_map.items(), desc=f"building data list"):
            path_split = file_path.split(os.path.sep)
            tar_path = f'{os.path.sep}'.join(path_split[:-1])

            sample_name = path_split[-1]
            
            sequence_name, sample_id, view_id = sample_name.split(".")[:3]
            view_id = int(view_id.replace("rgb", ""))
            subject_id = sequence_name.split("_")[0]
            sequence_name = sequence_name.replace(subject_id + "_", "")

            anno_key = f'{subject_id}/{sequence_name}/{view_id}/{sample_id}'

            if anno_key not in anno_keys:
                continue

            data_infos.append(
                dict(
                    img_id=sample_name,
                    file_path=file_path,
                    tar_path=tar_path,
                    rgb_offset=file_offset,
                    anno_key=anno_key
                ))

        return data_infos