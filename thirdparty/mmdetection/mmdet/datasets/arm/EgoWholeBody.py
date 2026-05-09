import mmengine
import os
import lmdb
import msgpack
import tarfile

from tqdm import tqdm
from mmdet.datasets.base_det_dataset import BaseDetDataset
from mmdet.registry import DATASETS
from datapipes.base_pipeline import BasePipelineCreator

    
def build_tar_index(tar_path):
    index = {}
    with tarfile.open(tar_path, 'r') as tar:
        for tarinfo in tqdm(tar, desc=f"Building tar index"):
            key = f"{tar_path}{os.path.sep}{tarinfo.name}"
            index[key] = tarinfo.offset_data
    return index


@DATASETS.register_module()
class EgoWholeBody(BaseDetDataset):
    METAINFO = {
       'classes': ('left_forearm', 'right_forearm', 'left_hand', 'right_hand'),
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)]
    }

    def load_data_list(self):
        subset = self.data_prefix['subset'].split(os.path.sep)[-1]
        data_root = self.data_root

        factory = BasePipelineCreator(data_root)
        multiplier = 2
        shard_size = factory.get_average_shard_sample_count(subset)
        num_workers = 8

        tar_index_map = dict()
        tar_file_paths = factory.get_tar_files_for_subsets(subset, component_groups=["images", "annotations"])
        for key, tar_paths in tar_file_paths.items():
            for tar_path in tar_paths:
                tar_index_map.update(build_tar_index(tar_path))

        data_infos = []
        for file_path, file_offset in tar_index_map.items():
            if 'rgb' not in file_path: continue

            path_split = file_path.split(os.path.sep)
            tar_path = f'{os.path.sep}'.join(path_split[:-1])
            sample_name = path_split[-1]
            identity_name, sequence_id = sample_name.split("#")
            sequence_name, sample_id = sequence_id.split(".")[:2]
            sequence_name = sequence_name.replace("$", ".")

            anno_key = f"{identity_name}.{sequence_name}.{sample_id}"

            data_infos.append(
                dict(
                    img_id=sample_name,
                    file_path=file_path,
                    tar_path=tar_path,
                    rgb_offset=file_offset,
                    anno_key=anno_key
                ))

        return data_infos