##############################################################################
# Copyright (c) 2023 DFKI GmbH - All Rights Reserved
# Written by Michael Fürst <Michael.Fürst@dfki.de>, June 2023
##############################################################################
from __future__ import annotations

import os.path
from argparse import ArgumentParser

from datapipes import dataset_converter
from datapipes.versions import api_version, metadata_file_format_version


class PathToFileInfoConverter:
    def __init__(self, in_path: str):
        with open(os.path.join(in_path, "train.txt")) as f:
            self.train_samples = [x.strip("\n") for x in f.readlines()]
        with open(os.path.join(in_path, "val.txt")) as f:
            self.val_samples = [x.strip("\n") for x in f.readlines()]

    def __call__(self, in_file_path: str) -> dataset_converter.FileInfo | None:
        if "devkit" in in_file_path:
            return None
        folder, file = os.path.split(in_file_path)
        if folder == "" and file.endswith(".txt"):
            return None

        path_parts = in_file_path.split("/")
        sequence_name = ""
        sample_name = path_parts[3].split(".")[0]
        component_id = path_parts[2]
        if path_parts[1] == "training":
            if sample_name in self.train_samples:
                subset_id = "train"
            elif sample_name in self.val_samples:
                subset_id = "val"
            else:
                raise ValueError("Encounter sample not assigned to any split")
        else:
            assert path_parts[1] == "testing"
            subset_id = "test"

        return dataset_converter.FileInfo(subset_id, sequence_name, sample_name, component_id)


def convert_dataset(in_path: str, out_path: str, target_tar_file_size: int, subsets_to_pre_shuffle: list[str],
                    preserve_sequential_ordering: bool):
    print(f"API version: {api_version}")
    print(f"File format version: {metadata_file_format_version}")
    # Gather information on each (relevant) file in the source dataset.
    file_infos = dataset_converter.gather_fileinfos(in_path, PathToFileInfoConverter(in_path), verbose=True)
    # Create a dictionary that maps the component group name to the list of components that shall be included in that
    # component group.
    component_map = {
        "image_2": ["image_2"],
        "image_3": ["image_3"],
        "annotation": ["calib", "label_2"],
        "lidar": ["velodyne"],
    }
    # Build the dataset structure based on gathered data and component map.
    dataset_structure = dataset_converter.build_dataset_structure(file_infos, component_map)
    # Try to automatically determine optimal shard sizes.
    shard_sizes = dataset_converter.suggest_shard_size(dataset_structure, target_tar_file_size,
                                                       preserve_sequential_ordering=preserve_sequential_ordering)
    # Disable pre-shuffling if we need to keep the sequential ordering intact.
    subsets_to_pre_shuffle = [] if preserve_sequential_ordering else subsets_to_pre_shuffle
    # Enable preserving of sequence boundaries if we need to keep the sequential ordering intact.
    preserve_sequence_boundaries_for_subsets = list(dataset_structure.keys()) if preserve_sequential_ordering else []
    # Restructure the dataset by (randomly) assigning samples and their components to shards.
    dataset_structure = dataset_converter.assign_files_to_shards(
        dataset_structure,
        shard_sizes,
        global_shuffle_for_subsets=subsets_to_pre_shuffle,
        preserve_sequence_boundaries_for_subsets=preserve_sequence_boundaries_for_subsets
    )
    # Write the meta-data to disk.
    metadata = dataset_converter.write_metadata(dataset_structure, out_path,
                                                subsets_to_pre_shuffle, preserve_sequence_boundaries_for_subsets)
    # Write the dataset shards to disk.
    dataset_converter.write_shards(metadata, dataset_structure, in_path, out_path)
    # Write information on the dataset to disk.
    ds_info = dataset_converter.encode_ds_info(
        short_name="kitti_object",
        full_name="Kitti Vision Benchmark - Object Detection",
        sensors=["RGB", "lidar"],
        camera_setup="stereo",
        nature_of_data="real",
        tasks=["object detection", "3d object detection", "bev object detection"],
        project_page="https://www.cvlibs.net/datasets/kitti/eval_3dobject.php",
        code_repo="https://www.cvlibs.net/datasets/kitti/eval_3dobject.php",
        paper_url="https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Geiger2012CVPR.pdf",
        license_name="academic use only",
        converted_by=["Michael Fürst", "Stephan Krauß"])
    dataset_converter.write_ds_info(ds_info, out_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in_dir",
                        type=str,
                        required=False,
                        default="/ds-av/public_datasets/kitti_object/original",
                        help="The path to the input directory containing the source dataset.")
    parser.add_argument("--out_dir",
                        type=str,
                        required=False,
                        default="/netscratch/fuerst/Datasets/kitti_object",
                        help="The path to the output directory. The sharded dataset will be stored there.")
    parser.add_argument("--target_tar_file_size",
                        type=int,
                        default=400,
                        help="The target size of the tar archives in MiB. Defaults to 400 MiB.")
    parser.add_argument("--pre_shuffle_subsets",
                        type=str,
                        default="train",
                        help="Comma-separated list of the subsets (data splits) that shall be pre-shuffled.")
    parser.add_argument("--preserve_sequential_ordering",
                        type=bool,
                        default=False,
                        help="Indicated whether the data in this dataset is required to keep its sequence structure and"
                             "ordering within the sequences. If enabled all samples of a sequence will be kept within"
                             " the same shard and pre-shuffling will be disabled.")
    args = parser.parse_args()
    # Convert comma-separated string parameters to lists.
    pre_shuffle_subsets = args.pre_shuffle_subsets.split(",")
    # Run the dataset conversion.
    convert_dataset(args.in_dir, args.out_dir, args.target_tar_file_size,
                    pre_shuffle_subsets, args.preserve_sequential_ordering)
