##############################################################################
# Copyright (c) 2022 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, December 2022
##############################################################################
from __future__ import annotations

import os.path
from argparse import ArgumentParser

from datapipes.dataset_converter import convert_dataset, FileInfo
from datapipes.dataset_converter import encode_ds_info, write_ds_info
from datapipes.utils.path_utils import split_path


def get_subset_mapping(splits_path: str) -> dict[str, list[str]]:
    """ Creates for each data split (subset) a list of the sequences that belong to it."""
    split_files = {split: os.path.join(splits_path, f"{split}.txt") for split in ["train", "validation", "test"]}
    subsets = dict()

    for split, split_file_path in split_files.items():
        current_subset = []
        with open(split_file_path) as f:
            for line in f.readlines():
                path_fragments = line.split("/")
                current_subset.append("/".join(path_fragments[2:])[:-1])
        subsets[split] = current_subset
    return subsets


class PathToFileInfoConverter:
    def __init__(self, in_path: str):
        # maps each subset to a list of paths/sequences that belong to it
        self.subsets = get_subset_mapping(in_path)
        # maps each (generated) component name to a shorter one
        self.component_renaming_map = {"fisheye_final_image-camera_left": "left_rgb",
                                       "fisheye_final_image-camera_right": "right_rgb",
                                       "fisheye_depth_image-camera_left": "left_depth",
                                       "fisheye_depth_image-camera_right": "right_depth",
                                       "json": "annotations"}

    def __call__(self, in_file_path: str) -> FileInfo | None:
        # Skip the txt files containing the data splits for the train, val and test subsets
        if os.path.splitext(in_file_path)[1] not in [".json", ".png", ".webp", ".jpg"]:
            return None

        # Split the path into a part pointing to an archive (outer_path) and one pointing to a file inside the archive
        # (inner_path).
        outer_path, inner_path = split_path(in_file_path)

        # Split the path inside each archive into fragments (folders and file names).
        path_fragments = inner_path.split("/")

        # Elements that identify samples are spread out over 3 levels of folders for images and 2 for annotations files.
        if inner_path.endswith(".png") or inner_path.endswith(".webp"):
            sample_dir_depth = 3
        else:
            sample_dir_depth = 2

        # Assemble a unique identifier for each sequence from the folder names.
        sequence_name = "/".join(path_fragments[:-sample_dir_depth])

        # Look-up to which subset the sample belongs to
        subset_id = None
        for split, split_paths in self.subsets.items():
            if sequence_name in split_paths:
                subset_id = split
                break
        if subset_id is None:
            # Add left-over sequences to the train set
            subset_id = "train"
        # abbreviate "validation" to "val"
        if subset_id == "validation":
            subset_id = "val"

        # Replace slashes to create valid sequence id
        sequence_name = sequence_name.replace("/", "-")

        # Extract the sample name.
        sample_name = f"{int(os.path.splitext(path_fragments[-1])[0].split('_')[1]):04d}"

        # Extract the component identifier. Then shorten it for brevity.
        if sample_dir_depth == 3:
            component_id = f"{path_fragments[-3]}-{path_fragments[-2]}"
        else:
            component_id = path_fragments[-2]
        try:
            component_id = self.component_renaming_map[component_id]
        except KeyError:
            print("Invalid component ID", component_id, "for sample", path_fragments)
        # store downscaled images in separate component (so we can choose the resolution we want to use)
        if "downscaled" in in_file_path:
            component_id += "_512"

        return FileInfo(subset_id, sequence_name, sample_name, component_id)


def convert_unreal_ego(in_path: str, out_path: str, splits_to_pre_shuffle: list[str]):
    # store full and reduced resolution images separately, so they can be loaded independently
    component_map = {"rgb": ["left_rgb", "right_rgb"],
                     "rgb_512": ["left_rgb_512", "right_rgb_512"],
                     "depth": ["left_depth", "right_depth"],
                     "annotations": ["annotations"]}
    # shard the dataset
    convert_dataset(in_path, out_path, PathToFileInfoConverter(in_path), component_map,
                    splits_to_pre_shuffle=splits_to_pre_shuffle)
    # Optional: encode information on the dataset and write it to disk.
    ds_info = encode_ds_info(
        short_name="UnrealEgo",
        full_name="UnrealEgo: A New Dataset for Robust Egocentric 3D Human Motion Capture",
        sensors=["RGB", "depth"],
        camera_setup="stereo",
        nature_of_data="synthetic",
        tasks=["body pose estimation", "hand pose estimation"],
        project_page="https://4dqv.mpi-inf.mpg.de/UnrealEgo/",
        code_repo="https://github.com/hiroyasuakada/UnrealEgo",
        paper_url="https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4021_ECCV_2022_paper.php",
        license_name="",
        converted_by=["Stephan Krauß"]
    )
    write_ds_info(ds_info, out_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in_dir",
                        type=str,
                        required=True,
                        help="The path to the input directory containing the source dataset.")
    parser.add_argument("--out_dir",
                        type=str,
                        required=True,
                        help="The path to the output directory. The sharded dataset will be stored there.")
    parser.add_argument("--pre_shuffle_splits",
                        type=str,
                        default="train",
                        help="Comma-separated list of the subsets (data splits) that shall be pre-shuffled.")
    args = parser.parse_args()
    # Convert comma-separated string parameters to lists.
    pre_shuffle_subsets = args.pre_shuffle_splits.split(",")
    # Run the dataset conversion.
    convert_unreal_ego(args.in_dir, args.out_dir, pre_shuffle_subsets)
