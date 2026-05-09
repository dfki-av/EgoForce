##############################################################################
# Copyright (c) 2024 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, April 2024
##############################################################################
from __future__ import annotations

import os.path
from argparse import ArgumentParser

from datapipes import dataset_converter
from datapipes.utils import path_utils


def convert_dataset(in_dir: str, out_dir: str, target_tar_file_size: int | None) -> None:
    """
    Converts the files of the UnrealEgo2 dataset into a sharded dataset.

    :param in_dir: The directory that contains the files that belong to the UnrealEgo2 dataset
    :param out_dir: The directory where the sharded dataset will be stored.
    :param target_tar_file_size: The target size of the shards.
    """
    # Store each sample component separately (in a component group of the same name).
    # Sample components not listed here, will not be included in the dataset.
    component_grouping = {"images": ["left_rgb", "right_rgb"],
                          "images_ed_256": ["left_rgb_ed_256", "right_rgb_ed_256"],
                          "images_256": ["left_rgb_256", "right_rgb_256"],
                          "depth": ["left_depth", "right_depth"],
                          "annotations": ["annotations"]}
    # Convert the dataset using the PathToFileInfoConverter class to produce a FileInfo object for each file.
    dataset_converter.convert_dataset(in_dir, out_dir, PathToFileInfoConverter(in_dir), component_grouping,
                                      target_tar_file_size=target_tar_file_size,
                                      splits_to_pre_shuffle=["train", "val"],
                                      path_filter=lambda x: "preprocessed" in x)
    # Optional: Create some meta-data for the website listing the available datasets.
    ds_info = dataset_converter.encode_ds_info(
        short_name="UnrealEgo2",
        full_name="UnrealEgo2",
        sensors=["RGB", "depth"],
        camera_setup="stereo",
        nature_of_data="synthetic",
        tasks=["body pose estimation", "hand pose estimation"],
        project_page="https://4dqv.mpi-inf.mpg.de/UnrealEgo2/",
        code_repo="https://github.com/hiroyasuakada/"
                  "Self-Supervised-Learning-of-Domain-Invariant-Features-for-Depth-Estimation",
        paper_url="https://openaccess.thecvf.com/content/WACV2022/html/Akada_Self-"
                  "Supervised_Learning_of_Domain_Invariant_Features_for_Depth_Estimation_WACV_2022_paper.html",
        license_name="Simplified BSD",
        commercial_use="no",
        converted_by=["Stephan Krauß"]
    )
    # Optional: Store the meta-data along with the dataset.
    dataset_converter.write_ds_info(ds_info, out_dir)


class PathToFileInfoConverter:
    def __init__(self, in_dir: str):
        """ Create a new PathToFileInfoConverter which produces a FileInfo object for each file that shall be included
        in the sharded dataset.

        :param in_dir: The directory that contains the source dataset.
        """
        # maps each subset to a list of paths/sequences that belong to it
        self.subsets = get_subset_mapping(in_dir)
        # maps each (generated) component name to a new one
        self.component_renaming_map = {"fisheye_final_image-camera_left": "left_rgb",
                                       "fisheye_final_image-camera_right": "right_rgb",
                                       "fisheye_depth_image-camera_left": "left_depth",
                                       "fisheye_depth_image-camera_right": "right_depth",
                                       "json": "annotations"}

    def __call__(self, in_file_path: str) -> dataset_converter.FileInfo | None:
        """ Produces a FileInfo object for a given path.

        A FileInfo object provides the following information on each file that shall be included in the dataset:

        - subset: The data split the file belongs to, e.g. "train", "val" or "test"
        - sequence: The sequence the file belongs to (if any, otherwise the empty string ""), e.g. "city_drive_01".
        - sample: The sample the file belongs to, e.g. "frame_0000".
        - component: The sample component stored in this file, e.g. "depth", "mask", "label", "rgb" or "image".

        Additional information like file path and file size are automatically added by the file system scanning code of
        the dataset converter.

        :param in_file_path: The path to the file for which to produce a FileInfo object.
        :return: A FileInfo object if the file shall be included, None otherwise.
        """
        # Skip everything that is not an image or an annotation file, e.g. the txt files containing the data splits.
        outer_path_fragments, inner_path_fragments = path_utils.split_path(in_file_path)
        outer_path_fragments = outer_path_fragments.split(os.path.sep)
        inner_path_fragments = inner_path_fragments.split("/")
        filename = inner_path_fragments[-1] if len(inner_path_fragments) > 0 else outer_path_fragments[-1]
        file_extension = os.path.splitext(filename)[1]
        if file_extension not in [".json", ".mpk", ".jpg", ".png", ".webp"]:
            return None

        # Elements that identify samples are spread out over 3 levels of folders for images and 2 for annotations files.
        sample_dir_depth = 3 if file_extension in [".png", ".webp", ".jpg"] else 2

        # Extract the part of the path from which the sequence name will be constructed.
        sequence_name = "/".join(inner_path_fragments[:-sample_dir_depth])
        # Default to the train set in case the sequence is not assigned to any subset
        subset_id = "train"
        # Look up to which subset the sequence belongs to
        for split, split_paths in self.subsets.items():
            if sequence_name in split_paths:
                subset_id = split
                break
        # Replace slashes to create a valid sequence name
        sequence_name = sequence_name.replace("/", "-")

        # Extract the frame number from the file name to construct the sample name.
        sample_name = f"{int(os.path.splitext(filename)[0].split('_')[1]):04d}"

        # Extract the component identifier. Then shorten it for brevity.
        if sample_dir_depth == 3:
            component_id = f"{inner_path_fragments[-3]}-{inner_path_fragments[-2]}"
        else:
            component_id = inner_path_fragments[-2]
        try:
            component_id = self.component_renaming_map[component_id]
        except KeyError:
            print(f"Invalid component ID {component_id} for file {'/'.join(inner_path_fragments)}")
        # assign equidistant images to their own sample component, so know what we are dealing with later on
        if component_id != "annotations":
            for path_fragment in outer_path_fragments:
                if path_fragment.endswith("_ed"):
                    component_id += "_ed"
                    break
        # assign downscaled images to their own sample component, so we can choose later which one to use
        if outer_path_fragments[-1].endswith("_256.zip"):
            component_id += "_256"

        return dataset_converter.FileInfo(subset_id, sequence_name, sample_name, component_id)


def get_subset_mapping(splits_path: str) -> dict[str, list[str]]:
    """ Creates for each subset (data split) a list of the sequences that belong to it."""
    # map each subset name to the name of the file that list the sequences that shall be included
    split_files = {"train": "train.txt",
                   "val": "validation.txt",
                   "test": "test.txt"}
    # lists the sequences that belong to each subset
    subsets = {}
    for split, split_file in split_files.items():
        current_subset = []
        with open(os.path.join(splits_path, split_file)) as f:
            for line in f.readlines():
                # drop the top level directory and the trailing newline character
                idx = line.index("/")
                current_subset.append(line[idx+1:-1])
        subsets[split] = current_subset
    return subsets


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
    parser.add_argument("--target_tar_file_size",
                        type=int,
                        default=400,
                        help="The target size of the tar shards in MiB.")
    args = parser.parse_args()
    convert_dataset(args.in_dir, args.out_dir, args.target_tar_file_size)
