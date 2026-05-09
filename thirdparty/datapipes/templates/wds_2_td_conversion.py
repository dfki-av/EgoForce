##############################################################################
# Copyright (c) 2024 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, September 2024
##############################################################################
from __future__ import annotations

import os.path
import re
from argparse import ArgumentParser

from datapipes import dataset_converter
from datapipes.versions import api_version, metadata_file_format_version


def decompose_path(path: str) -> tuple[list[str], str, str, str]:
    folders, basename = os.path.split(path)
    folders = folders.split(os.path.pathsep)
    folders, tar_name = folders[:-1], folders[-1]
    match = re.match(r"^((?:.*/|)[^.]+)[.]([^/]*)$", basename)
    if not match:
        raise RuntimeError("Path does not contain a file name and a file extension.")
    sample_name, all_extensions = match.group(1), match.group(2)
    return folders, tar_name, sample_name, all_extensions


def extract_meta_data(in_file_path: str) -> dataset_converter.FileInfo | None:
    """
    Extracts meta-data for a file from its path.

    Returns a FileInfo object with meta-data or None for files that shall not be included in the dataset.
    :param in_file_path: The path to the source file.
    :return: The meta-data for that file as a FileInfo object or None.
    """
    # Chop the path into parts that are easier to work with.
    folders, tar_name, sample_name, all_extensions = decompose_path(in_file_path)

    # The subset (data split) of the dataset is often either the name of a folder or part of the name of the tar file.
    # TODO: extract subset/data split information from folder names or tar file name
    subset_id = ""
    # Webdataset does not support the sequence concept. So just keep it as an empty string.
    sequence_id = ""
    # In webdataset everything up to the first dot is used to identify the sample.
    # This part was already extracted by the helper function.
    sample_id = sample_name
    # In webdataset the sample components are identified by everything after the first dot.
    # Often the part up to the second dot (or end if there is none), is what we are looking for.
    # TODO: You may need to adjust this for your dataset.
    component_id = all_extensions.split(".")[0]

    return dataset_converter.FileInfo(subset_id, sequence_id, sample_id, component_id)

def filter_paths(path: str) -> bool:
    # TODO: [Optional] Exclude directories and archives by returning False for specific dirs or archives.
    return True

def convert_dataset(in_path: str, out_path: str, target_tar_file_size: int, subsets_to_pre_shuffle: list[str]):
    # Display versions.
    print(f"Versions: API: {api_version}, file format: {metadata_file_format_version}")

    # Create a dictionary that maps the component group name to the list of components that shall be included in that
    # component group. To place all components in the same group, simply leave the dictionary empty.
    # TODO: [Optional] Decide which components shall be stored together and which will end up in separate tars.
    component_grouping = {}
    dataset_converter.convert_dataset(in_path, out_path, extract_meta_data, component_grouping, target_tar_file_size,
                                      splits_to_pre_shuffle=subsets_to_pre_shuffle, path_filter=filter_paths)


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
                        help="The target size of the tar archives in MiB. Defaults to 400 MiB.")
    parser.add_argument("--pre_shuffle_subsets",
                        type=str,
                        default="train",
                        help="Comma-separated list of the subsets (data splits) that shall be pre-shuffled.")
    args = parser.parse_args()
    # Convert comma-separated string parameters to lists.
    pre_shuffle_subsets = args.pre_shuffle_subsets.split(",")
    # Run the dataset conversion.
    convert_dataset(args.in_dir, args.out_dir, args.target_tar_file_size, pre_shuffle_subsets)
