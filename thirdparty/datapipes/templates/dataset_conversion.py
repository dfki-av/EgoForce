##############################################################################
# Copyright (c) 2022 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, December 2022
##############################################################################
from __future__ import annotations

import os.path
from argparse import ArgumentParser

from datapipes import dataset_converter
from datapipes.versions import api_version, metadata_file_format_version


class PathToFileInfoConverter:
    def __init__(self, in_path: str):
        # TODO You can gather additional information on the mapping of files to samples, sequences and/or subsets here,
        # e.g. by reading text files that list samples or sequences that belong to a specific subset (data split).
        pass

    def __call__(self, in_file_path: str) -> dataset_converter.FileInfo | None:
        # TODO Skip files that should not go into the shards by returning None, e.g. based on the file extension.
        if os.path.splitext(in_file_path)[1] in []:
            return None

        # TODO extract the information for the fields below from the file path
        subset_id = ""     # An identifier for the subset (data split) the file belongs to (e.g. "train", "val", ...).
        sequence_id = ""   # An identifier for the recording the file belongs to. Compose multiple elements with "_".
        sample_name = ""   # A string that identifies the sample the file belongs to, e.g. a frame number.
        component_id = ""  # An identifier for the sample component contained in the file at hand.

        return dataset_converter.FileInfo(subset_id, sequence_id, sample_name, component_id)


def convert_dataset(in_path: str, out_path: str, target_tar_file_size: int, subsets_to_pre_shuffle: list[str],
                    preserve_sequential_ordering: bool):
    # Display versions
    print(f"Versions: API: {api_version}, file format: {metadata_file_format_version}")
    # TODO Decide which components will be stored together and which will end up in separate tars.
    # Create a dictionary that maps the component group name to the list of components that shall be included in that
    # component group. To place all components in the same group, simply leave the dict empty.
    component_grouping = {}
    dataset_converter.convert_dataset(in_path, out_path, PathToFileInfoConverter(in_path), component_grouping,
                                      target_tar_file_size,
                                      splits_to_pre_shuffle=subsets_to_pre_shuffle,
                                      preserve_sequential_ordering=preserve_sequential_ordering)
    # TODO Fill in information on the dataset that will be displayed on our webpage listing available datasets.
    ds_info = dataset_converter.encode_ds_info(
        short_name="short name of the dataset",
        full_name="full name of the dataset",
        sensors=["sensor types, e.g.:", "RGB", "radar", "lidar"],
        camera_setup="none/mono/stereo/matrix/other",
        nature_of_data="real/synthetic/mixed",
        tasks=["intended tasks, e.g.", "object detection", "semantic segmentation"],
        project_page="https://www.example.com/project",
        code_repo="https://www.example.com/code",
        paper_url="https://www.example.com/paper.pdf",
        license_name="license name, e.g.: GPLv2, MIT, CC-BY-SA, ...",
        commercial_use="yes/no",
        converted_by=["Your Name"])
    dataset_converter.write_ds_info(ds_info, out_path)


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
