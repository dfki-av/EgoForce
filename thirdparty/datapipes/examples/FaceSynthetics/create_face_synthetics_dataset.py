##############################################################################
# Copyright (c) 2022 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, May 2022
##############################################################################
from __future__ import annotations

from argparse import ArgumentParser

from datapipes.dataset_converter import convert_dataset, FileInfo
from datapipes.dataset_converter import encode_ds_info, write_ds_info


class PathToFileInfoConverter:
    def __init__(self):
        pass

    def __call__(self, in_file_path: str) -> FileInfo | None:
        parts = in_file_path.split("/")
        filename = parts[-1]
        if '_' in filename:
            sample_id, component_id = filename[:-4].split('_')
        else:
            sample_id = filename.split('.')[0]
            component_id = "image"
        subset_id = "train"  # FaceSynthetics only has a single split named "train"
        sequence_id = ""     # FaceSynthetics does not have sequences; all samples are independent

        return FileInfo(subset_id, sequence_id, sample_id, component_id)


def convert_face_synthetics(in_path: str, out_path: str, target_tar_file_size: int, splits_to_pre_shuffle: list[str]):
    convert_dataset(in_path, out_path, PathToFileInfoConverter(),
                    target_tar_file_size=target_tar_file_size,
                    splits_to_pre_shuffle=splits_to_pre_shuffle)
    # Optional: create a JSON file with information on the dataset
    ds_info = encode_ds_info(
        short_name="Face Synthetics",
        full_name="Face Synthetics",
        sensors=["RGB"],
        camera_setup="mono",
        nature_of_data="synthetic",
        tasks=["face landmark localization", "face parsing"],
        project_page="https://microsoft.github.io/FaceSynthetics/",
        code_repo="https://github.com/microsoft/FaceSynthetics",
        paper_url="https://openaccess.thecvf.com/content/ICCV2021/html/"
                  "Wood_Fake_It_Till_You_Make_It_Face_Analysis_in_the_ICCV_2021_paper.html",
        license_name="R-UDA",
        converted_by=["Stephan Krauß"])
    write_ds_info(ds_info, out_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in_dir",
                        type=str,
                        default="/ds-av/public_datasets/face_synthetics/original",
                        help="The path to the input directory containing the source dataset.")
    parser.add_argument("--out_dir",
                        type=str,
                        required=True,
                        help="The path to the output directory. The sharded dataset will be stored there.")
    parser.add_argument("--target_tar_file_size",
                        type=int,
                        default=400,
                        help="The target size of the tar archives in MiB. Defaults to 400 MiB.")
    parser.add_argument("--pre_shuffle_splits",
                        type=str,
                        default="train",
                        help="Comma-separated list of the subsets (data splits) that shall be pre-shuffled.")
    args = parser.parse_args()
    # Convert comma-separated string parameters to lists.
    pre_shuffle_splits = args.pre_shuffle_splits.split(",")
    # Run the dataset conversion.
    convert_face_synthetics(args.in_dir, args.out_dir, args.target_tar_file_size, pre_shuffle_splits)
