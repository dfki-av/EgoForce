##############################################################################
# Copyright (c) 2022 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, October 2022
# and David Michael Fürst <david_michael.fuerst@dfki.de>, October 2022
##############################################################################
"""doc
# datapipes.dataset_converter: Dataset Conversion

> Convert datasets into shards with metadata.
"""

from __future__ import annotations
import math
import sys
from typing import Callable, Optional

import json
import os
import random
import tarfile
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass

import msgpack
from tqdm import tqdm

from datapipes.versions import api_version, metadata_file_format_version, filetree_file_format_version
from datapipes.utils.path_utils import split_path as _split_path
import datapipes.utils.term_colors as tc


"""doc
## Simplified Conversion

Use convert_samples() to convert one or more sets of samples to sharded datasets with a single function call.

It does not work with sequential datasets and unlike the full conversion process, it cannot be customized.
Furthermore, the function will not scan the filesystem for you. Therefore, it is up to you to come up with the list
of samples for each subset (or data split).
Alternatively, you can use the convert_dataset() function (see below). It will scan the filesystem for you,
but you need to provide a function that extracts some information from the paths of the discovered files.
"""


def _scan_samples(in_dir: str, samples: dict[str, list[dict[str, str]]]) -> list[FileInfo]:
    tar_archives: dict[str, tarfile.TarFile] = {}
    zip_archives: dict[str, zipfile.ZipFile] = {}
    files = []
    for split, samples in samples.items():
        index = 0
        for sample in samples:
            for name, path in sample.items():
                if not path.startswith(in_dir):
                    path = os.path.join(in_dir, path)
                if ".zip" in path:
                    archive_path, inner_path = _split_path(path)
                    if archive_path not in zip_archives:
                        zip_archives[archive_path] = zipfile.ZipFile(archive_path)
                    archive = zip_archives[archive_path]
                    file_size = archive.getinfo(inner_path).file_size
                elif ".tar" in path:
                    archive_path, inner_path = _split_path(path)
                    if archive_path not in tar_archives:
                        tar_archives[archive_path] = tarfile.TarFile(archive_path)
                    archive = tar_archives[archive_path]
                    info = archive.getmember(inner_path)
                    file_size = info.size
                else:
                    file_size = os.stat(path).st_size
                file_path = os.path.relpath(path, in_dir)
                file = FileInfo(split, "", f"{index:05d}", name, file_path, file_size)
                files.append(file)
            index += 1
    return files


def convert_samples(in_dir: str,
                    out_dir: str,
                    samples: dict[str, list[dict[str, str]]],
                    target_shard_size_mb: int = 400,
                    splits_to_pre_shuffle: list[str] | None = None) -> None:
    """ Converts one or more sets of samples into a sharded dataset.

    The only required parameters are in_dir, out_dir and samples. See parameter description below to learn how to
    specify the samples. Unless you read the data from a local SSD it is recommended that the source dataset is stored
    in ZIP (or TAR) archives.

    This function makes a few assumptions:
    - You do not care about the order of the samples within the sequences (if any).
    - You do not care about the names of the files in the resulting shards.
    - You do not wish to store some of the sample components separately.
    - You are willing to write your own code to gather a list of samples for each data split in exchange for not
      having to learn how to use the code that does it for you.

    If any of those assumptions is not met, please use the full conversion pipeline.

    Parameters:
    :param in_dir: The path to the directory that contains the dataset in its current form. The paths of the files in
        the dataset will be retained relative to this path.
    :param out_dir: The path to the directory the dataset shall be written to.
    :param samples: The samples of the dataset that shall be sharded, structured as follows:
        A dictionary mapping the names of the data splits (or subsets), e.g. "train", "val", "test" to a list of samples
        that shall be included in each split. Each sample in the list is a dictionary mapping the name of the component
        to the file which contains the data for it, e.g.
        ```
        {"image": "images.zip/seq/image/frame_0000.jpg",
         "depth": "images.zip/seq/depth/frame_0000.png",
         "label": "labels.zip/seq/label_0000.txt"}
        ```
        The path may point to a file inside a zip or tar archive, but archives may not be nested.
        The path may be absolute or relative to `in_dir`.
    :param target_shard_size_mb: The targeted size of each shard. If unsure use the default, which is 400 MB.
    :param splits_to_pre_shuffle: Which splits shall be pre-shuffled. By default, all are. Typically, you want the
        "train" split to be shuffled, while shuffling is not necessary for "test". For "val" it depends on whether you
        would like to train with a mix of "train" + "val" at some point.
    """
    print(f"datapipes versions: API: {api_version}, file format: {metadata_file_format_version}")
    print(f"Creating sharded dataset from files in {in_dir}.")
    # Gather information on each (relevant) file in the source dataset.
    file_infos = _scan_samples(in_dir, samples)
    if splits_to_pre_shuffle is None:
        splits_to_pre_shuffle = list(samples.keys())
    component_map = {}
    # Build the dataset structure based on gathered data and component map.
    dataset_structure = build_dataset_structure(file_infos, component_map)
    # Try to automatically determine optimal shard sizes.
    shard_sizes = suggest_shard_size(dataset_structure, target_shard_size_mb)
    # Restructure the dataset by (randomly) assigning samples and their components to shards.
    dataset_structure = assign_files_to_shards(dataset_structure, shard_sizes,
                                               global_shuffle_for_subsets=splits_to_pre_shuffle)
    # Write the meta-data to disk.
    metadata = write_metadata(dataset_structure, out_dir, splits_to_pre_shuffle)
    # Write the dataset shards to disk.
    write_shards(metadata, dataset_structure, out_dir)


"""doc

## Full Conversion Pipeline 

This supports more use-cases and allows for customization of the conversion process.
Furthermore, you do not have to implement scanning the filesystem and/or archives for samples yourself.

Conversion of a dataset is done in three steps, which will be detailed in later sections.
Additionally, a dataset info file can be created to provide more information our webpage listing available datasets.

1. Gather information
2. Shard the dataset
3. Write metadata and shards
4. Optional: Write dataset info file

You can run steps 1-3 together by calling the convert_dataset() function.
If you would like to customize the conversion process further, you can also call them separately.
Please find an example of how to run the steps individually below.
A template for this is also available in the "templates" directory.
```python
# How are components grouped?
component_map = {} # <- implement or leave empty for default grouping

# 1. Gather Info
def convert_path_to_fileinfo(filepath: str) -> Optional[FileInfo]:
    return None # <- implement

file_infos = gather_fileinfos(in_dir, convert_path_to_fileinfo)
ds_structure = build_dataset_structure(file_infos, component_map)
shard_sizes = suggest_shard_size(structure)

# 2. Shard the dataset
shard_structure = assign_files_to_shards(ds_structure, shard_sizes, ...)

# 3. Write metadata and shards
metadata = write_metadata(shard_structure, out_dir, subsets_to_pre_shuffle)
write_shards(metadata, shard_structure, in_dir, out_dir)

# 4. Write dataset info file
ds_info = encode_ds_info(...)
write_ds_info(ds_info, out_dir)
```
"""


def convert_dataset(in_dir: str,
                    out_dir: str,
                    path_to_file_info_converter: Callable[[str], FileInfo | None],
                    component_grouping: dict[str, list[str]] | None = None,
                    target_tar_file_size: int = 400,
                    target_gpu_count: int = 8,
                    splits_to_pre_shuffle: list[str] | None = None,
                    path_filter: Callable[[str], bool] = lambda x: True,
                    preserve_sequential_ordering: bool = False) -> None:
    """ Creates a sharded version of a dataset.

    Further customization is possible by copying the code of this function or by using the "dataset_conversion"
    template in the "templates" folder. It also contains a template for the implementation of the
    path_to_file_info_converter callable.

    :param in_dir: The directory that contains the source dataset.
    :param out_dir: The directory the sharded dataset will be written to.
    :param path_to_file_info_converter: A function or callable class that creates a FileInfo object from a file path.
    :param component_grouping: A dictionary which indicates which sample components shall be stored together.
     By default, all sample components are stored together.
    :param target_tar_file_size: The average size of the resulting shards. Defaults to 400 MiB.
    :param target_gpu_count: The maximum number of GPUs that will be supported by the sharded dataset. Defaults to 8.
    :param splits_to_pre_shuffle: Which subsets (or data splits) shall be pre-shuffled. Defaults to all.
    :param path_filter: A function that returns a boolean to indicate whether the given path (dir or archive) should be
        recursed into or not.
    :param preserve_sequential_ordering: Whether to create a sequence dataset, where the order of samples within each
        sequence is preserved. Defaults to False.
    """
    print(f"datapipes versions: API: {api_version}, file format: {metadata_file_format_version}")

    if component_grouping is None:
        component_grouping = {}

    file_infos = gather_fileinfos(in_dir, path_to_file_info_converter, verbose=True, path_filter=path_filter)
    dataset_structure = build_dataset_structure(file_infos, component_grouping)
    shard_sizes = suggest_shard_size(dataset_structure, target_tar_file_size, target_gpu_count,
                                     preserve_sequential_ordering=preserve_sequential_ordering)

    if preserve_sequential_ordering:
        splits_to_pre_shuffle = []
        preserve_sequence_boundaries_for_subsets = list(dataset_structure.keys())
    else:
        if splits_to_pre_shuffle is None:
            splits_to_pre_shuffle = list(dataset_structure.keys())
        preserve_sequence_boundaries_for_subsets = []

    dataset_structure = assign_files_to_shards(
        dataset_structure,
        shard_sizes,
        global_shuffle_for_subsets=splits_to_pre_shuffle,
        preserve_sequence_boundaries_for_subsets=preserve_sequence_boundaries_for_subsets
    )
    metadata = write_metadata(dataset_structure, out_dir,
                              splits_to_pre_shuffle, preserve_sequence_boundaries_for_subsets)
    write_shards(metadata, dataset_structure, out_dir)


"""doc
## **Step 1**: Gather information on the dataset

Use these functions to build a representation of your dataset suited for sharding.

The goal is to generate a nested dictionary representing the dataset structure as this hierarchy:
 dict[subset][sequence][sample][component_group][component] = filepath.
For example, it could look like this:

```json
{
    "train": {
        "sequence0000": {
            "sample_f21102022": {
                "camera1": {"camera1": "my_data/images/left/000001.png"},
                "camera2": {"camera2": "my_data/images/right/000001.png"},
                "annotations": {"box": "my_data/annotations/box/000001.txt"}
            },
            "sample_a1337f42": {
                "camera1": {"camera1": "my_data/images/left/000002.png"},
                "camera2": {"camera2": "my_data/images/right/000002.png"},
                "annotations": {"box": "my_data/annotations/box/000002.txt"}
            },
            ...
        },
        "sequence0001": {...},
        ...
    },
    "val": {...},
    "test": {...}
}
```
"""

class DatasetInfo:
    def __init__(self, 
                 in_dir: str,
                 path_to_file_info_converter: Callable[[str], FileInfo | None],
                 path_filter: Callable[[str], bool] = lambda x: True):
        self.in_dir = in_dir
        self.path_to_file_info_converter = path_to_file_info_converter
        self.path_filter = path_filter
    

def convert_multiple_datasets(dataset_infos: list[DatasetInfo],
                              out_dir: str,
                              component_grouping: dict[str, list[str]] | None = None,
                              target_tar_file_size: int = 400,
                              target_gpu_count: int = 8,
                              splits_to_pre_shuffle: list[str] | None = None,
                              preserve_sequential_ordering: bool = False) -> None:
    """ Creates a sharded version of muliple datasets.

    :param dataset_infos: A list of DatasetInfo objects that contain information on the datasets to be converted.
    :param out_dir: The directory the sharded dataset will be written to.
    :param component_grouping: A dictionary which indicates which sample components shall be stored together.
     By default, all sample components are stored together.
    :param target_tar_file_size: The average size of the resulting shards. Defaults to 400 MiB.
    :param target_gpu_count: The maximum number of GPUs that will be supported by the sharded dataset. Defaults to 8.
    :param splits_to_pre_shuffle: Which subsets (or data splits) shall be pre-shuffled. Defaults to all.
    :param preserve_sequential_ordering: Whether to create a sequence dataset, where the order of samples within each
        sequence is preserved. Defaults to False.
    """
    print(f"datapipes versions: API: {api_version}, file format: {metadata_file_format_version}")

    if component_grouping is None:
        component_grouping = {}

    file_infos = []
    for dataset_info in dataset_infos:
        file_infos += gather_fileinfos(dataset_info.in_dir, dataset_info.path_to_file_info_converter, verbose=True, path_filter=dataset_info.path_filter)
    
    dataset_structure = build_dataset_structure(file_infos, component_grouping)
    shard_sizes = suggest_shard_size(dataset_structure, target_tar_file_size, target_gpu_count,
                                     preserve_sequential_ordering=preserve_sequential_ordering)

    if preserve_sequential_ordering:
        splits_to_pre_shuffle = []
        preserve_sequence_boundaries_for_subsets = list(dataset_structure.keys())
    else:
        if splits_to_pre_shuffle is None:
            splits_to_pre_shuffle = list(dataset_structure.keys())
        preserve_sequence_boundaries_for_subsets = []

    dataset_structure = assign_files_to_shards(
        dataset_structure,
        shard_sizes,
        global_shuffle_for_subsets=splits_to_pre_shuffle,
        preserve_sequence_boundaries_for_subsets=preserve_sequence_boundaries_for_subsets
    )
    metadata = write_metadata(dataset_structure, out_dir,
                              splits_to_pre_shuffle, preserve_sequence_boundaries_for_subsets)
    write_shards(metadata, dataset_structure, out_dir)


if sys.version_info < (3, 9):
    from typing import Dict, Tuple
    SubSetMetaDataStructure = Dict[str, Dict[str, Dict[str, Dict[str, Tuple[str, int]]]]]
    MetaDataStructure = Dict[str, SubSetMetaDataStructure]
else:
    SubSetMetaDataStructure = dict[str, dict[str, dict[str, dict[str, tuple[str, int]]]]]
    MetaDataStructure = dict[str, SubSetMetaDataStructure]


@dataclass
class FileInfo:
    """
    Information about a file.

    The subset and sequence (if any) that the sample belongs.
    The name/identifier of the sample.
    A string identifying the sample component contained in the file.
    The "sequence_name", "sample_name" and "component_id" must not contain any dots (".").
    """
    subset: str
    sequence_name: str
    sample_name: str
    component_id: str
    file_path: str = ""
    file_size: int = 0
    in_dir: str = ""

    def __post_init__(self):
        if self.subset == "__meta__":
            raise ValueError("'__meta__' is not allowed as a subset name")
        if "." in self.sequence_name:
            raise ValueError("'.' is not allowed in the sequence name")
        if "." in self.sample_name:
            raise ValueError("'.' is not allowed in the sample name")
        if "." in self.component_id:
            raise ValueError("'.' is not allowed in the component id")


def gather_fileinfos(in_dir: str,
                     convert_path_to_fileinfo: Callable[[str], Optional[FileInfo]],
                     verbose: bool = False,
                     path_filter: Callable[[str], bool] = lambda x: True) -> list[FileInfo]:
    """
    Find all relevant files on disk and meta information about them by walking over a directory and its subdirectories.

    This function will also recurse into zip archives and uncompressed tar archives. It is, therefore, not necessary to
    extract datasets that are stored in such archives.
    :param in_dir: Path to folder to traverse for searching files.
    :param convert_path_to_fileinfo: A function that maps filepaths to relevant information on a file.
        The function receives a path and returns a FileInfo object.
    :param verbose: Whether to print information on the progress of the directory traversal.
    :param path_filter: A filter that specifies whether a given directory or archive shall be processed (True) or
        skipped (False).

    Example for a convert_path_to_fileinfo function.
    ```python
    def _convert_path_to_fileinfo(in_file_path: str) -> Optional[FileInfo]:
        if ignore_file:
            return None

        subset_id = ""      # typically one of "train", "val", "test" or the name of the dataset if no subsets exist
        sequence_name = ""  # e.g "seq01" or "" (if no sequences exist or there is only one)
        sample_name = ""    # e.g. "frame0001"
        component_id = ""   # e.g. "camera1", "image", "class", "mask", ...

        return FileInfo(subset_id, sequence_name, sample_name, component_id)
    ```
    """
    file_infos = []
    in_dir = os.path.normpath(in_dir)

    def _process_file(file_path: str, file_size: int) -> None:
        file_path = os.path.relpath(file_path, in_dir)
        info = convert_path_to_fileinfo(file_path)
        if info is not None:
            # Automatically set relative path to the file.
            if info.file_path == "":
                info.file_path = file_path
            info.in_dir = in_dir
            info.file_size = file_size
            # store file info object in list
            file_infos.append(info)

    # Traverse the specified directory and its subdirectories.
    for root_dir, dir_names, file_names in os.walk(in_dir):
        if path_filter(root_dir) or root_dir == in_dir:
            if verbose:
                print("Processing", root_dir)
        else:
            if verbose:
                tc.print_c(f"Skipping {root_dir}", tc.Color.BLUE)
            continue
        for file_name in file_names:
            src_path = os.path.join(root_dir, file_name)
            extension = os.path.splitext(file_name)[1]
            if extension in [".tar", ".zip"]:
                if path_filter(src_path):
                    if verbose:
                        print("Processing", src_path)
                else:
                    if verbose:
                        tc.print_c(f"Skipping {src_path}", tc.Color.BLUE)
                    continue
                if extension == ".tar":
                    with tarfile.open(src_path, "r") as in_file:
                        for tar_info in in_file:
                            if tar_info.isfile():
                                _process_file(os.path.join(src_path, tar_info.name), tar_info.size)
                elif extension == ".zip":
                    with zipfile.ZipFile(src_path, "r") as in_file:
                        for zip_info in in_file.infolist():
                            if not zip_info.filename.endswith("/"):
                                _process_file(os.path.join(src_path, zip_info.filename), zip_info.file_size)
            else:
                _process_file(src_path, os.path.getsize(src_path))

    return file_infos


def _find_component_group(component_map: dict[str, list[str]], component_name: str) -> str:
    """
    Find to which of the component groups in the component map a component belongs.

    :param component_map: A dictionary mapping component groups to a list of components in it.
    :param component_name: A name for a component that should be found.
    :return: The name of the component group the given component belongs to.
    """
    for component_group, components in component_map.items():
        if component_name in components:
            return component_group
    # If no assignment of components to component groups is given, then put everything in the same component group.
    if len(component_map) == 0:
        return ""  # Return a single, unnamed component group.
    raise ValueError(f"Component {component_name} not found in component map")


def build_dataset_structure(file_infos: list[FileInfo],
                            component_map: dict[str, list[str]],
                            find_component_group=_find_component_group) -> MetaDataStructure:
    """
    Given a list of file_infos build a structure for the dataset.

    :param file_infos: A list of FileInfo objects as created by gather_fileinfos().
    :param component_map: A dictionary mapping component groups to a list of components in it.
        For example with images in separate groups:
         `{"camera1": ["camera1"], "camera2": ["camera2"], "annotations": ["label1", "label2"])`.
        Or without groups: `{"": ["camera1", "camera2", "label1", "label2"]}`.
    :param find_component_group: (Optional) A function that finds the component name in the component map.
     (Default: _find_component_group)
    :return: A nested dictionary representing the dataset structure as this hierarchy:
     dict[subset][sequence][sample][component_group][component] = filepath.
    """
    dataset_structure: MetaDataStructure = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for info in file_infos:
        component_group = find_component_group(component_map, info.component_id)
        sample_id = info.sample_name
        if info.sequence_name != "":
            sample_id = f"{info.sequence_name}.{sample_id}"
        dataset_structure[info.subset][info.sequence_name][sample_id][component_group][
            info.component_id] = (info.in_dir, info.file_path, info.file_size)
    return dataset_structure


def suggest_shard_size(dataset_structure: MetaDataStructure,
                       target_tar_size_mb: int = 400,
                       target_gpu_count: int = 8,
                       small_ds_size_threshold_gb: int = 16,
                       max_shards: int = 2080,
                       preserve_sequential_ordering: bool = False) -> dict[str, int]:
    """ Suggests a suitable shard size for each subset.

    First, it will check whether the dataset is small. If that is the case, it will place all files in one shard for
    each subset and component group.
    For large datasets it will try to determine the optimal shard size for each subset. If the component groups have
    different sizes (in terms of bytes) it tries to adjust the target TAR size such that the shard file size of the
    smallest component group is above min_tar_size and the shard file size of the largest component group is below
    max_tar_size. Furthermore, it ensures that the number of shards is divisible by target_gpu_count.

    :param dataset_structure: Information on the structure of the dataset.
    :param target_tar_size_mb: The target size of the TAR shards in MiB. Defaults to 400 MiB.
    :param target_gpu_count: The number of GPU that shall be supported for parallel training and testing. Defaults to 8.
    :param small_ds_size_threshold_gb: The size threshold in GiB below which datasets shall be considered small and
     hence be placed in a single tar file. Defaults to 16 GiB. Reasonable range: 8 GiB - 26 GiB.
    :param max_shards: The maximum number of shards to generate. Defaults to 2080.
    :param preserve_sequential_ordering: Whether the ordering of samples within each sequence shall be preserved.
    :return: The suggested shard sizes (in number of samples) for each subset.
    """
    # Determine the size of subsets in bytes and number of samples.
    subset_file_sizes = defaultdict(lambda: defaultdict(lambda: 0))
    sample_count = defaultdict(lambda: 0)
    subset_file_count = dict()
    for subset, sequences in dataset_structure.items():
        current_file_count = 0
        for sequence, samples in sequences.items():
            for sample, component_groups in samples.items():
                sample_count[subset] += 1
                for component_group, components in component_groups.items():
                    for component, (_, file_path, file_size) in components.items():
                        subset_file_sizes[subset][component_group] += file_size
                        current_file_count += 1
        subset_file_count[subset] = current_file_count
    tc.print_c("\nDataset and Subset Sizes", tc.Format.BOLD, tc.Format.UNDERLINE)
    total_size = 0
    for subset, component_groups in subset_file_sizes.items():
        for component_group, size in component_groups.items():
            total_size += size
    print(f"total size: {total_size / (1024 * 1024 * 1024):.1f} GiB")

    for subset, file_sizes in subset_file_sizes.items():
        tc.print_c(f"\nsubset \"{subset}\":", tc.Format.BOLD)
        print("  sample count:", sample_count[subset])
        for cg, cg_size in sorted(file_sizes.items()):
            if cg_size > 1024 * 1024 * 1024:
                if cg == "":
                    print(f"  size: {cg_size / (1024 * 1024 * 1024):.1f} GiB")
                else:
                    print(f"  size of '{cg}': {cg_size / (1024 * 1024 * 1024):.1f} GiB")
            else:
                if cg == "":
                    print(f"  size: {cg_size / (1024 * 1024):.1f} MiB")
                else:
                    print(f"  size of '{cg}': {cg_size / (1024 * 1024):.1f} MiB")

    if total_size <= small_ds_size_threshold_gb * 1024 * 1024 * 1024:
        print("\nThe dataset is small. Will create one TAR file per subset and component group.")
        return {subset: sample_count[subset] for subset in subset_file_sizes.keys()}

    # Sequences longer than 2080 cannot be shuffled perfectly anymore due to the cycle length of the RNG.
    shard_limit = (max_shards + target_gpu_count - 1) // target_gpu_count * target_gpu_count

    tc.print_c("\nPerformance Stats", tc.Format.BOLD, tc.Format.UNDERLINE)
    shard_size_per_subset = defaultdict(lambda: 0)
    for subset, component_groups in subset_file_sizes.items():
        tc.print_c(f"\nsubset \"{subset}\":", tc.Format.BOLD)
        max_cg_size = max(component_groups.values())
        min_cg_size = min(component_groups.values())

        # determine the number of shards needed to get to the targeted average file size of each shard
        target_shards = ((max_cg_size // (target_tar_size_mb * 1024 * 1024)) + target_gpu_count - 1) // target_gpu_count
        target_shards = max(target_shards * target_gpu_count, target_gpu_count)

        if target_shards > shard_limit:
            tc.print_c(f"  Selected shard size of {target_tar_size_mb} MiB would result\n"
                       f"  in {target_shards} shards. Limiting to {shard_limit} shards.", tc.Color.YELLOW)
            target_shards = shard_limit

        cg_sizes = [size / (target_shards * 1024 * 1024) for _, size in sorted(component_groups.items())]
        print("  Average estimated TAR file size(s) per component group: ", end="")
        for i in range(len(cg_sizes)-1):
            print(f"{cg_sizes[i]:.1f} MiB, ", end=" ")
        print(f"{cg_sizes[-1]:.1f} MiB")

        if min_cg_size / target_shards < 10 * 1024 * 1024:
            tc.print_c("  Warning: The size of the TAR files of the smallest component group will on\n"
                       "  average be below 10 MiB. Please reconsider the assignment of components to\n"
                       "  component groups and/or the targeted TAR file size.\n", tc.Color.YELLOW)

        # round up when dividing the number of samples by the number of shards that we want to get
        shard_size_per_subset[subset] = (sample_count[subset] + target_shards - 1) // target_shards

        # Print some additional statistics if the data needs to be stored in a way that preserves the sequence
        # boundaries.
        if preserve_sequential_ordering:
            min_sequence_len = 0
            max_sequence_len = 0
            sequences = dataset_structure[subset]
            for sequence, samples in sequences.items():
                max_sequence_len = max(max_sequence_len, len(samples))
                min_sequence_len = min(min_sequence_len, len(samples)) if min_sequence_len > 0 else len(samples)
            print(f"  Length of shortest sequence: {min_sequence_len}")
            print(f"  Length of longest sequence: {max_sequence_len}")
            print(f"  Number of sequences: {len(sequences)}")
            print(f"  Target shard length: {shard_size_per_subset[subset]}\n")
            if max_sequence_len > shard_size_per_subset[subset]:
                tc.print_c("  WARNING: Longest sequence exceeds target shard length! "
                           "Consider increasing the target TAR size.\n", tc.Color.YELLOW)
        # HDD properties
        hdd_latency = 0.015  # 15 ms
        hdd_read_rate = 200 * 1024 * 1024  # 200 MB/s
        # estimate reading efficiency
        subset_size = sum(component_groups.values())
        tar_file_count = target_shards * len(component_groups)
        wait_time = tar_file_count * hdd_latency
        read_time = subset_size / hdd_read_rate
        efficiency = 100 * read_time / (read_time + wait_time)
        original_wait_time = subset_file_count[subset] * hdd_latency
        original_efficiency = 100 * read_time / (read_time + original_wait_time)
        speedup = (read_time + original_wait_time) / (read_time + wait_time)
        print(f"  Estimated reading efficiency before/after conversion: {original_efficiency:.2f} % / "
              f"{efficiency:.2f} %\n  Estimated dataset reading speed-up: {speedup:.1f}")

    return shard_size_per_subset


"""doc
## **Step 2**: Shard the dataset

Use these functions to split your dataset into shards.
A shard is a chunk of data that is later stored in a single file on disk (a single shard should be between 100-400 MiB).
Thus, we want to ensure, that samples end up in the same shards.

Goal is to generate a nested dictionary representing the sharded dataset as this hierarchy:
 `dict[subset][shard_id][sample_id][component_group][component] = filepath`.
Note that `sample_id` is `sequence_name.sample_name`, if there is no sequences just use the `sample_name`.
This could look like this:

```json
{
    "train": {
        "0000": {
            "sequence0000.sample_f21102022": {
                "camera1": {"camera1": "my_data/images/left/000001.png"},
                "camera2": {"camera2": "my_data/images/right/000001.png"},
                "annotations": {"box": "my_data/annotations/box/000001.txt"}
            },
            "sequence0000.sample_a1337f42": {
                "camera1": {"camera1": "my_data/images/left/000002.png"},
                "camera2": {"camera2": "my_data/images/right/000002.png"},
                "annotations": {"box": "my_data/annotations/box/000002.txt"}
            },
            ...
        },
        "0001": {...},
        ...
    },
    "val": {...},
    "test": {...}
}
```
"""


def _assign_files_to_shards_preserving_sequence_boundaries(
        sharded_dataset_structure: MetaDataStructure,
        subset_name: str,
        subset_data: SubSetMetaDataStructure,
        max_samples_per_shard: int) -> None:
    """ Assigns the samples to shards in a way that preserves the sequence boundaries.

    Sequences (if they exist) may be broken up into multiple shards or multiple sequences may be placed in the same
    shard, depending on the length of the sequences and how many samples shall go in one shard (at most).
    """
    def _add_chunk(name: str, sequence: dict):
        if len(chunks) < target_chunk_count:
            # add new chunk
            shard_sequence_names.append(name)
            chunks.append(sequence)
        else:
            # extend smallest chunk
            sizes = [len(x) for x in chunks]
            smallest_chunk_idx = sizes.index(min(sizes))
            chunks[smallest_chunk_idx].update(sequence)

    def _compute_avail_size():
        if len(chunks) < target_chunk_count:
            return max_samples_per_shard
        else:
            sizes = [len(x) for x in chunks]
            return max_samples_per_shard - min(sizes)

    shard_sequence_names = []
    chunks = []
    sample_count = sum([len(x) for x in subset_data.values()])
    target_chunk_count = math.ceil(sample_count / max_samples_per_shard)

    for sequence_name, sequence_data in reversed(sorted(subset_data.items(), key=lambda pair: len(pair[1]))):
        # allow shards to exceed targeted size (number of samples) by up to 20%
        if len(sequence_data) - _compute_avail_size() < 0.2 * max_samples_per_shard:
            # add sequence data as a whole
            _add_chunk(sequence_name, sequence_data)
        else:
            tc.print_c(f"Splitting sequence '{sequence_name}' because it does not fit into shard.",
                       tc.Color.YELLOW)
            # break sequence data into fragments of size max_samples_per_shard (or less)
            sequence_fragment = {}
            for idx, (x, y) in enumerate(sorted(sequence_data.items())):
                sequence_fragment[x] = y
                if len(sequence_fragment) == _compute_avail_size():
                    _add_chunk(sequence_name, sequence_fragment)
                    sequence_fragment = {}
            if len(sequence_fragment) > 0:
                _add_chunk(sequence_name, sequence_fragment)

    for idx, (sequence_name, chunk) in enumerate(zip(shard_sequence_names, chunks)):
        shard_id = f"{idx:04d}"
        sharded_dataset_structure[subset_name][shard_id] = chunk


def _assign_files_to_shards_ignoring_sequence_boundaries(
        sharded_dataset_structure: MetaDataStructure,
        subset_name: str,
        subset_data: SubSetMetaDataStructure,
        max_samples_per_shard: int,
        global_shuffle: bool = False) -> None:
    """ Assigns the samples to shards ignoring sequence boundaries.

    Sequences (if they exist) may be broken up into multiple shards or multiple sequences may be placed in the same
    shard, depending on the length of the sequences and how many samples shall go in one shard (at most). Ignoring
    sequence boundaries helps to balance the shards sizes better.
    """
    sample_count = 0

    for sequence_data in subset_data.values():
        sample_count += len(sequence_data)

    # determine how many samples go into one shard
    number_of_shards = (sample_count + max_samples_per_shard - 1) // max_samples_per_shard
    samples_per_shard = sample_count / number_of_shards

    sample_count = 0
    shard_counter = -1
    shard_id = None
    samples = []
    for sequence_name, sequence_data in sorted(subset_data.items()):
        for sample in sorted(sequence_data.items()):
            samples.append(sample)

    if global_shuffle:
        random.shuffle(samples)

    for sample_id, sample_data in samples:
        sample_count += 1
        # check whether to move to next shard
        if sample_count > round((shard_counter + 1) * samples_per_shard):
            shard_counter += 1
            shard_id = f"{shard_counter:04d}"
        # add sample to shard
        sharded_dataset_structure[subset_name][shard_id][sample_id] = sample_data


def assign_files_to_shards(dataset_structure: MetaDataStructure,
                           max_samples_per_shard: int | dict[str, int],
                           global_shuffle_for_subsets: list[str] = [],
                           preserve_sequence_boundaries_for_subsets: list[str] = []) -> MetaDataStructure:
    """
    Assigns the files to shards either ignoring or preserving the boundaries of sequences based on the user's choice.

    :param dataset_structure: The dataset structure as defined by build_dataset_structure above.
    :param max_samples_per_shard: A single shard will contain at maximum this many samples. Since sharding tries to make
        all shards of equal size the actual number of samples per shard might be substantially lower.
        Can also be a dict mapping subset names to the size to allow different sizing of shards for subsets.
    :param global_shuffle_for_subsets: A list of all subsets that should be globally shuffled. (Only works if sequence
        boundaries are ignored.)
    :param preserve_sequence_boundaries_for_subsets: A list of subsets for which the sequence boundaries should be
        obeyed. Preserving sequence boundaries will assign entire sequences to shards instead of individual samples. If
        sequences vary in size (number of samples) this will negatively impact load distribution in case of multi-GPU
        training and hence should be avoided (unless needed for training on sequence data). (Default: [])
    :return: A dict like the dataset structure, except that sequences have been replaced with shards and samples are
     assigned to these.
    """
    sharded_dataset_structure: MetaDataStructure = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for subset_name, subset_data in dataset_structure.items():
        max_samples = max_samples_per_shard
        if isinstance(max_samples, dict):
            max_samples = max_samples[subset_name]
        if subset_name in preserve_sequence_boundaries_for_subsets:
            _assign_files_to_shards_preserving_sequence_boundaries(sharded_dataset_structure, subset_name, subset_data,
                                                                   max_samples)
        else:
            global_shuffle = subset_name in global_shuffle_for_subsets
            _assign_files_to_shards_ignoring_sequence_boundaries(sharded_dataset_structure, subset_name, subset_data,
                                                                 max_samples, global_shuffle)
    return sharded_dataset_structure


"""doc
## **Step 3**: Write Shards and Metadata

Finally, we only need to write the shards and metadata to disk.

The metadata must have the following information:
```json
{
    "train": {
        "sample_count": 1337,
        "samples_per_shard": {
            "0000": 42,
            "0001": 13,
            "0002": 37,
            ...
        },
        "component_groups": {
            "camera1": {"min_components": 1, "max_components": 1, "all_components": ["name_of_component_1"]},
            "camera2": {"min_components": 1, "max_components": 1, "all_components": ["name_of_component_2"]},
            "annotations": {"min_components": 1, "max_components": 2,
                            "all_components": ["name_of_component_3", "name_of_component_4"]}
        }
    }
}
```

The shards must then be stored in tar files on the disk.
subset/shardid.componentgroup.tar

For example:

```bash
train/0000.camera1.tar
train/0000.camera2.tar
train/0000.annotations.tar
train/0001.camera1.tar
train/0001.camera2.tar
train/0001.annotations.tar
```
"""


def write_metadata(sharded_dataset_structure: MetaDataStructure,
                   out_dir: str,
                   subsets_to_pre_shuffle: list[str],
                   preserve_sequence_boundaries_for_subsets: list[str] = [],
                   check_sample_uniqueness_globally: bool = False) -> MetaDataStructure:
    """
    Writes the metadata to disk.

    :param sharded_dataset_structure: The structure of shards created by assign_files_to_shards.
    :param out_dir: The path where the outputs should be stored on disk.
    :param subsets_to_pre_shuffle: Whether the subset shall be pre-shuffled or not.
    :param preserve_sequence_boundaries_for_subsets: List of subsets for which the sequence boundaries were preserved
     during sharding.
    :param check_sample_uniqueness_globally: Whether the check for duplicate samples shall be performed across subset
     boundaries. If disabled, warnings will only be printed if a sample is duplicated within a subset. Disabled by
     default.

    :return: The generated metadata.
    """
    metadata = {x: {} for x in sharded_dataset_structure.keys()}
    for subset, subset_data in sharded_dataset_structure.items():
        metadata[subset]["sample_count"] = sum([len(shard) for shard in subset_data.values()])
        metadata[subset]["samples_per_shard"] = {}
        metadata[subset]["sequence_lengths_per_shard"] = {}
        metadata[subset]["component_groups"] = {}
        # store information on the type of pre-shuffling (if any) for each subset
        if subset not in subsets_to_pre_shuffle:
            shuffle_level = "no"
        else:
            if subset in preserve_sequence_boundaries_for_subsets:
                shuffle_level = "on_shard_level"
            else:
                shuffle_level = "on_subset_level"
        metadata[subset]["pre-shuffled"] = shuffle_level

        min_components = defaultdict(lambda: 100000)
        all_components = defaultdict(set)

        for shard_id, samples in subset_data.items():
            prev_sample_id = None
            sequence_lengths = []
            for sample_id, component_groups in samples.items():
                if preserve_sequence_boundaries_for_subsets:
                    sequence_id = sample_id.split(".")[0]
                    if sequence_id != prev_sample_id:
                        sequence_lengths.append(1)
                        prev_sample_id = sequence_id
                    else:
                        sequence_lengths[-1] += 1
                for component_group_name, components in component_groups.items():
                    component_count = len(components)
                    min_components[component_group_name] = min(min_components[component_group_name], component_count)
                    all_components[component_group_name].update(components)
            metadata[subset]["samples_per_shard"][shard_id] = len(samples)
            metadata[subset]["sequence_lengths_per_shard"][shard_id] = sequence_lengths
        for component_group_name in min_components.keys():
            metadata[subset]["component_groups"][component_group_name] = \
                {"min_components": min_components[component_group_name],
                 "max_components": len(all_components[component_group_name]),
                 "all_components": sorted(list(all_components[component_group_name]))}

    metadata["__meta__"] = {"version": metadata_file_format_version}

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with open(os.path.join(out_dir, "metadata.json"), 'w') as f:
        f.write(json.dumps(metadata, indent=2))

    filetree = {"__version__": filetree_file_format_version, "size": 0, "files": {}}
    tars = set()
    all_sample_ids = dict()
    warned = False
    for subset, sequences in sharded_dataset_structure.items():
        if not check_sample_uniqueness_globally:
            all_sample_ids = dict()
        for shard_id, samples in sequences.items():
            for sample_id, component_groups in samples.items():
                # Check whether the sample IDs are unique
                if sample_id in all_sample_ids.keys():
                    if not warned:
                        warned = True
                        tc.print_c("\nWarning: The sample IDs are not unique!", tc.Color.YELLOW)
                    print(f"Duplicate sample: '{sample_id}'")
                    print(f"- found here: '{subset}/{shard_id}/{sample_id}'")
                    print(f"- first encountered here: '{all_sample_ids[sample_id]}/{sample_id}'")
                else:
                    all_sample_ids[sample_id] = f"{subset}/{shard_id}"
                # Write information needed to create file tree in datapipefs
                for component_group, components in component_groups.items():
                    for component_name, (_, file_path, file_size) in components.items():
                        current_dir = filetree
                        segments = file_path.split(os.path.sep)
                        for segment in segments[:-1]:
                            if segment.endswith(".zip") or segment.endswith(".tar"):
                                continue
                            if segment not in current_dir["files"]:
                                current_dir["size"] += 1
                                current_dir["files"][segment] = {"size": 0, "files": {}}
                            current_dir = current_dir["files"][segment]
                        cg_suffix = f".{component_group}" if component_group != "" else ""
                        ext = file_path.split(".")[-1].lower()
                        tar_path = f"{subset}/{shard_id}{cg_suffix}.tar"
                        tars.add(tar_path)
                        current_dir["files"][segments[-1]] = {"archive": tar_path,
                                                              "file": f"{sample_id}.{component_name}.{ext}",
                                                              "size": file_size}

    with open(os.path.join(out_dir, "filetree.mpk"), "wb") as f:
        tars = list(tars)
        tars.sort()
        msgpack.dump([filetree, tars], f)

    return metadata


def write_shards(metadata: MetaDataStructure,
                 sharded_dataset_structure: MetaDataStructure,
                 out_dir: str,
                 overwrite: bool = False) -> None:
    """
    Writes the shards to disk.

    It supports reading the input files from zip archives, uncompressed tar archives or directly from the filesystem.

    :param metadata: The metadata generated for this dataset. It is used to look-up which components and component
     groups are present
    :param sharded_dataset_structure: The structure of shards created by assign_files_to_shards.
    :param in_dir: The path from which the dataset will be read.
    :param out_dir: The path where the outputs should be stored on disk.
    :param overwrite: Whether existing tars should be overwritten. (Default: False)
    """
    input_archives = dict()
    mode = "w" if overwrite else "x"
    tc.print_c("\nWriting shards for subsets:", tc.Format.BOLD, flush=True)
    for subset_name, sequences in sharded_dataset_structure.items():
        subset_dir = os.path.join(out_dir, subset_name)
        if not os.path.exists(subset_dir):
            os.makedirs(subset_dir)
        component_groups = metadata[subset_name]["component_groups"]
        last_subset_size_written = 0
        subset_size_written = 0
        sequences_tqdm = tqdm(sequences.items(), desc=f"{subset_name} 0 GiB")
        for shard_id, samples in sequences_tqdm:
            if metadata[subset_name]["pre-shuffled"] != "no":
                sample_ids = list(samples.keys())
                random.shuffle(sample_ids)
            else:
                sample_ids = sorted(samples.keys())

            for component_group_name, component_group_info in component_groups.items():
                if component_group_name != "":
                    tar_file_name = shard_id + "." + component_group_name
                else:
                    tar_file_name = shard_id
                out_tar_path = os.path.join(subset_dir, tar_file_name + ".tar")
                components = component_group_info["all_components"]
                with tarfile.open(out_tar_path, mode) as out_file:
                    for sample_id in sample_ids:
                        for component_name in components:
                            if component_name in samples[sample_id][component_group_name]:
                                in_dir, in_file_path = samples[sample_id][component_group_name][component_name][:2]
                                ext = in_file_path.split(".")[-1].lower()
                                in_file_path = os.path.join(in_dir, in_file_path)
                                new_file_name = f"{sample_id}.{component_name}.{ext}"
                                base_path, inner_path = _split_path(in_file_path)
                                if inner_path != "":
                                    inner_path = inner_path.replace("\\", "/")
                                    if base_path.endswith(".zip"):
                                        if base_path not in input_archives:
                                            input_archives[base_path] = zipfile.ZipFile(base_path, "r")
                                        archive = input_archives[base_path]
                                        zip_info = archive.getinfo(inner_path)
                                        tar_info = tarfile.TarInfo(new_file_name)
                                        tar_info.size = zip_info.file_size
                                        tar_info.mtime = time.mktime((*zip_info.date_time, 0, 0, 0))
                                        with archive.open(zip_info) as member:
                                            out_file.addfile(tar_info, member)
                                            subset_size_written += tar_info.size
                                    elif base_path.endswith(".tar"):
                                        if base_path not in input_archives:
                                            input_archives[base_path] = tarfile.open(base_path)
                                        archive = input_archives[base_path]
                                        tar_info = archive.getmember(inner_path)
                                        member = archive.extractfile(tar_info)
                                        tar_info.name = new_file_name
                                        out_file.addfile(tar_info, member)
                                        subset_size_written += tar_info.size
                                    else:
                                        raise NotImplementedError(f"Reading from archive type not supported:"
                                                                  f" {base_path[-4:]}")
                                else:
                                    out_file.add(in_file_path, new_file_name)
                                    file_size = os.stat(in_file_path).st_size
                                    subset_size_written += file_size
                            else:
                                tc.print_c(f"Warning: no component '{component_name}' in sample '{sample_id}' of subset"
                                           f" '{subset_name}'", tc.Color.YELLOW, flush=True)
                        description = f"{subset_name} {round(subset_size_written / (1024 * 1024 * 1024), 1)} GiB"
                        if (subset_size_written - last_subset_size_written) > 0.1 * 1024 * 1024 * 1024:
                            sequences_tqdm.set_description(description)
                            last_subset_size_written = subset_size_written
    for archive in input_archives.values():
        archive.close()


"""doc
## **Step 4**: Write dataset info file

Lastly, store information about the dataset in a JSON file. This way web pages that list the datasets available and
provide useful information about them can be generated automatically.  

The dataset info file should have the following information:
```json
[
  {
    "short name": "abbreviation/acronym as str",
    "full name": "full name of dataset as str",
    "sensors": [
      "list of sensors for which data is included in the dataset, e.g.:",
      "RGB",
      "depth",
      "lidar",
      "radar",
      "IMU",
      "..."
    ],
    "camera setup": "none/mono/stereo/matrix/custom",
    "nature of data": "synthetic/real/mixed as str",
    "tasks": [
      "a list of task the dataset was created for, e.g.:",
      "image classification",
      "object detection",
      "object segmentation",
      "scene segmentation",
      "body pose estimation",
      "hand pose estimation",
      "..."
    ],
    "project page": "link to project page, e.g. on GitHub (if any) as str",
    "download page": "link to website from which the dataset can be downloaded as str",
    "paper_url": "link to the paper",
    "license name": "name of license as str, e.g. EULA, GPL-3.0, MIT, BSD-3-Clause, ... "
                    "see: https://opensource.org/licenses/alphabetical",
    "converted by": [
      "list of names of the people",
      "that converted the dataset to a tar/sharded dataset"
    ]
  }
]
```

First encode the information on the dataset in a string using encode_ds_info(). Then write it to disk using
write_ds_info().

"""


def encode_ds_info(*, short_name: str, full_name: str, sensors: list[str], camera_setup: str, nature_of_data: str,
                   tasks: list[str], project_page: str, code_repo: str, paper_url: str,
                   license_name: str, commercial_use: str = "", converted_by: list[str]) -> dict[str, str | list[str]]:
    """ Writes a JSON file with information on the dataset into the specified output directory.

    :param short_name: The short name of the dataset. This is often an abbreviation or an acronym.
    :param full_name: The full name of the dataset.
    :param sensors: A list of the types of sensors that where used to record (or simulate) the data.
    :param camera_setup: The type of camera setup used. Should typically be on of: none, mono, stereo, matrix, custom.
    :param nature_of_data: Whether the data is synthetic, real or mixed.
    :param tasks: A list of tasks the dataset is intended for, e.g.: "image classification", "object detection",
      "object segmentation", "scene segmentation", "body pose estimation", "hand pose estimation".
    :param project_page: The URL pointing to the webpage of the project that created the dataset.
    :param code_repo: The URL pointing to the webpage where the code related to the dataset is stored.
    :param paper_url: The URL pointing to the paper on the dataset.
    :param license_name: The name of the license in case a standardized license is used for the dataset.
    :param commercial_use: Whether the license allows commercial use.
    :param converted_by: A list of the names of the people that converted the dataset.
    :return: A dictionary containing the dataset information.
    """
    params = {"short_name": (short_name, str), "full_name": (full_name, str), "sensors": (sensors, list),
              "camera_setup": (camera_setup, str), "nature_of_data": (nature_of_data, str), "tasks": (tasks, list),
              "project_page": (project_page, str), "code_repo": (code_repo, str),
              "paper_url": (paper_url, str), "license_name": (license_name, str),
              "commercial_use": (commercial_use, str),  "converted_by": (converted_by, list)}
    # check the parameters have the correct type
    for param_name, (param, data_type) in params.items():
        if not isinstance(param, data_type):
            raise TypeError(f"'{param_name}' must be a '{data_type.__name__}'")
    # check that a minimal set of parameters is not empty
    non_empty_params = {"short_name": short_name, "full_name": full_name, "converted_by": converted_by}
    for param_name, param in non_empty_params.items():
        if len(param) == 0:
            raise ValueError(f"'{param_name}' may not be empty")
    # create the data structure for the dataset information
    ds_info = {"short name": short_name, "full name": full_name, "sensors": sensors, "camera setup": camera_setup,
               "nature of data": nature_of_data, "tasks": tasks, "project page": project_page, "paper": paper_url,
               "code repository": code_repo, "license name": license_name, "commercial use": commercial_use,
               "converted by": converted_by}
    return ds_info


def write_ds_info(ds_info: dict | list[dict], output_dir: str) -> None:
    """ Writes a JSON file with information on the dataset into the specified output directory.

    :param ds_info: The dataset info that shall be written to disk. Can be a single one or a list of ds_info dicts.
    :param output_dir: The path to the directory where the dataset info file will be stored.
    """
    if isinstance(ds_info, dict):
        ds_info = [ds_info]
    with open(os.path.join(output_dir, "ds_info.json"), "w") as out_file:
        json.dump(ds_info, out_file, indent=2)
