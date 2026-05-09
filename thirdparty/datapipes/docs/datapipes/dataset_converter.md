[Back to Overview](../README.md)

# datapipes.dataset_converter: Dataset Conversion

> Convert datasets into shards with metadata.

## Simplified Conversion

Use convert_samples() to convert one or more sets of samples to sharded datasets with a single function call.

It does not work with sequential datasets and unlike the full conversion process, it cannot be customized.
Furthermore, the function will not scan the filesystem for you. Therefore, it is up to you to come up with the list
of samples for each subset (or data split).
Alternatively, you can use the convert_dataset() function (see below). It will scan the filesystem for you,
but you need to provide a function that extracts some information from the paths of the discovered files.

## *def* **convert_samples** [[src]](../../datapipes/dataset_converter.py#L79)
Converts one or more sets of samples into a sharded dataset.

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
* **in_dir**: The path to the directory that contains the dataset in its current form. The paths of the files in
    the dataset will be retained relative to this path.
* **out_dir**: The path to the directory the dataset shall be written to.
* **samples**: The samples of the dataset that shall be sharded, structured as follows:
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
* **target_shard_size_mb**: The targeted size of each shard. If unsure use the default, which is 400 MB.
* **splits_to_pre_shuffle**: Which splits shall be pre-shuffled. By default, all are. Typically, you want the
    "train" split to be shuffled, while shuffling is not necessary for "test". For "val" it depends on whether you
    would like to train with a mix of "train" + "val" at some point.

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

## *def* **convert_dataset** [[src]](../../datapipes/dataset_converter.py#L184)
Creates a sharded version of a dataset.

Further customization is possible by copying the code of this function or by using the "dataset_conversion"
template in the "templates" folder. It also contains a template for the implementation of the
path_to_file_info_converter callable.

* **in_dir**: The directory that contains the source dataset.
* **out_dir**: The directory the sharded dataset will be written to.
* **path_to_file_info_converter**: A function or callable class that creates a FileInfo object from a file path.
* **component_grouping**: A dictionary which indicates which sample components shall be stored together.
 By default, all sample components are stored together.
* **target_tar_file_size**: The average size of the resulting shards. Defaults to 400 MiB.
* **target_gpu_count**: The maximum number of GPUs that will be supported by the sharded dataset. Defaults to 8.
* **splits_to_pre_shuffle**: Which subsets (or data splits) shall be pre-shuffled. Defaults to all.
* **path_filter**: A function that returns a boolean to indicate whether the given path (dir or archive) should be
    recursed into or not.
* **preserve_sequential_ordering**: Whether to create a sequence dataset, where the order of samples within each
    sequence is preserved. Defaults to False.

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

## *class* **FileInfo** [[src]](../../datapipes/dataset_converter.py#L285)
Information about a file.

The subset and sequence (if any) that the sample belongs.
The name/identifier of the sample.
A string identifying the sample component contained in the file.
The "sequence_name", "sample_name" and "component_id" must not contain any dots (".").

## *def* **gather_fileinfos** [[src]](../../datapipes/dataset_converter.py#L312)
Find all relevant files on disk and meta information about them by walking over a directory and its subdirectories.

This function will also recurse into zip archives and uncompressed tar archives. It is, therefore, not necessary to
extract datasets that are stored in such archives.
* **in_dir**: Path to folder to traverse for searching files.
* **convert_path_to_fileinfo**: A function that maps filepaths to relevant information on a file.
    The function receives a path and returns a FileInfo object.
* **verbose**: Whether to print information on the progress of the directory traversal.
* **path_filter**: A filter that specifies whether a given directory or archive shall be processed (True) or
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

## *def* **build_dataset_structure** [[src]](../../datapipes/dataset_converter.py#L409)
Given a list of file_infos build a structure for the dataset.

* **file_infos**: A list of FileInfo objects as created by gather_fileinfos().
* **component_map**: A dictionary mapping component groups to a list of components in it.
    For example with images in separate groups:
     `{"camera1": ["camera1"], "camera2": ["camera2"], "annotations": ["label1", "label2"])`.
    Or without groups: `{"": ["camera1", "camera2", "label1", "label2"]}`.
* **find_component_group**: (Optional) A function that finds the component name in the component map.
 (Default: _find_component_group)
* **returns**: A nested dictionary representing the dataset structure as this hierarchy:
 dict[subset][sequence][sample][component_group][component] = filepath.

## *def* **suggest_shard_size** [[src]](../../datapipes/dataset_converter.py#L437)
Suggests a suitable shard size for each subset.

First, it will check whether the dataset is small. If that is the case, it will place all files in one shard for
each subset and component group.
For large datasets it will try to determine the optimal shard size for each subset. If the component groups have
different sizes (in terms of bytes) it tries to adjust the target TAR size such that the shard file size of the
smallest component group is above min_tar_size and the shard file size of the largest component group is below
max_tar_size. Furthermore, it ensures that the number of shards is divisible by target_gpu_count.

* **dataset_structure**: Information on the structure of the dataset.
* **target_tar_size_mb**: The target size of the TAR shards in MiB. Defaults to 400 MiB.
* **target_gpu_count**: The number of GPU that shall be supported for parallel training and testing. Defaults to 8.
* **small_ds_size_threshold_gb**: The size threshold in GiB below which datasets shall be considered small and
 hence be placed in a single tar file. Defaults to 16 GiB. Reasonable range: 8 GiB - 26 GiB.
* **max_shards**: The maximum number of shards to generate. Defaults to 2080.
* **preserve_sequential_ordering**: Whether the ordering of samples within each sequence shall be preserved.
* **returns**: The suggested shard sizes (in number of samples) for each subset.

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

## *def* **assign_files_to_shards** [[src]](../../datapipes/dataset_converter.py#L704)
Assigns the files to shards either ignoring or preserving the boundaries of sequences based on the user's choice.

* **dataset_structure**: The dataset structure as defined by build_dataset_structure above.
* **max_samples_per_shard**: A single shard will contain at maximum this many samples. Since sharding tries to make
    all shards of equal size the actual number of samples per shard might be substantially lower.
    Can also be a dict mapping subset names to the size to allow different sizing of shards for subsets.
* **global_shuffle_for_subsets**: A list of all subsets that should be globally shuffled. (Only works if sequence
    boundaries are ignored.)
* **preserve_sequence_boundaries_for_subsets**: A list of subsets for which the sequence boundaries should be
    obeyed. Preserving sequence boundaries will assign entire sequences to shards instead of individual samples. If
    sequences vary in size (number of samples) this will negatively impact load distribution in case of multi-GPU
    training and hence should be avoided (unless needed for training on sequence data). (Default: [])
* **returns**: A dict like the dataset structure, except that sequences have been replaced with shards and samples are
 assigned to these.

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

## *def* **write_metadata** [[src]](../../datapipes/dataset_converter.py#L782)
Writes the metadata to disk.

* **sharded_dataset_structure**: The structure of shards created by assign_files_to_shards.
* **out_dir**: The path where the outputs should be stored on disk.
* **subsets_to_pre_shuffle**: Whether the subset shall be pre-shuffled or not.
* **preserve_sequence_boundaries_for_subsets**: List of subsets for which the sequence boundaries were preserved
 during sharding.
* **check_sample_uniqueness_globally**: Whether the check for duplicate samples shall be performed across subset
 boundaries. If disabled, warnings will only be printed if a sample is duplicated within a subset. Disabled by
 default.

* **returns**: The generated metadata.

## *def* **write_shards** [[src]](../../datapipes/dataset_converter.py#L898)
Writes the shards to disk.

It supports reading the input files from zip archives, uncompressed tar archives or directly from the filesystem.

* **metadata**: The metadata generated for this dataset. It is used to look-up which components and component
 groups are present
* **sharded_dataset_structure**: The structure of shards created by assign_files_to_shards.
* **in_dir**: The path from which the dataset will be read.
* **out_dir**: The path where the outputs should be stored on disk.
* **overwrite**: Whether existing tars should be overwritten. (Default: False)

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

## *def* **encode_ds_info** [[src]](../../datapipes/dataset_converter.py#L1041)
Writes a JSON file with information on the dataset into the specified output directory.

* **short_name**: The short name of the dataset. This is often an abbreviation or an acronym.
* **full_name**: The full name of the dataset.
* **sensors**: A list of the types of sensors that where used to record (or simulate) the data.
* **camera_setup**: The type of camera setup used. Should typically be on of: none, mono, stereo, matrix, custom.
* **nature_of_data**: Whether the data is synthetic, real or mixed.
* **tasks**: A list of tasks the dataset is intended for, e.g.: "image classification", "object detection",
  "object segmentation", "scene segmentation", "body pose estimation", "hand pose estimation".
* **project_page**: The URL pointing to the webpage of the project that created the dataset.
* **code_repo**: The URL pointing to the webpage where the code related to the dataset is stored.
* **paper_url**: The URL pointing to the paper on the dataset.
* **license_name**: The name of the license in case a standardized license is used for the dataset.
* **commercial_use**: Whether the license allows commercial use.
* **converted_by**: A list of the names of the people that converted the dataset.
* **returns**: A dictionary containing the dataset information.

## *def* **write_ds_info** [[src]](../../datapipes/dataset_converter.py#L1083)
Writes a JSON file with information on the dataset into the specified output directory.

* **ds_info**: The dataset info that shall be written to disk. Can be a single one or a list of ds_info dicts.
* **output_dir**: The path to the directory where the dataset info file will be stored.
