[Back to Overview](../../README.md)

> Copyright (c) 2022 DFKI GmbH - All Rights Reserved
> 
> Written by Michael Fürst <Michael.Fuerst@dfki.de>, October 2022

# Example: NuScenes - Create Tars

Please read the documentation on converting datasets before following this example. (See in doc "datapipes.dataset_converter")

This dataset and tutorial was created with the following versions.

Example:
```python
from datapipes.versions import api_version, metadata_file_format_version

print(f"API-Version: {api_version}")
print(f"File Format Version: {metadata_file_format_version}")
```

Output:
```
API-Version: 0.1.0
File Format Version: 1
```

## **Step 1**: Gather information on the dataset

The original nuScenes dataset has all annotations in a single annotation database which is indexed using tokens (sample token and sensor token), the images and pointclouds are then stored separately in a file for each. Filenames can be retrieved from the annotation database if the token is known. However, you cannot infer the token given the filename.

### Extract FileInfo

Due to this structure, we iterate over the dataset and collect all FileInfo for the files that way, instead of parsing it from the filenames.
With that list we can use the build_dataset_structure function from the library and continue using pre-implemented functions.

Example:
```python
import os
from typing import Dict, List
from tqdm import tqdm
from os.path import join
from nuscenes import NuScenes
from nuscenes.utils.splits import mini_train, mini_val

from datapipes.dataset_converter import FileInfo

RAW_DATASET="/ds-av/public_datasets/nuscenes/raw"
CAMERA_SENSORS = [
    "CAM_FRONT","CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
]
LIDAR_SENSORS = ["LIDAR_TOP"]
RADAR_SENSORS = [
    "RADAR_FRONT", "RADAR_FRONT_RIGHT", "RADAR_FRONT_LEFT",
    "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"
]

def gather_fileinfos(dataroot, subsets) -> List[FileInfo]:
    """
    Get a list of sample info.
    """
    file_infos = []
    for subset_name, version, sequences in subsets:
        dataset = NuScenes(version=version, dataroot=dataroot, verbose=False)
    
        for scene in tqdm(dataset.scene, desc="Collecting tokens"):
            if scene["name"] not in sequences: continue
            sample_token = scene['first_sample_token']
            sample = dataset.get('sample', sample_token)
            file_infos.extend(
                collect_fileinfo(dataroot, subset_name, dataset, scene, sample_token, sample)
            )
            while sample_token != scene['last_sample_token']:
                sample_token = sample["next"]
                sample = dataset.get('sample', sample_token)
                file_infos.extend(
                    collect_fileinfo(dataroot, subset_name, dataset, scene, sample_token, sample)
                )
    return file_infos

def collect_fileinfo(dataroot, subset_name, dataset, scene, sample_token, sample):
    file_infos = []
    for sensor in CAMERA_SENSORS + LIDAR_SENSORS + RADAR_SENSORS:
        sample_data = dataset.get('sample_data', sample['data'][sensor])
        file_path = join(dataroot, sample_data["filename"])
        file_infos.append(FileInfo(
            subset=subset_name,
            sequence_name=scene["name"],
            sample_name=sample_token,
            component_id=sensor,
            file_path=file_path,
            file_size=os.path.getsize(file_path)
        ))
    return file_infos

file_infos = gather_fileinfos(RAW_DATASET, [
    ("mini_train", "v1.0-mini", mini_train),
    ("mini_val", "v1.0-mini", mini_val),
])

for info in file_infos[:2]:
    print(info)
print(f"(output truncated after 10/{len(file_infos)} entries)")
```

Output:
```
Collecting tokens: 100%|██████████| 10/10 [00:00<00:00, 18.56it/s]
Collecting tokens: 100%|██████████| 10/10 [00:00<00:00, 71.66it/s]FileInfo(subset='mini_train', sequence_name='scene-0061', sample_name='ca9a282c9e77460f8360f564131a8af5', component_id='CAM_FRONT', file_path='/ds-av/public_datasets/nuscenes/raw/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg', file_size=131197)
FileInfo(subset='mini_train', sequence_name='scene-0061', sample_name='ca9a282c9e77460f8360f564131a8af5', component_id='CAM_FRONT_RIGHT', file_path='/ds-av/public_datasets/nuscenes/raw/samples/CAM_FRONT_RIGHT/n015-2018-07-24-11-22-45+0800__CAM_FRONT_RIGHT__1532402927620339.jpg', file_size=141131)
(output truncated after 10/4848 entries)

```

### Use Cases

The most common usage for nuScenes is to use all keyframes, but not all sensors. Common sensor configurations in use are:

* Lidar only
* Radar (all) only
* Camera (all) only
* Camera front only
* Lidar + Camera
* Radar + Camera
* any other combination

### Packaging Separately

Due to the size of the data and based on these use cases, we want to split data into separate TARs.

1. LiDAR is in a separate tar.
2. Cameras are in a separate tar per camera.
3. Radar is in a separate tar, but all Radars are in a single tar (they are small pointclouds).
4. Annotations are not packaged at all, since they are in a single Database stored as a JSON file already.

### Build dataset structure

Since we created the list of fileinfos it is very simple to build the dataset structure. We simply call a function from the datapipes library and define a component map. The component map defines how sensors are mapped into groups of components stored in a single tar.

Example:
```python
from datapipes.dataset_converter import build_dataset_structure

COMPONENT_MAP = dict(
    # The lidar goes in a tar
    lidar=["LIDAR_TOP"],
    # All radars go in a single tar
    radar=RADAR_SENSORS,
    # Put each camera sensor in a separate tar
    **{ x.lower(): [x] for x in CAMERA_SENSORS}
)

dataset_structure = build_dataset_structure(file_infos, COMPONENT_MAP)

# Print a sample
print(list(dataset_structure["mini_train"]["scene-0061"]["scene-0061.ca9a282c9e77460f8360f564131a8af5"].keys()))
print(dataset_structure["mini_train"]["scene-0061"]["scene-0061.ca9a282c9e77460f8360f564131a8af5"]["radar"])
```

Output:
```
['cam_front', 'cam_front_right', 'cam_front_left', 'cam_back', 'cam_back_left', 'cam_back_right', 'lidar', 'radar']
{'RADAR_FRONT': ('/ds-av/public_datasets/nuscenes/raw/samples/RADAR_FRONT/n015-2018-07-24-11-22-45+0800__RADAR_FRONT__1532402927664178.pcd', 9471), 'RADAR_FRONT_RIGHT': ('/ds-av/public_datasets/nuscenes/raw/samples/RADAR_FRONT_RIGHT/n015-2018-07-24-11-22-45+0800__RADAR_FRONT_RIGHT__1532402927639817.pcd', 8611), 'RADAR_FRONT_LEFT': ('/ds-av/public_datasets/nuscenes/raw/samples/RADAR_FRONT_LEFT/n015-2018-07-24-11-22-45+0800__RADAR_FRONT_LEFT__1532402927652686.pcd', 9342), 'RADAR_BACK_LEFT': ('/ds-av/public_datasets/nuscenes/raw/samples/RADAR_BACK_LEFT/n015-2018-07-24-11-22-45+0800__RADAR_BACK_LEFT__1532402927668356.pcd', 9471), 'RADAR_BACK_RIGHT': ('/ds-av/public_datasets/nuscenes/raw/samples/RADAR_BACK_RIGHT/n015-2018-07-24-11-22-45+0800__RADAR_BACK_RIGHT__1532402927635538.pcd', 9471)}
```

## **Step 2**: Shard the dataset

First let's figure out the shard sizes. For this we can use the suggestion tool and see if the results make sense.

Example:
```python
from datapipes.dataset_converter import suggest_shard_size

shard_sizes = suggest_shard_size(dataset_structure)
```

Output:
```

Subset statistics
=================

subset "mini_train":
  sample count: 323
  size of component group 'cam_back': 52.6 MiB
  size of component group 'cam_back_left': 55.7 MiB
  size of component group 'cam_back_right': 59.6 MiB
  size of component group 'cam_front': 58.2 MiB
  size of component group 'cam_front_left': 58.0 MiB
  size of component group 'cam_front_right': 60.4 MiB
  size of component group 'lidar': 213.9 MiB
  size of component group 'radar': 11.9 MiB

subset "mini_val":
  sample count: 81
  size of component group 'cam_back': 9.5 MiB
  size of component group 'cam_back_left': 11.5 MiB
  size of component group 'cam_back_right': 11.4 MiB
  size of component group 'cam_front': 11.3 MiB
  size of component group 'cam_front_left': 12.2 MiB
  size of component group 'cam_front_right': 11.7 MiB
  size of component group 'lidar': 53.6 MiB
  size of component group 'radar': 3.0 MiB

total size of dataset: 0.7 GiB

The dataset is small. Will create one TAR file per subset and component group.
```

Now that we have the size suggestions and inspected it, we actually assign the files to the shards.

For sharding we will do it different for "train" and "val"/"test".
The latter should remain sorted, so that trackers could be applied, but for training, we want as much randomness as possible.

This means on "train" we will merge all sequences in a single sequence and shuffle before sharding. On the other hand for "val" and "test" we will create the shards while preserving the sequence boundaries.

Example:
```python
from datapipes.dataset_converter import assign_files_to_shards

shard_info = assign_files_to_shards(
    dataset_structure,
    shard_sizes,
    preserve_sequence_boundaries_for_subsets=["mini_val"],
    global_shuffle_for_subsets=["mini_train"]
)

print(f"Shards (mini_train): {list(shard_info['mini_train'].keys())}")
print(f"Shards (mini_val): {list(shard_info['mini_val'].keys())}")
```

Output:
```
Shards (mini_train): ['0000']
Shards (mini_val): ['0000', '0001']
```

## **Step 3**: Write Shards and Metainfo

Finally with the shardinfo we can write the shards and metainfo so we can load them again.

Example:
```python
from datapipes.dataset_converter import write_metadata, write_shards

TAR_DATASET = "/home/fuerst/Desktop/Datasets/nuscenes/keyframes_mini"

metadata = write_metadata(shard_info, TAR_DATASET, subsets_to_pre_shuffle=["train"])
write_shards(metadata, shard_info, RAW_DATASET, TAR_DATASET, overwrite=True)
```

Output:
```
100%|██████████| 1/1 [00:15<00:00, 15.28s/it]
100%|██████████| 2/2 [00:03<00:00,  1.87s/it]
```

We also need to copy over the original annotation database.
Since these are monolithic files anyways and we want to use the original loader we do not put them in tars. A simple copy is enough.

Example:
```python
import os
import shutil

def copy_annotation_databases(dataset_path, target_path, version):
    database_files = [
        "attribute.json",
        "calibrated_sensor.json",
        "category.json",
        "ego_pose.json",
        "instance.json",
        "lidarseg.json",
        "log.json",
        "map.json",
        "panoptic.json",
        "sample.json",
        "sample_annotation.json",
        "sample_data.json",
        "scene.json",
        "sensor.json",
        "visibility.json"
    ]
    os.makedirs(os.path.join (target_path, version), exist_ok=True)
    for name in database_files:
        shutil.copyfile(os.path.join(dataset_path, version, name), os.path.join (target_path, version, name))

copy_annotation_databases(RAW_DATASET, TAR_DATASET, version="v1.0-mini")
```

## **Step 4**: Validate the tars

We list and inspect all the tars. For speed we will only use the mini dataset for testing.

To do this we first list all tars that match and then print first N filenames in the tar to check if the order matches.

Example:
```python
from os import listdir
from os.path import join
import tarfile

def check_tars(tar_folder, split, mini=False):
    tars = listdir(tar_folder)
    def _is_tar(fname):
        return fname.endswith(".tar")
    tars = filter(_is_tar, tars)
    if mini:
        tars = filter(lambda fname: "mini" in fname, tars)
    else:
        tars = filter(lambda fname: not "mini" in fname, tars)
    tars = filter(lambda fname: split in fname, tars)
    tars = sorted(tars)
    for tar_idx, fname in enumerate(tars):
        print(fname)
        if tar_idx == 0:
            with tarfile.open(join(tar_folder, fname), mode="r") as tar:
                for idx, info in enumerate(tar):
                    print(f"   {info.name}")
                    if idx >= 2:             
                        print("   ...")
                        break
        else:
            print("   ...")

check_tars(tar_folder=TAR_DATASET, split="train", mini=True)
```

## **Step 5**: Redo for full dataset

So far we have only processed nuScenes mini for testing purposes. Now it is time to process the entire nuScenes dataset.

We do this in two parts: First we create the shard assignment and inspect it; then we write it to disk.

Example:
```python
from nuscenes.utils.splits import train, val, test

file_infos = gather_fileinfos(RAW_DATASET, [
    ("train", "v1.0-trainval", train),
    ("val", "v1.0-trainval", val),
    ("test", "v1.0-test", test),
])
dataset_structure = build_dataset_structure(file_infos, COMPONENT_MAP)
shard_sizes = suggest_shard_size(dataset_structure)
shard_info = assign_files_to_shards(
    dataset_structure,
    shard_sizes,
    global_shuffle_for_subsets=["train"],
    preserve_sequence_boundaries_for_subsets=["val", "test"]
)
for subset, shards in shard_info.items():
    print(f"Shards ({subset}): {list(shards.keys())}")
```

Output:
```
Collecting tokens: 100%|██████████| 850/850 [00:55<00:00, 15.33it/s]
Collecting tokens: 100%|██████████| 850/850 [00:11<00:00, 76.21it/s] 
Collecting tokens: 100%|██████████| 150/150 [00:11<00:00, 13.24it/s]

Subset statistics
=================

subset "train":
  sample count: 28130
  size of component group 'cam_back': 3.6 GiB
  size of component group 'cam_back_left': 3.9 GiB
  size of component group 'cam_back_right': 4.1 GiB
  size of component group 'cam_front': 3.9 GiB
  size of component group 'cam_front_left': 4.0 GiB
  size of component group 'cam_front_right': 4.1 GiB
  size of component group 'lidar': 18.2 GiB
  size of component group 'radar': 1.0 GiB

subset "val":
  sample count: 6019
  size of component group 'cam_back': 785.6 MiB
  size of component group 'cam_back_left': 850.8 MiB
  size of component group 'cam_back_right': 882.4 MiB
  size of component group 'cam_front': 845.8 MiB
  size of component group 'cam_front_left': 870.4 MiB
  size of component group 'cam_front_right': 871.2 MiB
  size of component group 'lidar': 3.9 GiB
  size of component group 'radar': 227.5 MiB

subset "test":
  sample count: 6008
  size of component group 'cam_back': 816.6 MiB
  size of component group 'cam_back_left': 911.6 MiB
  size of component group 'cam_back_right': 936.7 MiB
  size of component group 'cam_front': 907.2 MiB
  size of component group 'cam_front_left': 967.1 MiB
  size of component group 'cam_front_right': 932.5 MiB
  size of component group 'lidar': 3.9 GiB
  size of component group 'radar': 226.6 MiB

total size of dataset: 61.5 GiB

Subset "train":
  Final tar file size(s) per component group: 77.5 MiB,  82.6 MiB,  88.4 MiB,  83.5 MiB,  86.0 MiB,  87.3 MiB,  388.1 MiB,  21.8 MiB

  Reading efficiency for subset train estimated at 96.62%. Compared to a
  reading efficiency of 2.23% when accessing individual unzipped files.
  Estimated dataset reading speed-up: 43.4

Subset "val":
  Final tar file size(s) per component group: 49.1 MiB,  53.2 MiB,  55.1 MiB,  52.9 MiB,  54.4 MiB,  54.5 MiB,  249.1 MiB,  14.2 MiB

  Reading efficiency for subset val estimated at 94.79%. Compared to a
  reading efficiency of 0.48% when accessing individual unzipped files.
  Estimated dataset reading speed-up: 197.0

Subset "test":
  Final tar file size(s) per component group: 51.0 MiB,  57.0 MiB,  58.5 MiB,  56.7 MiB,  60.4 MiB,  58.3 MiB,  248.7 MiB,  14.2 MiB

  Reading efficiency for subset test estimated at 94.97%. Compared to a
  reading efficiency of 0.50% when accessing individual unzipped files.
  Estimated dataset reading speed-up: 190.1
Shards (train): ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020', '0021', '0022', '0023', '0024', '0025', '0026', '0027', '0028', '0029', '0030', '0031', '0032', '0033', '0034', '0035', '0036', '0037', '0038', '0039', '0040', '0041', '0042', '0043', '0044', '0045', '0046', '0047']
Shards (val): ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014', '0015', '0016']
Shards (test): ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014', '0015', '0016']
```

Looks good. Let's write it.

Example:
```python
TAR_DATASET = "/home/fuerst/Desktop/Datasets/nuscenes/keyframes"
metadata = write_metadata(shard_info, TAR_DATASET, subsets_to_pre_shuffle=["train"])
write_shards(metadata, shard_info, RAW_DATASET, TAR_DATASET)
```

Output:
```
100%|██████████| 48/48 [25:52<00:00, 32.35s/it]
100%|██████████| 17/17 [05:22<00:00, 18.95s/it]
100%|██████████| 17/17 [04:54<00:00, 17.30s/it]
```

And copy all the annotation databases.

Example:
```python
copy_annotation_databases(RAW_DATASET, TAR_DATASET, version="v1.0-trainval")
copy_annotation_databases(RAW_DATASET, TAR_DATASET, version="v1.0-test")
```

## **Step 6**: Create a file with dataset info

Finally, to show up correctly on the dataset overview, we have to create a file with an overview over the dataset.
To create that file we use the write_ds_info file.

Note that the file is only created once for all variants (mini and full), since these properties are shared and not specific to the conversion done here.

Example:
```python
from datapipes.dataset_converter import encode_ds_info, write_ds_info

ds_info = encode_ds_info(
    short_name="nuscenes",
    full_name="nuScenes - Detection",
    sensors=["RGB", "radar", "lidar"],
    camera_setup="other",
    nature_of_data="real",
    tasks=["3d object detection"],
    project_page="https://www.nuscenes.org/nuscenes",
    code_repo="https://github.com/nutonomy/nuscenes-devkit",
    paper_url="https://openaccess.thecvf.com/content_CVPR_2020/papers/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.pdf",
    license_name="CC BY-NC-SA 4.0",
    converted_by=["Michael Fuerst"],
)
write_ds_info(
    ds_info,
    output_dir="/home/fuerst/Desktop/Datasets/nuscenes"
)
```

