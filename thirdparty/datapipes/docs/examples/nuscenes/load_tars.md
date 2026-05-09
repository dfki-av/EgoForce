[Back to Overview](../../README.md)

> Copyright (c) 2022 DFKI GmbH - All Rights Reserved
> 
> Written by Michael Fürst <Michael.Fuerst@dfki.de>, October 2022

# Example: NuScenes - Loading from Tars

> Here we load nuscenes from the tars into datapipes.

## Inspecting the metadata

First we create the base datapipeline creator to inspect the metadata in order to choose our parameters.

Example:
```python
from datapipes.base_pipeline import BasePipelineCreator

TAR_DATASET = "/home/fuerst/Desktop/Datasets/nuscenes/keyframes_mini"
factory = BasePipelineCreator(TAR_DATASET)
```

Our goal is to create a datapipe, to do this we need to know the following parameters:
* subset
* shuffle_buffer_size
* component_group_filter
* shuffle_shards

To choose correctly it makes sense to learn more about our shards first.
Let's print:
* available subsets
* the number of tars for each subset
* available component groups
* stats for each component group

Example:
```python
for subset in factory.get_subsets():
    print(f"Subset: {subset}")
    tars = factory.get_tar_files_for_subsets(subsets=[subset])
    print(f"Num Shards: {len(tars)}")
    groups = factory.get_component_groups(subset)
    print(f"Num Slices: {len(tars) / len(groups)}")
    shardlen = factory.get_average_shard_sample_count(subset)
    print(f"Avg Shard Len: {shardlen}")
    stats = factory.get_component_groups_stats(subset)
    for group in groups:
        if subset == "train":
            print(f"  Component Group: {group} {stats[group]}")
```

Output:
```
Subset: mini_train
Num Shards: 8
Num Slices: 1.0
Avg Shard Len: 323
Subset: mini_val
Num Shards: 8
Num Slices: 1.0
Avg Shard Len: 40
```

Now we can create a datapipe by specifying our parameters.
We want to use the `mini_train` subset and use all components except radar.
As we use it for training, we will shuffle the shards.

Choosing the `shuffle_buffer_size` is more difficult. In experiments simulating the randomness of the dataset we learned we want roughly `sequence_length * batch_size * 10`. However, in this example we will use only 40, to keep memory consumption on the local machine low.

Example:
```python
batch_size = 1
components = [x for group, stats in factory.get_component_groups_stats("mini_train").items() if group != "radar" for x in stats["all_components"]]

pipe = factory.create_datapipe(
    subsets=["mini_train"], 
    shuffle_buffer_size=int(factory.get_average_shard_sample_count("mini_train") * batch_size),
    shuffle_shards=True,
    components=components
)
```

## Add a decoder

Now the pipe will give us raw data, we need to decode it to use it. For this we can use decoders.

Example:
```python
from datapipes.utils.dispatcher import Dispatcher
from datapipes.decoders.image_decoder import ImageDecoder
from datapipes.decoders.pointcloud_decoder import PointcloudDecoder

decoders = {
    'CAM_FRONT': ImageDecoder('torchrgb'),
    'CAM_FRONT_RIGHT': ImageDecoder('torchrgb'),
    'CAM_FRONT_LEFT': ImageDecoder('torchrgb'),
    'CAM_BACK': ImageDecoder('torchrgb'),
    'CAM_BACK_LEFT': ImageDecoder('torchrgb'),
    'CAM_BACK_RIGHT': ImageDecoder('torchrgb'),
    'LIDAR_TOP': PointcloudDecoder(output_torch=True),
}

pipe = pipe.map(Dispatcher(decoders))
```

## Benchmark the pipeline

We benchmark the read speed of the pipeline to see if our additions after this stage are efficient or not.

Example:
```python
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

start = time.time()
start1 = start
N = 200
FPS = []
for i, sample in enumerate(tqdm(pipe, desc="Benchmarking")):
    stop = time.time()
    FPS.append(1.0 / ((stop - start1)))
    start1 = stop
    if i >= N - 1:
        break
stop = time.time()

plt.plot(FPS)
plt.show()

FPS = 1.0 / ((stop - start) / N)
print(FPS)
```

Output:
```
Benchmarking:  62%|██████▏   | 199/323 [00:25<00:15,  7.78it/s]
7.815569578771798
```
![data](../../../docs/jlabdev_images/c943d7beb8daba9fcd3ebcb50c26970d.png)

## Inject Annotations from Global File

Having the data decoded we want to inject our global annotations now.
Annotations can be read using the nuScenes dataset implementation, but we will not load images from there.

Example:
```python
from nuscenes import NuScenes
from nuscenes.utils.splits import train, val, test

class InjectNuscenesAnnotations(object):
    def __init__(self, version, dataroot) -> None:
        self.dataset = NuScenes(version=version, dataroot=dataroot, verbose=False)
    
    def __call__(self, sample):
        out = {}
        for fname, data in sample:
            out["sample_token"] = fname.split(".")[-3]
            out["scene"] = fname.split("/")[-1].split(".")[0]
            out[fname.split(".")[-2]] = data
        
        sample = self.dataset.get('sample', out["sample_token"])
        out["annotations"] = []
        for token in sample['anns']:
            out["annotations"].append(self.dataset.get('sample_annotation', token))

        return out

RAW_DATASET="/ds-av/public_datasets/nuscenes/raw"
pipe = pipe.map(InjectNuscenesAnnotations("v1.0-mini", RAW_DATASET))
```

## Test the pipe

With the pipe created, we want to test if we can get a sample from it.
To do this simply iterate over it with a for loop.

Example:
```python
import matplotlib.pyplot as plt
import numpy as np

for i, sample in enumerate(pipe):
    print(f"Sample: {i}")
    plot_idx = 1
    plt.figure(figsize=(14,6))
    for component_name, component_data in sample.items():
        if "CAM" in component_name:
            plt.subplot(1, 6, plot_idx)
            plt.title(component_name)
            plt.imshow(component_data.permute(1, 2, 0))
            plot_idx += 1
        else:
            print("# " + component_name)
            if isinstance(component_data, np.ndarray):
                print(component_data.shape)
            else:
                text = str(component_data)
                append = ""
                if len(text) > 400:
                    append = " ..."
                print(text[:400] + append)
    plt.show()
    if i >= 2:
        break
```

Output:
```
Sample: 0
# sample_token
c9304cf98aad4ff0bb5b0cff0aab65d2
# scene
scene-0655
# LIDAR_TOP
tensor([[ -3.0941,  -3.2652,  -3.4587,  ..., -11.2454, -11.2570, -11.2618],
        [ -0.4045,  -0.4026,  -0.4010,  ...,   0.0393,   0.0432,   0.0471],
        [ -1.8557,  -1.8554,  -1.8600,  ...,   1.5804,   1.8495,   2.1218],
        [  4.0000,   5.0000,   8.0000,  ...,  21.0000,  28.0000,  19.0000]])
# annotations
[{'token': '948ad541a70f4d0888a2e2eb8ac01db2', 'sample_token': 'c9304cf98aad4ff0bb5b0cff0aab65d2', 'instance_token': '493d6306f26f48b9bb128cb4f0a7976a', 'visibility_token': '4', 'attribute_tokens': ['58aa28b1c2a54dc88e169808c07331e3'], 'translation': [1797.224, 858.852, 1.21], 'size': [2.065, 4.893, 1.938], 'rotation': [0.7145703641621893, 0.0, 0.0, -0.6995635744241663], 'prev': '0286c2f07778461ca ...
Sample: 1
# sample_token
9768a199e4884a0db50d015703efe969
# scene
scene-0655
# LIDAR_TOP
tensor([[-3.0637e+00, -3.2117e+00, -3.4011e+00,  ..., -1.0881e+01,
         -1.0906e+01, -1.0912e+01],
        [-4.0635e-01, -4.0479e-01, -4.0360e-01,  ...,  2.0879e-02,
          2.4736e-02,  2.8586e-02],
        [-1.8413e+00, -1.8290e+00, -1.8329e+00,  ...,  1.5292e+00,
          1.7918e+00,  2.0559e+00],
        [ 4.0000e+00,  5.0000e+00,  7.0000e+00,  ...,  2.1000e+01,
          2.8000e+01,  2 ...
# annotations
[{'token': '425da3a4b25045d8aea52ea00b4e69c8', 'sample_token': '9768a199e4884a0db50d015703efe969', 'instance_token': '493d6306f26f48b9bb128cb4f0a7976a', 'visibility_token': '4', 'attribute_tokens': ['58aa28b1c2a54dc88e169808c07331e3'], 'translation': [1797.224, 858.84, 1.294], 'size': [2.065, 4.893, 1.938], 'rotation': [0.7145703641621893, 0.0, 0.0, -0.6995635744241663], 'prev': '6d5e4447aa3d4d6aa ...
Sample: 2
# sample_token
2ff86dc19c4644a1a88ce5ba848f56e5
# scene
scene-0061
# LIDAR_TOP
tensor([[-3.0383e+00, -3.1686e+00, -3.3133e+00,  ...,  9.1574e-06,
         -1.9163e+01,  9.1574e-06],
        [-1.8308e-01, -1.7995e-01, -1.7709e-01,  ...,  2.0849e-06,
         -1.0036e-02,  2.0849e-06],
        [-1.8129e+00, -1.7911e+00, -1.7723e+00,  ..., -2.0387e-07,
          3.1484e+00, -2.0387e-07],
        [ 1.8000e+01,  1.4000e+01,  1.6000e+01,  ...,  7.2000e+01,
          2.0000e+00,  1 ...
# annotations
[{'token': '7fa3a688931b4500b7ce29d187d3b975', 'sample_token': '2ff86dc19c4644a1a88ce5ba848f56e5', 'instance_token': '6dd2cbf4c24b4caeb625035869bca7b5', 'visibility_token': '4', 'attribute_tokens': ['4d8821270b4a47e3a8a300cbec48188e'], 'translation': [373.152, 1130.357, 1.25], 'size': [0.621, 0.669, 1.642], 'rotation': [0.9831098797903927, 0.0, 0.0, -0.18301629506281616], 'prev': '1e8e35d365a441a1 ...
```
![data](../../../docs/jlabdev_images/b142155c3db311f86974b14feadf14d2.png)
![data](../../../docs/jlabdev_images/6d60fdf7f9f5bee289ff1bfe18ed9afa.png)
![data](../../../docs/jlabdev_images/a37638f8826d339d673d3f244215ada0.png)

Benchmark again to see if the speed changed.

Example:
```python
import time
from tqdm import tqdm

start = time.time()
start1 = start
N = 200
FPS = []
for i, sample in enumerate(tqdm(pipe, desc="Benchmarking")):
    stop = time.time()
    FPS.append(1.0 / ((stop - start1)))
    start1 = stop
    if i >= N - 1:
        break
stop = time.time()

plt.plot(FPS)
plt.show()

FPS = 1.0 / ((stop - start) / N)
print(FPS)
```

Output:
```
Benchmarking:  62%|██████▏   | 199/323 [00:24<00:15,  8.08it/s]
8.122564290888109
```
![data](../../../docs/jlabdev_images/ff72cd939fede2f6227cf9bcb23bc8c9.png)

As we can see the injection of the annotations produces no additional load. Since they are stored in single database file which is loaded into RAM once.

