# DataPipes

This project contains tools to
- convert datasets into a set of sharded tar archives,
- provide index-based access to datasets that fit into memory, and
- stream datasets that do not fit into memory. 

Additionally, templates and example scripts are also available.

> documentation of API and examples: [docs/README.md](docs/README.md)
> pyTorch backend: https://github.com/pytorch/pytorch/tree/main/torch/utils/data/datapipes

## Requirements and Installation
Datapipes requires pyTorch 1.13.0 or later.
TorchData is no longer required.
To install datapipes, you need to clone the repository and install it with pip.
You have two options how to do that:

#### Installation as an independent package

You can clone and install datapipes as an independent package like this:

```bash
git clone git@pc-4501.kl.dfki.de:Common/datapipes.git
pip install -e datapipes
```

#### Installation as a git submodule

It is, however, recommended to add it to your project as a git submodule.
This allows you to tie the datapipes version that you are using to the version of your code.
This way switching to a newer version is explicit and recorded by git.
This simplifies things when the repo needs to be cloned later or when trying to run an old version of your code.
Git can then also get you the appropriate version of datapipes.

```bash
git submodule add git@pc-4501.kl.dfki.de:Common/datapipes.git
pip install -e datapipes
```

## Preprocessing Files

If you need to preprocess files in the dataset before sharding, you can use the `ArchiveProcessor`.
It allows you to process files stored in ZIP or TAR archives without extracting them to disk.
The processed files will be written to new ZIP archives.
These archives can then be used in the process of converting the dataset into sharded form or used directly by datapipeFS.
Typical use cases include changing the resolution of images and converting data or file formats.

## Dataset Conversion

To read datasets efficiently, they need to be sharded.
Datasets can be sharded with a single function call. You have two main options that are described below.

### Option 1: Call convert_samples()

This is the right choice for people who wish to read as little as possible and in turn are willing to write more code themselves.
It makes several assumptions to simplify the process which limit how much it can be customized.
The assumption with the biggest impact is: "each sample is independent".
This implies that it cannot be used for datasets where samples need to keep their order within a sequence. 

The interface of this conversion function is as follows:

```python
from datapipes.dataset_converter import convert_samples
convert_samples(
    in_dir,                         # the input path to the directory that contains the original dataset
    out_dir,                        # the output path for the sharded dataset
    samples,                        # the samples for each split of the dataset (see below)
    target_shard_size_mb = 400,     # the target shard size in MiB, defaults to 400 MiB
    splits_to_pre_shuffle = None)   # the data splits to pre-shuffle, typically ["train"], defaults to all
```

You need to provide the first three parameters. The remaining ones have default values and hence are optional.
For `samples` provide a dictionary that maps from the name of a split to the list of samples, e.g:

```python
samples = {"train": train_samples,
           "val": val_samples,
           "test": test_samples}
```

For each list of samples provide a dictionary that maps from the name of a sample component to the path of the file
in which it is stored, e.g.:

```python
train_samples = []
for i in range(1000):
    train_samples.append({"rgb": f"train/images/frame_{i:04d}.jpg",
                          "depth": f"train/depth/frame_{i:04d}.png",
                          "label": f"train/classes/label_{i:04d}.txt"})
```

Paths should be relative to the input directory.
They may point inside a zip archive like so: `train.zip/images/frame_0000.jpg`.
Nested archives are not supported, however.
For more information check the documentation in `docs` or the code in `dataset_converter.py` . 

### Option 2: Call convert_dataset()

This is the right choice for people who would like to minimize the amount of code they need to write and in turn
are willing to read up on how to write the `path_to_file_info_converter` function.
This option also provides more options for customization and supports sequence datasets.

```python
from datapipes.dataset_converter import convert_dataset
convert_dataset(
    in_dir,                                 # the input path to the directory that contains the original dataset
    out_dir,                                # the output path for the sharded dataset
    path_to_file_info_converter,            # a function that provides information on each file that shall be included in the dataset
    component_grouping = None,              # defines which sample components will be stored together; defaults to all
    target_tar_file_size = 400,             # target shard size in MiB, defaults to 400 MiB
    target_gpu_count = 8,                   # how many GPUs to support for parallel training, the default 8 gives support for 1, 2, 4 and 8
    splits_to_pre_shuffle = None,           # the data splits to pre-shuffle, typically ["train"], defaults to all
    path_filter = lambda x: True,           # a filter function that defines which directories to include.
    preserve_sequential_ordering = False)   # whether to preserve the ordering of samples within each sequence. 
```

You need to provide the first three parameters.
The remaining ones have default values and are therefore optional.
The main task of the `path_to_file_info_converter` is to tell for each file to which data split,
sequence and sample each file belongs and which component it contains.
For details on how to write the `path_to_file_info_converter` function check the template
in `templates/dataset_conversion.py` and the examples in `examples`.
A look at the documentation in `docs` or the code in `dataset_converter.py` may also be helpful.

## Accessing Datasets

To make use of the tools for accessing datasets provided here, the dataset needs to be sharded first.
If the dataset has not been converted yet, the dataset converter can be used to create a sharded version of the dataset.
Small datasets that fit into memory can then be accessed via the InMemoryDataset.
It provides index-based, random access to the samples.
Datasets that do not fit into memory can be accessed via streaming pipelines.
They provide an iterable sequence of samples.

### InMemoryDataset

For small datasets that fit entirely into memory the InMemoryDatset class can be used.
It preloads a sharded dataset into main memory (RAM) and provides index-based random access to its samples.
The sample components (files) can be automatically decoded on access if a decoder map is provided.
The InMemoryDataset supports multiple GPUs on a single node with multiple DataLoader workers without duplicating the dataset in memory.
The folder `templates` contains a template for how to integrate it into your own code.
Datasets that are too large to fit into memory can be accessed via streaming pipelines as described in the next section. 

### Streaming pipelines

For datasets that do not fit into memory, you can make use of the BasePipelineCreator to create a basic pipeline for
streaming the samples of the dataset.
If you need a fixed number of consecutive frames for training, you can use the SequencePipelineCreator.
All pipelines can be extended with additional operations like decoding and data augmentation.
You can use `base_pipeline.py` and `sequence_pipeline.py` in the `templates` directory as a basis depending on what
kind of pipeline you need.
Just copy the relevant code pieces to your project and extend/modify them as needed.

## Examples

There are a couple of examples that show how to shard and load datasets.
They can be found in the `examples` folder.
A minimal example can be found in the `FaceSynthetics` folder.

A good starting point to dive deeper is the UnrealEgo example.
It shows how to:
- pre-process files stored in archives,
- convert the dataset into sharded form,
- stream the sharded dataset and decode the samples on the fly, e.g. during training,
- or, alternatively, to load the dataset into main memory and decode the samples on access.