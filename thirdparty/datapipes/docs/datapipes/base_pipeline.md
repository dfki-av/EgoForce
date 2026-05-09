[Back to Overview](../README.md)

# datapipes.base_pipeline: Dataset Streaming

> Automatic creation of data pipelines for streaming sharded datasets that have metadata.

## *class* **Filter** [[src]](../../datapipes/base_pipeline.py#L52)
Creates a filter that removes all components not included in the list.

* **components**: The components that shall be kept.

## *class* **BasePipelineCreator** [[src]](../../datapipes/base_pipeline.py#L68)
The BasePipelineCreator can be used to create a base pipeline for a specific dataset and to query its
metadata.

* **dataset_dir**: The path to the directory containing the dataset.
* **metadata_filename**: The name of the file containing the metadata. Defaults to metadata.json
* **additional_dataset_dirs**: A list of paths to other folders containing extensions of the dataset.
    For example expensive pre-processing of labels for a model may be required. Then the data would be
    stored in additional shards in a single or multiple folders separate to the original dataset.
    Each folder in the list can add or replace shards of previous folder(s). The folders are increasingly
    specialized (think inheritance), thus the metadata from the last folder will be used.

### *def* **get_subsets** [[src]](../../datapipes/base_pipeline.py#L105)
Lists the names of the subsets of the dataset.

* **returns**: A list of the names of the subsets of the dataset.

### *def* **get_sample_count** [[src]](../../datapipes/base_pipeline.py#L112)
Read the sample count for the specified subset from the metadata file.

* **subset**: The subset for which the number of samples shall be retrieved.
* **returns**: The number of samples contained in the specified subset.

### *def* **get_shard_count** [[src]](../../datapipes/base_pipeline.py#L120)
Determines the number of shards in a subset.

* **subset**: The subset for which the number of shards shall be retrieved.
* **returns**: The number of shards that make up the specified subset.

### *def* **get_component_groups** [[src]](../../datapipes/base_pipeline.py#L128)
Lists the component groups (one per tar stream) that the dataset is composed of.

* **subset**: The subset for which to query the component groups.
* **returns**: A list of the names of the component groups.

### *def* **get_component_groups_stats** [[src]](../../datapipes/base_pipeline.py#L136)
Reads the information on components groups from the metadata file.

* **subset**: The subset for which to extract the component group information.
* **returns**: A dictionary containing the minimal and maximal number of components for each component group.

### *def* **get_average_shard_sample_count** [[src]](../../datapipes/base_pipeline.py#L144)
Computes the number of samples that the shards of a specific subset hold on average.

* **subset**: The subset for which to compute the average shard sample count.
* **returns**: The average number of samples per shard in a specific subset rounded to an integer value.

### *def* **get_tar_files_for_subsets** [[src]](../../datapipes/base_pipeline.py#L154)
Search the file system for tar files matching the standard naming pattern for sharded datasets.

* **subsets**: The subsets for which to gather the tar shards.
* **component_groups**: The component groups that shall be included. They must be available in each requested
 subset.
* **returns**: A dictionary mapping the component group name to the tar shards that belong to the respective subsets
 and component groups.

### *def* **create_datapipe** [[src]](../../datapipes/base_pipeline.py#L188)
Creates a base pipeline from streaming datasets stored in tar shards.

* **subsets**: The subset(s) for which the pipeline shall be constructed.
* **shuffle_buffer_size**: The size of the shuffle buffer (for each worker). If possible a size should be
 chosen that is larger than the number of samples in a single shard.
 Ideally: shuffler_buffer_size = samples_per_shard * batch_size
* **components**: The components that shall be included. Components not listed will be discarded. The default
 value None will, however, result in all components being included.
* **shuffle_shards**: Whether to shuffle the shards for improved randomness. Only needed for training.
* **gpus**: The number of used GPUs. Deprecated and no longer used.
* **max_samples**: The maximum number of samples that shall be emitted by the pipeline per GPU. If set to None,
 no limit will be imposed. Defaults to None (no limit).
* **min_required_components**: The minimal number of components that you expect to see for each sample. If the
 data is stored in a single component group a single int may be specified. In case of multiple component groups,
 a dictionary mapping from component groups to ints may be specified. If neither of that or explicitly None is
 specified, no lower bound is set and checked on in the pipeline.
* **drop_incomplete_samples**: Whether to silently drop samples that do not have at least
 min_required_components components or to raise an error (default behavior if min_required_components is set).
* **add_component_fn**: An optional function or callable class that adds additional components to a sample.
 The function will be called with three parameters: subset (data split), sequence id and sample id.
 It must return a list of tuples with the component ID and file extension as the first element and the
 component data as the second, e.g. ("depth.png", <image data>).
* **batch_size**: The number of samples that will be combined into a batch by the DataLoader. Used during
 pipeline length computation. You need to ensure that the number specified here, matches the batch size used
 during the actual batch creation.
* **kwargs**: Additional keyword arguments that may be used in subclasses.
* **returns**: The dataset as an IterDataPipe.

### *def* **create_single_pipe** [[src]](../../datapipes/base_pipeline.py#L281)
Create a pipeline for a single shard stream.

* **subsets**: The subsets of the dataset for which to create the data streaming pipeline.
* **tar_files_per_cg**: A dictionary mapping the component group names to the corresponding tar files of the
 specified subset. Should contain only one entry (additional entries are ignored).
* **shuffle_buffer_size**: The number of samples that the shuffle buffer shall hold.
* **shuffle_shards**: Whether to shuffle the shards. Should be set to True for training and False for
 evaluation subsets.
* **component_group_filter**: A dictionary that lists the components that shall be included for each component
 group.
* **max_samples**: The maximum number of samples that shall be emitted by the pipeline per GPU. If set to None,
 no limit will be imposed. Defaults to None (no limit).
* **min_required_components**: The minimal number of components that you expect to see for each sample.
* **drop_incomplete_samples**: Whether to silently drop samples that do not have at least
 min_required_components components or to raise an error (default behavior if min_required_components is set).
* **add_component_fn**: An optional function or callable class that adds additional components to a sample.
* **batch_size**: The number of samples that will be combined into a batch by the DataLoader. Used during
 pipeline length computation. You need to ensure that the number specified here, matches the batch size used
 during the actual batch creation.
* **kwargs**: Additional keyword arguments that may be used in subclasses.
* **returns**: The data streaming base pipeline.

### *def* **create_zipped_pipe** [[src]](../../datapipes/base_pipeline.py#L359)
Creates a data streaming base pipeline for multiple shard streams that will be zipped together.

First a pipeline for each shard stream (one for each component group) is created. Then these separate pipelines
are zipped together to form a single pipeline. The order of the lists of tar files for each component group
needs to be the same for each stream.

* **subsets**: The subsets of the dataset for which to create the data streaming pipeline.
* **tar_files_per_cg**: A dictionary mapping the component group names to the corresponding tar files of the
 specified subset.
* **shuffle_buffer_size**: The number of samples that the shuffle buffer shall hold.
* **shuffle_shards**: Whether to shuffle the shards. Should be set to True for training and False for
 evaluation subsets.
* **component_group_filter**: A dictionary that lists the components that shall be included for each component
 group.
* **max_samples**: The maximum number of samples that shall be emitted by the pipeline per GPU. If set to None,
 no limit will be imposed. Defaults to None (no limit).
* **min_required_components**: The minimal number of components for each component group that you expect to see
 for each sample.
* **drop_incomplete_samples**: Whether to silently drop samples that do not have at least
 min_required_components components or to raise an error (default behavior if min_required_components is set).
* **add_component_fn**: An optional function or callable class that adds additional components to a sample.
* **batch_size**: The number of samples that will be combined into a batch by the DataLoader. Used during
 pipeline length computation. You need to ensure that the number specified here, matches the batch size used
 during the actual batch creation.
* **kwargs**: Additional keyword arguments that may be used in subclasses.
* **returns**: The data streaming base pipeline.
