[Back to Overview](../README.md)

# datapipes.sequence_pipeline: Dataset Streaming for Sequence Data

> Automatic creation of data pipelines for streaming sharded datasets that have metadata. Each sample in the dataset
> is passed through a temporal sliding window filter to create fixed length overlapping slices of sequential samples.

## *class* **SequencePipelineCreator**(BasePipelineCreator) [[src]](../../datapipes/sequence_pipeline.py#L26)
The SequencePipelineCreator can be used to create a pipeline a specific dataset containing sequential data and to
query its metadata.

The interface is largely the same as for the BasePipelineCreator. In contrast to the BasePipelineCreator, the
SequencePipelineCreator will automatically wrap all components in a StateWrapper class. This is necessary to be
able to avoid repeating the same work multiple times on samples replicated by the temporal sliding window filter.
If a custom collate function is used, then the user has to explicitly remove the wrapper, by mapping the function
`remove_wrapper` onto the pipeline as the last pipeline operation.

### *def* **create_single_pipe** [[src]](../../datapipes/sequence_pipeline.py#L76)
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
 min_required_components or to raise an error (default behavior if min_required_components is set).
* **add_component_fn**: An optional function or callable class that adds additional components to a sample.
 The function will be called with three parameters: subset (data split), sequence id and sample id.
 It must return a list of tuples with the component ID and file extension as the first element and the
 component data as the second, e.g. ("depth.png", <image data>).
* **batch_size**: The number of samples that will be combined into a batch by the DataLoader. Used during
 pipeline length computation. You need to ensure that the number specified here, matches the batch size used
 during the actual batch creation.
* **kwargs**: Additional keyword arguments:
 Use `temporal_sliding_window_size` to specify how many temporally sequential frames shall be combined into
 one sample.
 Use `max_concurrent_sequences` to set the number of sequences from which to randomly pick samples. This is a
 more memory friendly way to shuffle sequence data. The shuffle_buffer_size is ignored in this case, since no
 shuffle buffer is used.
* **returns**: The data streaming base pipeline.

### *def* **create_zipped_pipe** [[src]](../../datapipes/sequence_pipeline.py#L127)
Creates a data streaming base pipeline for multiple shard streams that will be zipped together.

First a pipeline for each shard stream (one for each component group) is created. Then these separate pipelines
are zipped together to form a single pipeline. The order of the lists of tar files for each component group
needs to be the same for each stream.

* **subsets**: The subset of the dataset for which to create the data streaming pipeline.
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
 min_required_components or to raise an error (default behavior if min_required_components is set).
* **add_component_fn**: An optional function or callable class that adds additional components to a sample.
* **batch_size**: The number of samples that will be combined into a batch by the DataLoader. Used during
 pipeline length computation. You need to ensure that the number specified here, matches the batch size used
 during the actual batch creation.
* **kwargs**: Additional keyword arguments:
 Use `temporal_sliding_window_size` to specify how many temporally sequential frames shall be combined into
 one sample.
 Use `max_concurrent_sequences` to set the number of sequences from which to randomly pick samples. This is a
 more memory friendly way to shuffle sequence data. The shuffle_buffer_size is ignored in this case, since no
 shuffle buffer is used.
* **returns**: The data streaming base pipeline.
