##############################################################################
# Copyright (c) 2022 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, December 2022
##############################################################################
"""doc
# datapipes.sequence_pipeline: Dataset Streaming for Sequence Data

> Automatic creation of data pipelines for streaming sharded datasets that have metadata. Each sample in the dataset
> is passed through a temporal sliding window filter to create fixed length overlapping slices of sequential samples.

"""
from __future__ import annotations

from typing import Callable

from torch.utils.data.datapipes.datapipe import IterDataPipe

from datapipes.base_pipeline import BasePipelineCreator
from datapipes.iter.pipe_length import LengthLimiter, SequenceShardingFilter
from datapipes.iter.seq_aware_shuffler import SequenceAwareShuffler
from datapipes.iter.stream_reader import StreamReaderIterDataPipe
from datapipes.iter.sliding_window import SlidingWindowIterDataPipe
from datapipes.utils.pipeline_helpers import WrapperAdder


class SequencePipelineCreator(BasePipelineCreator):
    """
    The SequencePipelineCreator can be used to create a pipeline a specific dataset containing sequential data and to
    query its metadata.

    The interface is largely the same as for the BasePipelineCreator. In contrast to the BasePipelineCreator, the
    SequencePipelineCreator will automatically wrap all components in a StateWrapper class. This is necessary to be
    able to avoid repeating the same work multiple times on samples replicated by the temporal sliding window filter.
    If a custom collate function is used, then the user has to explicitly remove the wrapper, by mapping the function
    `remove_wrapper` onto the pipeline as the last pipeline operation.
    """

    def __init__(self,
                 dataset_dir: str,
                 metadata_filename: str = "metadata.json",
                 additional_dataset_dirs: list[str] = None):
        """ The SequencePipelineCreator can be used to create a pipeline a specific dataset containing sequential data
        and to query its metadata. The interface is largely the same as for the BasePipelineCreator.

        :param dataset_dir: The path to the directory containing the dataset.
        :param metadata_filename: The name of the file containing the metadata. Defaults to metadata.json
        :param additional_dataset_dirs: A list of paths to other folders containing extensions of the dataset.
            For example expensive pre-processing of labels for a model may be required. Then the data would be
            stored in additional shards in a single or multiple folders separate to the original dataset.
            Each folder in the list can add or replace shards of previous folder(s). The folders are increasingly
            specialized (think inheritance), thus the metadata from the last folder will be used.
        """
        super().__init__(dataset_dir, metadata_filename, additional_dataset_dirs)
        if self.metadata_version < 2:
            print("Warning: Metadata format v2 required to automatically prevent GPU stalling in multi-GPU scenarios.")
            print("You manually need to ensure that all GPUs get the same number of samples!")
        for subset in self.get_subsets():
            if self.metadata[subset]["pre-shuffled"] != "no":
                print(f"Error: The {subset} subset has been pre-shuffled! Temporal sliding window creation will fail.")

    def _add_temporal_windowing(self, pipe: IterDataPipe, shuffle_buffer_size: int, **kwargs):
        if kwargs["shard_sequences"]:
            # distribute entire sequences to workers
            pipe = SequenceShardingFilter(pipe, self.metadata, self.metadata_version, kwargs["temporal_sliding_window_size"])
        pipe = StreamReaderIterDataPipe(pipe)
        pipe = pipe.map(WrapperAdder("encoded"))
        temporal_sliding_window_size = kwargs["temporal_sliding_window_size"]
        max_concurrent_sequences = kwargs["max_concurrent_sequences"] if "max_concurrent_sequences" in kwargs else None
        pipe = SlidingWindowIterDataPipe(pipe, temporal_sliding_window_size)
        if max_concurrent_sequences is not None:
            pipe = SequenceAwareShuffler(pipe, max_concurrent_sequences)
        elif shuffle_buffer_size > 1:
            pipe = pipe.shuffle(buffer_size=shuffle_buffer_size)
        return pipe

    def create_single_pipe(self,
                           subsets: list[str],
                           tar_files_per_cg: dict[str, list[str]],
                           shuffle_buffer_size: int,
                           shuffle_shards: bool,
                           component_group_filter: dict[str, list[str]] | None,
                           max_samples: int | None = None,
                           min_required_components: int | None = None,
                           drop_incomplete_samples: bool = False,
                           add_component_fn: Callable[[str, str, str], list[tuple]] | None = None,
                           batch_size: int | None = None,
                           **kwargs) -> IterDataPipe:
        """ Create a pipeline for a single shard stream.

        :param subsets: The subsets of the dataset for which to create the data streaming pipeline.
        :param tar_files_per_cg: A dictionary mapping the component group names to the corresponding tar files of the
         specified subset. Should contain only one entry (additional entries are ignored).
        :param shuffle_buffer_size: The number of samples that the shuffle buffer shall hold.
        :param shuffle_shards: Whether to shuffle the shards. Should be set to True for training and False for
         evaluation subsets.
        :param component_group_filter: A dictionary that lists the components that shall be included for each component
         group.
        :param max_samples: The maximum number of samples that shall be emitted by the pipeline per GPU. If set to None,
         no limit will be imposed. Defaults to None (no limit).
        :param min_required_components: The minimal number of components that you expect to see for each sample.
        :param drop_incomplete_samples: Whether to silently drop samples that do not have at least
         min_required_components or to raise an error (default behavior if min_required_components is set).
        :param add_component_fn: An optional function or callable class that adds additional components to a sample.
         The function will be called with three parameters: subset (data split), sequence id and sample id.
         It must return a list of tuples with the component ID and file extension as the first element and the
         component data as the second, e.g. ("depth.png", <image data>).
        :param batch_size: The number of samples that will be combined into a batch by the DataLoader. Used during
         pipeline length computation. You need to ensure that the number specified here, matches the batch size used
         during the actual batch creation.
        :param kwargs: Additional keyword arguments:
         Use `temporal_sliding_window_size` to specify how many temporally sequential frames shall be combined into
         one sample.
         Use `max_concurrent_sequences` to set the number of sequences from which to randomly pick samples. This is a
         more memory friendly way to shuffle sequence data. The shuffle_buffer_size is ignored in this case, since no
         shuffle buffer is used.
        :return: The data streaming base pipeline.
        """
        single_tar_handling = len(list(tar_files_per_cg.values())[0]) == 1
        pipe = super().create_single_pipe(subsets, tar_files_per_cg, 0, shuffle_shards, component_group_filter,
                                          None, min_required_components, drop_incomplete_samples,
                                          add_component_fn, no_sharding=single_tar_handling, **kwargs)
        kwargs["shard_sequences"] = single_tar_handling
        pipe = self._add_temporal_windowing(pipe, shuffle_buffer_size, **kwargs)
        pipe = LengthLimiter(pipe, batch_size, max_samples)
        return pipe

    def create_zipped_pipe(self,
                           subsets: list[str],
                           tar_files_per_cg: dict[str, list[str]],
                           shuffle_buffer_size: int,
                           shuffle_shards: bool,
                           component_group_filter: dict[str, list[str]] | None,
                           max_samples: int | None = None,
                           min_required_components: dict[str, int] | None = None,
                           drop_incomplete_samples: bool = False,
                           add_component_fn: Callable[[str, str, str], list[tuple]] | None = None,
                           batch_size: int | None = None,
                           **kwargs) -> IterDataPipe:
        """ Creates a data streaming base pipeline for multiple shard streams that will be zipped together.

        First a pipeline for each shard stream (one for each component group) is created. Then these separate pipelines
        are zipped together to form a single pipeline. The order of the lists of tar files for each component group
        needs to be the same for each stream.

        :param subsets: The subset of the dataset for which to create the data streaming pipeline.
        :param tar_files_per_cg: A dictionary mapping the component group names to the corresponding tar files of the
         specified subset.
        :param shuffle_buffer_size: The number of samples that the shuffle buffer shall hold.
        :param shuffle_shards: Whether to shuffle the shards. Should be set to True for training and False for
         evaluation subsets.
        :param component_group_filter: A dictionary that lists the components that shall be included for each component
         group.
        :param max_samples: The maximum number of samples that shall be emitted by the pipeline per GPU. If set to None,
         no limit will be imposed. Defaults to None (no limit).
        :param min_required_components: The minimal number of components for each component group that you expect to see
         for each sample.
        :param drop_incomplete_samples: Whether to silently drop samples that do not have at least
         min_required_components or to raise an error (default behavior if min_required_components is set).
        :param add_component_fn: An optional function or callable class that adds additional components to a sample.
        :param batch_size: The number of samples that will be combined into a batch by the DataLoader. Used during
         pipeline length computation. You need to ensure that the number specified here, matches the batch size used
         during the actual batch creation.
        :param kwargs: Additional keyword arguments:
         Use `temporal_sliding_window_size` to specify how many temporally sequential frames shall be combined into
         one sample.
         Use `max_concurrent_sequences` to set the number of sequences from which to randomly pick samples. This is a
         more memory friendly way to shuffle sequence data. The shuffle_buffer_size is ignored in this case, since no
         shuffle buffer is used.
        :return: The data streaming base pipeline.
        """
        single_tar_handling = len(list(tar_files_per_cg.values())[0]) == 1
        pipe = super().create_zipped_pipe(subsets, tar_files_per_cg, 0, shuffle_shards, component_group_filter,
                                          None, min_required_components, drop_incomplete_samples,
                                          add_component_fn, no_sharding=single_tar_handling, **kwargs)
        kwargs["shard_sequences"] = single_tar_handling
        pipe = self._add_temporal_windowing(pipe, shuffle_buffer_size, **kwargs)
        pipe = LengthLimiter(pipe, batch_size, max_samples)
        return pipe


def _test():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset-dir",
                        dest="dataset_dir",
                        type=str,
                        required=True,
                        help="The path to the directory containing the dataset.")
    args = parser.parse_args()

    creator = SequencePipelineCreator(args.dataset_dir)
    all_subsets = creator.get_subsets()
    print("subsets:", all_subsets)

    base_pipe = creator.create_datapipe(all_subsets[1], 0, max_samples=2, temporal_sliding_window_size=3)

    for samples in base_pipe:
        paths = [component[0] for sample in samples for component in sample]
        print("sample:", paths)


if __name__ == '__main__':
    _test()
