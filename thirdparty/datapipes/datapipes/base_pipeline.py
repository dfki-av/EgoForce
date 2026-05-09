##############################################################################
# Copyright (c) 2022 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, October 2022
# and David Michael Fürst <david_michael.fuerst@dfki.de>, October 2022
##############################################################################
"""doc
# datapipes.base_pipeline: Dataset Streaming

> Automatic creation of data pipelines for streaming sharded datasets that have metadata.

"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from typing import Any, Callable

import torch.distributed as dist
from torch.utils.data.datapipes.datapipe import DataChunk, IterDataPipe
from torch.utils.data.datapipes.iter import FileOpener, IterableWrapper

from datapipes.iter.grouping import SequentialGrouperIterDataPipe
from datapipes.iter.pipe_length import TarShardingFilter, LengthLimiter
from datapipes.iter.util.combining import UnZipperIterDataPipe as Unzipper
from datapipes.iter.util.header import LengthSetterIterDataPipe as LengthSetter
from datapipes.iter.util.tararchiveloader import TarArchiveLoaderIterDataPipe as TarArchiveLoader
from datapipes.utils.dataset_path_utils import get_sample_id
from datapipes.utils.pipeline_helpers import ComponentAdder


def _merge_fn(sample: tuple) -> DataChunk:
    """ Merge a varying number of inputs (RGB, RGB256 and/or depth) with the annotations.

    :param sample: A tuple of the individual elements (from separate data pipes) that shall be merged.
    :return: The merged sample as a DataChunk (essentially a tuple).
    """
    # The data is stored in nested tuples. The outer tuple contains a tuple for each component group. The inner tuples
    # contain the components of each component group. We dissolve the inner tuples and integrate the elements into the
    # outer tuple.
    components = []
    for element in sample:
        if isinstance(element, DataChunk):
            components.extend([*element])
        else:
            components.append(element)
    # Convert to DataChunks, because this is what the outer pipeline operations expect.
    merged_sample = DataChunk(tuple(components))
    return merged_sample


class Filter:
    __slots__ = "keep_list"

    def __init__(self, components: list[str]):
        """ Creates a filter that removes all components not included in the list.

        :param components: The components that shall be kept.
        """
        self.keep_list = components

    def __call__(self, data: tuple[str, Any]) -> bool:
        file_path = data[0]
        key = os.path.basename(file_path).split(".")[-2]
        return key in self.keep_list


class BasePipelineCreator:

    def __init__(self,
                 dataset_dir: str,
                 metadata_filename: str = "metadata.json",
                 additional_dataset_dirs: list[str] = None):
        """ The BasePipelineCreator can be used to create a base pipeline for a specific dataset and to query its
        metadata.

        :param dataset_dir: The path to the directory containing the dataset.
        :param metadata_filename: The name of the file containing the metadata. Defaults to metadata.json
        :param additional_dataset_dirs: A list of paths to other folders containing extensions of the dataset.
            For example expensive pre-processing of labels for a model may be required. Then the data would be
            stored in additional shards in a single or multiple folders separate to the original dataset.
            Each folder in the list can add or replace shards of previous folder(s). The folders are increasingly
            specialized (think inheritance), thus the metadata from the last folder will be used.
        """
        self.dataset_dirs = [dataset_dir] + (additional_dataset_dirs or [])
        for data_dir in self.dataset_dirs:
            if not os.path.exists(data_dir):
                raise RuntimeError(f"Dataset directory does not exist: {data_dir}")
        metadata_path = os.path.join(self.dataset_dirs[-1], metadata_filename)
        if not os.path.exists(metadata_path):
            raise RuntimeError(f"Metadata does not exist: {metadata_path}")
        with open(metadata_path) as f:
            file_content: dict = json.loads(f.read())
            if "__meta__" in file_content and "version" in file_content["__meta__"]:
                if file_content["__meta__"]["version"] <= 2:
                    self.metadata_version = file_content["__meta__"]["version"]
                    del file_content["__meta__"]
                    self.metadata = file_content
                else:
                    raise ValueError("Unsupported metadata file format version. Supported: 1-2, "
                                     f"Supplied {file_content['__meta__']['version']}.")
            else:
                raise KeyError("File format version information not found in metadata file.")

    def get_subsets(self) -> list[str]:
        """ Lists the names of the subsets of the dataset.

        :return: A list of the names of the subsets of the dataset.
        """
        return list(self.metadata.keys())

    def get_sample_count(self, subset: str) -> int:
        """ Read the sample count for the specified subset from the metadata file.

        :param subset: The subset for which the number of samples shall be retrieved.
        :return: The number of samples contained in the specified subset.
        """
        return self.metadata[subset]["sample_count"]

    def get_shard_count(self, subset: str):
        """ Determines the number of shards in a subset.

        :param subset: The subset for which the number of shards shall be retrieved.
        :return: The number of shards that make up the specified subset.
        """
        return len(self.metadata[subset]["samples_per_shard"])

    def get_component_groups(self, subset: str) -> list[str]:
        """ Lists the component groups (one per tar stream) that the dataset is composed of.

        :param subset: The subset for which to query the component groups.
        :return: A list of the names of the component groups.
        """
        return list(self.metadata[subset]["component_groups"].keys())

    def get_component_groups_stats(self, subset: str) -> dict[str, dict[str, int | list[str]]]:
        """ Reads the information on components groups from the metadata file.

        :param subset: The subset for which to extract the component group information.
        :return: A dictionary containing the minimal and maximal number of components for each component group.
        """
        return self.metadata[subset]["component_groups"]

    def get_average_shard_sample_count(self, subset: str) -> int:
        """ Computes the number of samples that the shards of a specific subset hold on average.

        :param subset: The subset for which to compute the average shard sample count.
        :return: The average number of samples per shard in a specific subset rounded to an integer value.
        """
        metadata = self.metadata[subset]
        average_sample_count_per_shard = round(metadata["sample_count"] / len(metadata["samples_per_shard"]))
        return average_sample_count_per_shard

    def get_tar_files_for_subsets(self,
                                  subsets: str | list[str],
                                  component_groups: list[str] | None = None) -> dict[str, list[str]]:
        """ Search the file system for tar files matching the standard naming pattern for sharded datasets.

        :param subsets: The subsets for which to gather the tar shards.
        :param component_groups: The component groups that shall be included. They must be available in each requested
         subset.
        :return: A dictionary mapping the component group name to the tar shards that belong to the respective subsets
         and component groups.
        """
        if isinstance(subsets, str):
            subsets = [subsets]
        if component_groups is None:
            component_groups = set(self.get_component_groups(next(iter(subsets))))
        # Check whether the requested component groups exist in all requested subsets.
        for subset in subsets:
            assert set(component_groups).issubset(set(self.get_component_groups(subset)))
        component_groups = list(component_groups)
        tar_files = {}
        for cg in component_groups:
            # Search tars in all directories (from general to specific)
            for dataset_dir in self.dataset_dirs:
                files = []
                for subset in subsets:
                    if cg == "":
                        search_pattern = os.path.join(dataset_dir, subset, "*.tar")
                    else:
                        search_pattern = os.path.join(dataset_dir, subset, f"*.{cg}.tar")
                    files.extend(sorted(glob.glob(search_pattern)))
                if len(files) > 0:  # Only overwrite if tars are found.
                    tar_files[cg] = files
        return tar_files

    def create_datapipe(self,
                        subsets: str | list[str],
                        shuffle_buffer_size: int | None,
                        components: list[str] | None = None,
                        shuffle_shards: bool = True,
                        gpus: int | None = None,
                        max_samples: int | None = None,
                        min_required_components: int | dict[str, int] | None = None,
                        drop_incomplete_samples: bool = False,
                        add_component_fn: Callable[[str, str, str], list[tuple]] | None = None,
                        batch_size: int | None = None,
                        **kwargs) -> IterDataPipe:
        """ Creates a base pipeline from streaming datasets stored in tar shards.

        :param subsets: The subset(s) for which the pipeline shall be constructed.
        :param shuffle_buffer_size: The size of the shuffle buffer (for each worker). If possible a size should be
         chosen that is larger than the number of samples in a single shard.
         Ideally: shuffler_buffer_size = samples_per_shard * batch_size
        :param components: The components that shall be included. Components not listed will be discarded. The default
         value None will, however, result in all components being included.
        :param shuffle_shards: Whether to shuffle the shards for improved randomness. Only needed for training.
        :param gpus: The number of used GPUs. Deprecated and no longer used.
        :param max_samples: The maximum number of samples that shall be emitted by the pipeline per GPU. If set to None,
         no limit will be imposed. Defaults to None (no limit).
        :param min_required_components: The minimal number of components that you expect to see for each sample. If the
         data is stored in a single component group a single int may be specified. In case of multiple component groups,
         a dictionary mapping from component groups to ints may be specified. If neither of that or explicitly None is
         specified, no lower bound is set and checked on in the pipeline.
        :param drop_incomplete_samples: Whether to silently drop samples that do not have at least
         min_required_components components or to raise an error (default behavior if min_required_components is set).
        :param add_component_fn: An optional function or callable class that adds additional components to a sample.
         The function will be called with three parameters: subset (data split), sequence id and sample id.
         It must return a list of tuples with the component ID and file extension as the first element and the
         component data as the second, e.g. ("depth.png", <image data>).
        :param batch_size: The number of samples that will be combined into a batch by the DataLoader. Used during
         pipeline length computation. You need to ensure that the number specified here, matches the batch size used
         during the actual batch creation.
        :param kwargs: Additional keyword arguments that may be used in subclasses.
        :return: The dataset as an IterDataPipe.
        """
        if isinstance(subsets, str):
            subsets = [subsets]
        for subset in subsets:
            if subset not in self.get_subsets():
                raise RuntimeError(f"The dataset does not have a subset named '{subset}'.")
        if shuffle_buffer_size is None:
            shuffle_buffer_size = 0

        if components is not None:
            component_group_filter = defaultdict(list)
            found_components = []

            # Iterate over the components to maintain their order
            for component in components:
                # For each component, find its group in subsets[0]
                for group in sorted(self.get_component_groups(subsets[0])):
                    group_stats = self.get_component_groups_stats(subsets[0])
                    if component in group_stats[group]["all_components"]:
                        component_group_filter[group].append(component)
                        found_components.append(component)
                        # break  # Once the component is found, exit the group loop; This logic is not implemented in the original code

            # Check for missing components
            if len(found_components) != len(components):
                if len(found_components) < len(components):
                    missing = [x for x in components if x not in found_components]
                    raise RuntimeError(f"The following requested sample components do not exist: {missing}")
                else:
                    raise RuntimeError(f"The requested components, {components}, are not unique across the datapipe's component groups, i.e., {found_components}.")
            
            tar_files_per_cg = self.get_tar_files_for_subsets(subsets, list(component_group_filter.keys()))
        else:
            # use all available component groups
            component_group_filter = None
            tar_files_per_cg = self.get_tar_files_for_subsets(subsets)

        if len(tar_files_per_cg) == 1:
            pipe = self.create_single_pipe(subsets, tar_files_per_cg, shuffle_buffer_size, shuffle_shards,
                                           component_group_filter, max_samples, min_required_components,
                                           drop_incomplete_samples, add_component_fn, batch_size, **kwargs)
        else:
            pipe = self.create_zipped_pipe(subsets, tar_files_per_cg, shuffle_buffer_size, shuffle_shards,
                                           component_group_filter, max_samples, min_required_components,
                                           drop_incomplete_samples, add_component_fn, batch_size, **kwargs)
        # set length of pipeline if it can be computed statically
        if not dist.is_initialized() or dist.get_world_size() == 1:
            pipe_length = 0
            if ("temporal_sliding_window_size" in kwargs and
                    kwargs["temporal_sliding_window_size"] is not None and
                    kwargs["temporal_sliding_window_size"] > 1):
                window_size = kwargs["temporal_sliding_window_size"]
                for subset in subsets:
                    for shard in self.metadata[subset]["sequence_lengths_per_shard"].values():
                        for sequence_len in shard:
                            pipe_length += max(sequence_len - window_size + 1, 0)
            else:
                for subset in subsets:
                    pipe_length += self.get_sample_count(subset)
            pipe = LengthSetter(pipe, pipe_length)
        pipe = LengthLimiter(pipe, batch_size, max_samples)
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
         min_required_components components or to raise an error (default behavior if min_required_components is set).
        :param add_component_fn: An optional function or callable class that adds additional components to a sample.
        :param batch_size: The number of samples that will be combined into a batch by the DataLoader. Used during
         pipeline length computation. You need to ensure that the number specified here, matches the batch size used
         during the actual batch creation.
        :param kwargs: Additional keyword arguments that may be used in subclasses.
        :return: The data streaming base pipeline.
        """
        tar_files = list(tar_files_per_cg.values())[0]
        num_shards = len(tar_files)
        window_size = kwargs["temporal_sliding_window_size"] if "temporal_sliding_window_size" in kwargs else None
        # Create an IterDataPipe which wraps this iterable sequence.
        pipe = IterableWrapper(tar_files)
        if num_shards > 1:
            if shuffle_shards:
                # Shuffle the order of the tar files. This is necessary to get good randomness during training.
                pipe = pipe.shuffle(buffer_size=len(tar_files))
            # Distribute work on the level of shards: each worker reads a distinct subset of tar archives.
            pipe = TarShardingFilter(pipe, self.metadata, self.metadata_version, window_size)
        # Open tar file for reading and load files from tar archive.
        pipe = FileOpener(pipe, mode="b")
        pipe = TarArchiveLoader(pipe)
        # Filter components, keep only those in the list.
        if component_group_filter is not None:
            components = next(iter(component_group_filter.values()))
            pipe = pipe.filter(Filter(components))
            max_components = len(components)
        else:
            # Determine the maximum number of components across the used subsets
            cg_name = list(tar_files_per_cg.keys())[0]
            component_count = []
            for subset in subsets:
                component_count.append(self.get_component_groups_stats(subset)[cg_name]["max_components"])
            max_components = max(component_count)
        # Group files together which belong to the same sample.
        pipe = SequentialGrouperIterDataPipe(pipe, group_key_fn=get_sample_id, buffer_size=max_components,
                                             guaranteed_group_size=min_required_components,
                                             drop_incomplete=drop_incomplete_samples)
        # Shuffle the files in memory.
        # Requires memory according to: num_workers * shuffle_buffer_size * size_of_a_sample.
        # Since this is done before decoding, the memory consumption is determined by file size not the decoded size.
        if shuffle_buffer_size > 1:
            pipe = pipe.shuffle(buffer_size=shuffle_buffer_size)
        if add_component_fn is not None:
            pipe = pipe.map(ComponentAdder(add_component_fn))
        if num_shards == 1 and ("no_sharding" not in kwargs or kwargs["no_sharding"] is False):
            # Since we have only one shard, we need to distribute the work on the level of samples.
            # It is not efficient and does not scale well, but it is the best we can do in this case.
            # The single shard will be read once by each worker but each uses a separate subset of samples.
            pipe = pipe.sharding_filter()
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

        :param subsets: The subsets of the dataset for which to create the data streaming pipeline.
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
         min_required_components components or to raise an error (default behavior if min_required_components is set).
        :param add_component_fn: An optional function or callable class that adds additional components to a sample.
        :param batch_size: The number of samples that will be combined into a batch by the DataLoader. Used during
         pipeline length computation. You need to ensure that the number specified here, matches the batch size used
         during the actual batch creation.
        :param kwargs: Additional keyword arguments that may be used in subclasses.
        :return: The data streaming base pipeline.
        """
        pipeline_count = len(tar_files_per_cg)
        zipped_shard_list = list(zip(*tar_files_per_cg.values()))
        num_shards = len(zipped_shard_list)
        # Wrap the zipped shard sequence to create an iterable pipeline
        pipe = IterableWrapper(zipped_shard_list)
        window_size = kwargs["temporal_sliding_window_size"] if "temporal_sliding_window_size" in kwargs else None
        if num_shards > 1:
            if shuffle_shards:
                # Shuffle the order of the tar files. This is necessary to get good randomness during training.
                pipe = pipe.shuffle(buffer_size=len(zipped_shard_list))
            # Distribute work on the level of shards: each worker reads a distinct subset of tar archives.
            pipe = TarShardingFilter(pipe, self.metadata, self.metadata_version, window_size)
        # unzip the tar sequence to create separate pipelines for the individual tar sequences
        pipes = Unzipper(pipe, sequence_length=pipeline_count, buffer_size=pipeline_count)
        # open and read the tar files in each sequence separately
        pipes = [TarArchiveLoader(FileOpener(pipe, mode="b")) for pipe in pipes]
        # Filter components. Keep only those in the list.
        if component_group_filter is not None:
            for idx, component_filter in enumerate(component_group_filter.values()):
                pipes[idx] = pipes[idx].filter(Filter(component_filter))
        # Determine the maximum number of components per component group
        max_components = dict()
        for idx, cg_name in enumerate(tar_files_per_cg.keys()):
            for subset in subsets:
                if component_group_filter is not None:
                    max_c = len(component_group_filter[cg_name])
                else:
                    max_c = self.get_component_groups_stats(subset)[cg_name]['max_components']
                max_components[cg_name] = max(max_components[cg_name], max_c) if cg_name in max_components else max_c
        for idx, cg_name in enumerate(tar_files_per_cg.keys()):
            # Group per pipe, so we advance the pipes at the correct individual speed
            if max_components[cg_name] > 1:
                min_components = min_required_components[cg_name] if min_required_components is not None else None
                pipes[idx] = SequentialGrouperIterDataPipe(pipes[idx], group_key_fn=get_sample_id,
                                                           buffer_size=max_components[cg_name],
                                                           guaranteed_group_size=min_components,
                                                           drop_incomplete=drop_incomplete_samples)
        # merge the pipelines back together
        pipe = pipes[0].zip(*pipes[1:])
        # fuse the individual tuples
        pipe = pipe.map(_merge_fn)
        # Shuffle the files in memory.
        # Requires memory according to: num_workers * shuffle_buffer_size * size_of_a_sample.
        # Since this is done before decoding, the memory consumption is determined by file size not the decoded size.
        if shuffle_buffer_size > 1:
            pipe = pipe.shuffle(buffer_size=shuffle_buffer_size)
        if add_component_fn is not None:
            pipe = pipe.map(ComponentAdder(add_component_fn))
        if num_shards == 1 and ("no_sharding" not in kwargs or kwargs["no_sharding"] is False):
            # Since we have only one shard, we need to distribute the work on the level of samples.
            # It is not efficient and does not scale well, but it is the best we can do in this case.
            # The single shard will be read once by each worker but each uses a separate subset of samples.
            pipe = pipe.sharding_filter()
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

    creator = BasePipelineCreator(args.dataset_dir)
    all_subsets = creator.get_subsets()
    print("subsets:", all_subsets)
    shard_size = creator.get_average_shard_sample_count(all_subsets[0])
    batch_size = 1
    base_pipe = creator.create_datapipe(all_subsets[0], shard_size * batch_size, max_samples=2)

    for my_sample in base_pipe:
        print("sample:", my_sample)


if __name__ == '__main__':
    _test()
