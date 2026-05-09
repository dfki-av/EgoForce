##############################################################################
# Copyright (c) 2022 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, October 2022
##############################################################################
"""doc
# datapipes.utils.collation_functions

> Functions for collating multiple samples into batches. The individual components are provided in tuples or dicts.

"""
from __future__ import annotations

from functools import reduce

import numpy as np
import torch

from torch.utils.data.datapipes.datapipe import DataChunk

from datapipes.utils.dataset_path_utils import get_component_name
from datapipes.utils.state_wrapper import StateWrapper


def _check_shapes(arrays: list[np.ndarray | torch.Tensor]):
    shapes_match = True
    for idx in range(1, len(arrays)):
        shapes_match = shapes_match and (arrays[0].shape == arrays[idx].shape)
    return shapes_match


def collate_batch_as_tuples(samples: list) -> tuple[tuple, tuple]:
    """ Collates individual samples (stored in a list of tuples) into batches.

    Combines corresponding elements of individual samples. Tensors and numpy arrays will be stacked in the batch
    dimension. Other data types are simply placed in lists. All aggregated components are then placed in a tuple in the
    same order as they appear in the sample tuples. Additionally, a tuple with the file paths for each file in the
    batch is returned as well.

    :param samples: A list of the samples that shall be combined into a batch.
    :return: The batched data in two tuples: The first tuple contains the filenames for each component.
     The second tuple contains the actual data of each component.
    """
    file_name_list = []
    data_object_list = []
    # iterate over the components of the samples that shall be batched
    for element_idx in range(len(samples[0])):
        # split the tuple into a list of the filenames and a list of the data elements
        file_names = [sample[element_idx][0] for sample in samples]
        data_objects = [sample[element_idx][1] for sample in samples]
        # stack torch tensors
        if isinstance(data_objects[0], torch.Tensor):
            if _check_shapes(data_objects):
                data_objects = torch.stack(data_objects)
        # combine numpy arrays
        elif isinstance(data_objects[0], np.ndarray):
            if _check_shapes(data_objects):
                data_objects = torch.from_numpy(np.array(data_objects))
        file_name_list.append(file_names)
        data_object_list.append(data_objects)
    return tuple(file_name_list), tuple(data_object_list)


def collate_batch_as_dict(samples: list):
    """ Collates individual samples (stored in a list of tuples) into batches.

    Combines corresponding elements of individual samples. Tensors and numpy arrays will be stacked in the batch
    dimension. Other data types are simply placed in lists.  All aggregated components are then placed in a dictionary
    where the keys are the component names and the values are the stacked tensors (or lists in case of other types of
    objects).

    :param samples: A list of the samples that shall be combined into a batch.
    :return: The batched data in a dictionary that maps the component names to the actual data for each component.
    """
    components_data: dict[str, list] = dict()
    # iterate over the components of the samples that shall be batched
    for component_idx in range(len(samples[0])):
        # split the tuples into a list of the filenames and a list of the data elements
        file_names = [sample[component_idx][0] for sample in samples]
        data_objects = [sample[component_idx][1] for sample in samples]
        # stack tensors
        if isinstance(data_objects[0], torch.Tensor):
            if _check_shapes(data_objects):
                data_objects = torch.stack(data_objects)
        # combine numpy arrays
        elif isinstance(data_objects[0], np.ndarray):
            if _check_shapes(data_objects):
                data_objects = np.array(data_objects)
        # aggregate data by component names
        component_name = get_component_name(file_names[0])
        components_data[component_name] = data_objects
    return dict(components_data)


def collate_batch_as_dicts(samples: list):
    """ Collates individual samples (stored in a list of tuples) into batches.

    Combines corresponding elements of individual samples. Tensors and numpy arrays will be stacked in the batch
    dimension. Other data types are simply placed in lists.  All aggregated components are then placed in a dictionary
    where the keys are the component names and the values are the stacked tensors (or lists in case of other types of
    objects).

    :param samples: A list of the samples that shall be combined into a batch.
    :return: The batched data in two dicts: The first dict contains a mapping of component names to filenames.
     The second dict maps the component names to the actual data for each component.
    """
    file_paths: dict[str, list] = dict()
    components_data: dict[str, list] = dict()
    # iterate over the components of the samples that shall be batched
    for component_idx in range(len(samples[0])):
        # split the tuples into a list of the filenames and a list of the data elements
        file_names = [sample[component_idx][0] for sample in samples]
        data_objects = [sample[component_idx][1] for sample in samples]
        # stack tensors
        if isinstance(data_objects[0], torch.Tensor):
            if _check_shapes(data_objects):
                data_objects = torch.stack(data_objects)
        # combine numpy arrays
        elif isinstance(data_objects[0], np.ndarray):
            if _check_shapes(data_objects):
                data_objects = np.array(data_objects)
        # aggregate data by component names
        component_name = get_component_name(file_names[0])
        file_paths[component_name] = file_names
        components_data[component_name] = data_objects
    return dict(file_paths), dict(components_data)


def collate_dict(samples: list) -> dict:
    """ Collates individual samples from a list of dicts into a single dict containing batches.

    Combines corresponding elements of individual samples. Tensors and numpy arrays will be stacked in the batch
    dimension. Other data types are simply placed in lists. All elements are kept with their original keys.

    :param samples: The samples that shall be combined into a batch.
    :return: The dictionary, where all elements with the same key in the input dicts have been stacked/batched.
    """
    batch = dict()
    for key, value in samples[0].items():
        # Gather all values for this key. (One from each sample.)
        values = [x[key] for x in samples]
        # Special handling for specific data types:
        if isinstance(value, dict):
            values = collate_dict(DataChunk(values))
        elif isinstance(value, torch.Tensor):
            values = torch.stack(values, dim=0)
        elif isinstance(value, np.ndarray):
            values = torch.from_numpy(np.array(values))
        # Place collated values in batch
        batch[key] = values
    return batch


def collate_sequences_as_tuples(sequences: list) -> tuple[tuple, tuple]:
    """ Collates individual samples (stored in lists of sequences) into batches.

    Combines corresponding elements of individual sequences. Tensors and numpy arrays will be stacked in the batch and
    sequence dimensions. Other data types are simply placed in nested lists. All aggregated components are then placed
    in a tuple in the same order as they appear in the sequence tuples.

    :param sequences: The sequences that shall be combined into a batch.
    :return: The batched data in two tuples: The first tuple contains the filenames for each component.
     The second tuple contains the actual data of each component.
    """
    component_count = len(sequences[0][0])
    # nested lists: components - batch - samples
    file_paths = [[]] * component_count
    components_data = [[]] * component_count
    # add batch dimension
    for idx in range(component_count):
        file_paths[idx] = [[]] * len(sequences)
        components_data[idx] = [[]] * len(sequences)
    # iterate over the batch (contains sliding windows)
    for seq_idx, window in enumerate(sequences):
        # iterate over the components of the samples that shall be batched
        for component_idx in range(len(window[0])):
            # split the tuple into a list of the filenames and a list of the data elements
            file_names = [sample[component_idx][0] for sample in window]
            data_objects = [sample[component_idx][1] for sample in window]
            # remove state wrappers in case they have not been removed manually
            for idx in range(len(data_objects)):
                if isinstance(data_objects[idx], StateWrapper):
                    data_objects[idx] = data_objects[idx].data
            # stack tensors (unlike numpy arrays, they can not be stacked in multiple dimensions at once)
            if isinstance(data_objects[0], torch.Tensor):
                if _check_shapes(data_objects):
                    data_objects = torch.stack(data_objects)
            # aggregate data by components
            file_paths[component_idx][seq_idx] = file_names
            components_data[component_idx][seq_idx] = data_objects
    # stack data for each component
    for component_idx in range(len(file_paths)):
        current_data = components_data[component_idx]
        if isinstance(current_data[0][0], np.ndarray):
            inner_shapes_match = reduce(lambda x, y: x and y, [_check_shapes(x) for x in current_data])
            outer_shapes_match = _check_shapes([x[0] for x in current_data])
            if inner_shapes_match and outer_shapes_match:
                current_data = torch.from_numpy(np.array(current_data))
            elif inner_shapes_match:
                for seq_idx in range(len(current_data)):
                    current_data[seq_idx] = np.array(current_data[seq_idx])
        elif isinstance(current_data[0], torch.Tensor):
            if _check_shapes(current_data):
                current_data = torch.stack(current_data)
        components_data[component_idx] = current_data
    return tuple(file_paths), tuple(components_data)


def collate_sequences_as_dicts(sequences: list):
    """ Collates individual samples (stored in lists of sequences) into batches.

    Combines corresponding elements of individual sequences. Tensors and numpy arrays will be stacked in the batch and
    sequence dimensions. Other data types are simply placed in nested lists. All aggregated components are then placed
    in a dictionary where the keys are the component names and the values are the stacked tensors (or nested lists in
    case of other types of objects).

    :param sequences: The samples that shall be combined into a batch.
    :return: The batched data in two dicts: The first dict maps from component names to the filenames.
     The second dict maps from the component names to the actual data for each component.
    """
    file_paths: dict[str, list] = dict()
    components_data: dict[str, list] = dict()
    for path, _ in sequences[0][0]:
        component_name = get_component_name(path)
        file_paths[component_name] = [[]] * len(sequences)
        components_data[component_name] = [[]] * len(sequences)
    # iterate over the batch (contains sliding windows)
    for seq_idx, window in enumerate(sequences):
        # iterate over the components of the samples that shall be batched
        for component_idx in range(len(window[0])):
            # split the tuple into a list of the filenames and a list of the data elements
            file_names = [sample[component_idx][0] for sample in window]
            data_objects = [sample[component_idx][1] for sample in window]
            # remove state wrappers in case they have not been removed manually
            for idx in range(len(data_objects)):
                if isinstance(data_objects[idx], StateWrapper):
                    data_objects[idx] = data_objects[idx].data
            # stack tensors (unlike numpy arrays, they can not be stacked in multiple dimensions at once)
            if isinstance(data_objects[0], torch.Tensor):
                if _check_shapes(data_objects):
                    data_objects = torch.stack(data_objects)
            # aggregate data by components
            component_name = get_component_name(file_names[0])
            file_paths[component_name][seq_idx] = file_names
            components_data[component_name][seq_idx] = data_objects
    # stack data for each component
    for component_name in file_paths.keys():
        current_data = components_data[component_name]
        if isinstance(current_data[0][0], np.ndarray):
            inner_shapes_match = reduce(lambda x, y: x and y, [_check_shapes(x) for x in current_data])
            outer_shapes_match = _check_shapes([x[0] for x in current_data])
            if inner_shapes_match and outer_shapes_match:
                # we can combine the arrays across both dimensions at once
                current_data = torch.from_numpy(np.array(current_data))
            elif inner_shapes_match:
                # only the inner dimensions match, so we can only combine the arrays within each sequence
                for seq_idx in range(len(current_data)):
                    current_data[seq_idx] = np.array(current_data[seq_idx])
        elif isinstance(current_data[0], torch.Tensor):
            if _check_shapes(current_data):
                current_data = torch.stack(current_data)
        components_data[component_name] = current_data
    return dict(file_paths), dict(components_data)


def sample_to_dict(sample: tuple):
    """ Collates a single sample into a dictionary.

    :param sample: A tuple containing the sample components.
    :return: A dictionary mapping from component IDs to sample components.
    """
    file_paths: dict[str, list] = dict()
    components_data: dict[str, list] = dict()
    for component_idx in range(len(sample)):
        # split the tuple into filename and data
        filename, data_object = sample[component_idx]
        # extract the name of the component from the file path
        component_name = get_component_name(filename)
        # put data in dict with component name as key
        file_paths[component_name] = filename
        components_data[component_name] = data_object
    return dict(file_paths), dict(components_data)
