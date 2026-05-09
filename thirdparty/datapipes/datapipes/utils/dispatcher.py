##############################################################################
# Copyright (c) 2022 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, October 2022
##############################################################################
"""doc
# datapipes.utils.dispatcher

> The dispatcher can be used to map transformation functions to individual elements of a sample, e.g. sample components.
"""
from __future__ import annotations

import io
from typing import Any, Callable

from torch.utils.data.datapipes.datapipe import DataChunk
from torch.utils.data.datapipes.utils.common import StreamWrapper

from datapipes.utils.state_wrapper import StateWrapper
from datapipes.utils.dataset_path_utils import get_component_name


class Dispatcher:
    def __init__(self,
                 transformations: dict[str, Callable[[str, Any], Any]],
                 in_state: str = "encoded",
                 out_state: str = "decoded",
                 key_fn: Callable[[str], str] = get_component_name):
        """ Creates an object that applies the specified transformations to the respective components of samples.

        :param transformations: A dictionary mapping a key to a transformation function. By default, the key is the
         component identifier.
        :param in_state: The expected instate of the data if it is encapsulated in a StateWrapper.
        :param out_state: The state of the data in the StateWrapper after it has been processed by the specified
         transformation function.
        :param key_fn: The function that shall be used to extract the key for indexing into the transformations
         dictionary from the file_path of each component. Defaults to a function that extracts the component identifier.
        """
        self.transformations = transformations
        self.in_state = in_state
        self.out_state = out_state
        self.key_fn = key_fn

    @staticmethod
    def _is_stream_handle(data):
        obj_to_check = data.file_obj if isinstance(data, StreamWrapper) else data
        return isinstance(obj_to_check, io.BufferedIOBase) or isinstance(obj_to_check, io.RawIOBase)

    def _transform1(self, file_path: str, data: Any):
        key = self.key_fn(file_path)

        if Dispatcher._is_stream_handle(data):
            ds = data
            # The behavior of .read can differ between streams (e.g. HTTPResponse), hence this is used instead
            data = b"".join(data)
            ds.close()

        if isinstance(data, StateWrapper):
            if data.state == self.in_state and key in self.transformations.keys():
                # apply appropriate transformation
                data.data = self.transformations[key](file_path, data.data)
                # update state of wrapped data
                data.state = self.out_state
        else:
            if key in self.transformations.keys():
                data = self.transformations[key](file_path, data)

        return data

    def _transform_data_chunk(self, sample: DataChunk):
        transformed_sample = []
        for file_path, data in sample:
            transformed_data = self._transform1(file_path, data)
            transformed_sample.append((file_path, transformed_data))
        return DataChunk(tuple(transformed_sample))

    def _transform_dict(self, sample: dict):
        transformed_sample = {}
        for key, val in sample.items():
            transformed_sample[key] = self._transform1(key, val)
        return transformed_sample

    def __call__(self, sample: dict | tuple | DataChunk):
        if isinstance(sample, tuple):
            # handle nested structure when training with sequence data (multiple samples)
            transformed_elements = []
            for element in sample:
                transformed_elements.append(self._transform_data_chunk(element))
            return tuple(transformed_elements)
        elif isinstance(sample, DataChunk):
            return self._transform_data_chunk(sample)
        elif isinstance(sample, dict):
            return self._transform_dict(sample)
        else:
            raise RuntimeError(f"Unsupported sample structure: '{type(sample)}' (must be tuple, DataChunk or dict)")
