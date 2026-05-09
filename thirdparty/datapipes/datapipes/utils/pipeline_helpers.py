from __future__ import annotations

from typing import Callable

from torch.utils.data.datapipes.datapipe import DataChunk

from datapipes.utils.dataset_path_utils import get_ids_from_path
from datapipes.utils.state_wrapper import StateWrapper


class ComponentAdder:
    __slots__ = "get_components"

    def __init__(self, component_add_fn: Callable[[str, str, str], list[tuple]]):
        """ Adds further components to a sample

        :param component_add_fn: A callable that provides additional components for a sample identified by three
         strings: the subset (data split), the sequence id and the sample id. It should return a list of tuples, where
         the first element of each tuple is a string containing component id and file extension ("component_id.ext") and
         the second element contains the data of the component as bytes (or a StreamWrapper object).
        """
        self.get_components = component_add_fn

    def __call__(self, sample: DataChunk):
        """ Adds additional components to a sample.

        :param sample: The sample to which components shall be added.
        :return: The sample with the additional components provided by the get_components function.
        """
        component_path: str = sample[0][0]
        path_components = get_ids_from_path(component_path)
        # get the additional components for the current sample
        new_components = self.get_components(*path_components[0:3])
        # construct a (fake) path for each new component to identify it
        new_dir = component_path[:component_path.rfind("/")]
        start = 2 if path_components[1] == "" else 1
        new_filename = ".".join(path_components[start:-2])
        prefix = f"{new_dir}/{new_filename}."
        new_components = [(prefix + x, y) for x, y in new_components]
        return DataChunk((*sample, *new_components))


class WrapperAdder:
    __slots__ = "state"

    def __init__(self, state: str):
        """ Creates a new callable class the can add a Wrapper to the components of a sample.

        :param state: A string identifying the state the data is in at the time of placing it in the wrapper.
        """
        self.state = state

    def __call__(self, sample: DataChunk):
        """ Wraps the data of each component of a sample with the StateWrapper class."""
        out_sample = []
        for element in sample:
            file_path, data = element
            out_sample.append((file_path, StateWrapper(data, self.state)))
        return DataChunk(tuple(out_sample))
