from __future__ import annotations

from typing import Any

from torch.utils.data.datapipes.datapipe import DataChunk

from datapipes.utils.state_wrapper import StateWrapper


def remove_wrapper(sample: dict | tuple | DataChunk):
    """ This function removes the StateWrapper from the components of samples.

    It should be added as the last step to a pipeline for sequence data.

    :param sample: The sample whose wrappers shall be removed.
    :return: The sample without StreamWrappers on the sample components.
    """
    def _remove_from_data_chunk(element: DataChunk):
        new_element = []
        for path, data in element:
            if isinstance(data, StateWrapper):
                new_element.append((path, data.data))
            else:
                new_element.append((path, data))
        return tuple(new_element)

    def _remove_from_dict(element: dict[str, Any]):
        out_dict = {}
        for path, data in element.items():
            if isinstance(data, StateWrapper):
                out_dict[path] = data.data
            else:
                out_dict[path] = data
        return out_dict

    if isinstance(sample, tuple):
        # handle nested structure when training with sequence data (multiple samples)
        parts = [_remove_from_data_chunk(p) for p in sample]
        return DataChunk(tuple(parts))
    else:
        # remove wrapper from a single sample
        if isinstance(sample, dict):
            return _remove_from_dict(sample)
        else:
            return _remove_from_data_chunk(sample)
