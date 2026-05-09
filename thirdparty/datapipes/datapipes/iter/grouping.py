##############################################################################
# Copyright (c) 2022 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, March 2023
# based on GrouperIterDataPipe in: https://github.com/pytorch/pytorch
##############################################################################
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, TypeVar, Optional

from torch.utils.data import functional_datapipe
from torch.utils.data.datapipes.datapipe import DataChunk, IterDataPipe
from torch.utils.data.datapipes.utils.common import _check_unpickable_fn

T_co = TypeVar('T_co', covariant=True)


@functional_datapipe('sequential_groupby')
class SequentialGrouperIterDataPipe(IterDataPipe[DataChunk]):
    """ Groups data from the input IterDataPipe by keys which are generated from group_key_fn, and yields a DataChunk
    with up to buffer_size elements (functional name: sequential_groupby).

    The samples are read sequentially from the source datapipe and a batch of samples belonging to the same group will
    be yielded once a sample arrives that does not belong to the group or once the buffer size is reached. If the
    guaranteed_group_size is specified, each group is checked whether it has this minimal size. If its size is smaller,
    it will either be dropped if drop_incomplete=True or an error will be raised if drop_incomplete=False.

    :param datapipe: The iterable datapipe that shall be grouped.
    :param group_key_fn: The function that will be used to generate the group key from the data of the source datapipe.
    :param keep_key: Option to yield the matching key along with the items in a tuple, resulting in (key, [items])
           otherwise returning [items].
    :param buffer_size: The size of the buffer for ungrouped data. Also implies the maximum size a group can have.
    :param guaranteed_group_size: The guaranteed minimum group size. Groups smaller than this may be dropped or result
           in an error being raised, depending on the choice of drop_incomplete.
    :param drop_incomplete: Specifies whether groups smaller than guaranteed_group_size will be dropped from the buffer.
    """
    def __init__(self,
                 datapipe: IterDataPipe[T_co],
                 group_key_fn: Callable[[T_co], Any],
                 *,
                 keep_key: bool = False,
                 buffer_size: int = 100,
                 guaranteed_group_size: Optional[int] = None,
                 drop_incomplete: bool = False):
        _check_unpickable_fn(group_key_fn)
        self.datapipe = datapipe
        self.group_key_fn = group_key_fn

        self.keep_key = keep_key
        self.max_buffer_size = buffer_size
        self.last_key = ""
        self.buffer_elements = []
        if guaranteed_group_size is not None:
            assert 0 < guaranteed_group_size <= buffer_size
        self.guaranteed_group_size = guaranteed_group_size
        self.drop_incomplete = drop_incomplete
        self.wrapper_class = DataChunk

    def __iter__(self):
        for x in self.datapipe:
            key = self.group_key_fn(x)

            if len(self.buffer_elements) == 0:
                self.last_key = key

            if self.last_key == key:
                self.buffer_elements.append(x)
                if len(self.buffer_elements) == self.max_buffer_size:
                    result: DataChunk[Any] = self.wrapper_class(self.buffer_elements)
                    yield (self.last_key, result) if self.keep_key else result
                    self.buffer_elements = []
            else:
                self.perform_size_check()
                result: DataChunk[Any] = self.wrapper_class(self.buffer_elements)
                yield (self.last_key, result) if self.keep_key else result
                self.last_key = key
                self.buffer_elements = [x]

        if len(self.buffer_elements) > 0:
            self.perform_size_check()
            result: DataChunk[Any] = self.wrapper_class(self.buffer_elements)
            yield (self.last_key, result) if self.keep_key else result
            self.reset()

    def perform_size_check(self) -> None:
        if self.guaranteed_group_size is not None and len(self.buffer_elements) < self.guaranteed_group_size:
            if self.drop_incomplete:
                self.reset()
            else:
                raise RuntimeError("Number of components below specified minimum:", str(self.buffer_elements))

    def reset(self) -> None:
        self.last_key = ""
        self.buffer_elements = []

    def __getstate__(self):
        state = (
            self.datapipe,
            self.group_key_fn,
            self.keep_key,
            self.max_buffer_size,
            self.guaranteed_group_size,
            self.drop_incomplete,
            self.wrapper_class,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        (
            self.datapipe,
            self.group_key_fn,
            self.keep_key,
            self.max_buffer_size,
            self.guaranteed_group_size,
            self.drop_incomplete,
            self.wrapper_class,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        self.curr_buffer_size = 0
        self.buffer_elements = defaultdict(list)

    def __len__(self):
        return len(self.datapipe)
