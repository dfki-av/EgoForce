# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of the torchdata repository's source tree.

from typing import Iterator, TypeVar

from torch.utils.data import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe

T_co = TypeVar("T_co", covariant=True)


@functional_datapipe("set_length")
class LengthSetterIterDataPipe(IterDataPipe[T_co]):
    r"""
    Set the length attribute of the DataPipe, which is returned by ``__len__`` (functional name: ``set_length``).
    This can be used after DataPipes whose final length cannot be known in advance (e.g. ``filter``). If you
    know the final length with certainty, you can manually set it, which can then be used by
    DataLoader or other DataPipes.

    Note:
        This DataPipe differs from :class:`.Header` in that this doesn't restrict the number of elements that
        can be yielded from the DataPipe; this is strictly used for setting an attribute so that it can be used later.

    Args:
        source_datapipe: a DataPipe
        length: the integer value that will be set as the length
    """

    def __init__(self, source_datapipe: IterDataPipe[T_co], length: int) -> None:
        self.source_datapipe: IterDataPipe[T_co] = source_datapipe
        assert length >= 0
        self.length: int = length

    def __iter__(self) -> Iterator[T_co]:
        yield from self.source_datapipe

    def __len__(self) -> int:
        return self.length
