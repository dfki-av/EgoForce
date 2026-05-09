##############################################################################
# Copyright (c) 2022 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, December 2022
##############################################################################
from __future__ import annotations
import os.path
from collections import deque
from typing import Sized, TypeVar

from torch.utils.data import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe


T_co = TypeVar('T_co', covariant=True)


@functional_datapipe("sliding_window")
class SlidingWindowIterDataPipe(IterDataPipe[T_co]):
    def __init__(self, source_dp: IterDataPipe, window_size: int) -> None:
        """ Groups samples within a temporal sliding window.

        The step size it currently fixed to 1.

        :param source_dp: The source data pipe from which to read.
        :param window_size: The size of the temporal window. Must be larger than zero. Size 1 results in an identity
         operator.
        """
        super().__init__()
        assert window_size > 0, "Window size needs to be larger than 0!"
        self.datapipe = source_dp
        self.window_size = window_size
        self._ring_buffer = deque(maxlen=self.window_size)
        self._current_sequence: str = ""

    def __iter__(self):
        for x in self.datapipe:
            if isinstance(x[0], tuple):
                # We have multiple components. Check the path of the first one.
                file_path, _ = x[0]
            else:
                # We have only one component.
                file_path, _ = x
            file_name_elements = os.path.basename(file_path).split('.')
            # The filename is structured like this: {sequence-id.}sample-name.component-id.extension
            # If the sequence-id is not present, assume that all samples belong to the same sequence.
            sequence_name = file_name_elements[0] if len(file_name_elements) == 4 else ""
            if sequence_name != self._current_sequence:
                # reset buffer when moving to next sequence
                self.reset()
                self._current_sequence = sequence_name
            self._ring_buffer.append(x)
            if len(self._ring_buffer) == self.window_size:
                sample = []
                for el in self._ring_buffer:
                    sample.append(el)
                yield tuple(sample)

    def __len__(self):
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)
        raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))

    def reset(self) -> None:
        self._ring_buffer = deque(maxlen=self.window_size)
        self._current_sequence = ""

    def __getstate__(self):
        state = (self.datapipe, self.window_size, self._ring_buffer, self._current_sequence)
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state: tuple[IterDataPipe, int, deque, str]):
        self.datapipe, self.window_size, self._ring_buffer, self._current_sequence = state


def _test():
    from torch.utils.data.datapipes.iter import IterableWrapper
    print("Example use of the temporal sliding window pipeline operator.\n"
          "Samples are represented by simple integer values.")
    sequence_len = 20
    dp = IterableWrapper(zip(["seq_a.s.c.e"] * 10 + ["seq_b.s.c.e"] * 10, range(sequence_len)))
    dp = dp.sliding_window(window_size=3)
    for i, sample in enumerate(dp):
        print(i, ":", sample)


if __name__ == "__main__":
    _test()
