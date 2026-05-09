##############################################################################
# Copyright (c) 2023 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, July 2023
##############################################################################
from __future__ import annotations

import os.path
import random
from collections import deque
from typing import Sized, TypeVar

from torch.utils.data import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe

from datapipes.iter.sliding_window import SlidingWindowIterDataPipe


T_co = TypeVar('T_co', covariant=True)


@functional_datapipe("seq_aware_shuffle")
class SequenceAwareShuffler(IterDataPipe[T_co]):

    def __init__(self, source_dp: IterDataPipe, max_concurrent_sequences: int) -> None:
        """ Randomly picks samples from up to max_concurrent_sequences. The order of the sequences is not changed.

        Due to pipelining the shuffler will require memory for approximately (max_concurrent_sequences + 1) / 2
        sequences to draw samples from. Once the pipelines is filled, samples randomly selected from
        max_concurrent_sequences will be emitted.

        :param source_dp: The source data pipe from which to read.
        :param max_concurrent_sequences: The maximum number of concurrent sequences from which to draw samples.
        """
        super().__init__()
        assert max_concurrent_sequences > 0, "Window size needs to be larger than 0!"
        self.datapipe = source_dp
        self.max_concurrent_sequences = max_concurrent_sequences
        self._sequence = []
        self._generations = deque(maxlen=self.max_concurrent_sequences)
        for _ in range(max_concurrent_sequences):
            self._generations.append([])
        self._current_sequence_name = ""

    @staticmethod
    def extract_sequence_name(sample) -> str:
        # get first frame of sequence fragment
        x = sample[0]
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
        return sequence_name

    def distribute_samples(self) -> None:
        sample_count = len(self._sequence)
        chunk_size = (sample_count + self.max_concurrent_sequences - 1) // self.max_concurrent_sequences
        for i in range(self.max_concurrent_sequences):
            self._generations[i].extend(self._sequence[i * chunk_size:(i + 1) * chunk_size])
        self._sequence.clear()

    def __iter__(self):
        def yield_samples():
            # shuffle the samples before yielding them
            random.shuffle(self._generations[0])
            # yield all samples from oldest generation
            for sample in self._generations[0]:
                yield sample
            # drop the oldest generation and add an empty new generation
            self._generations.append([])

        for x in self.datapipe:
            sequence_name = self.extract_sequence_name(x)
            if sequence_name != self._current_sequence_name:
                self._current_sequence_name = sequence_name
                # split up the sequence and distribute one fragment to each generation
                self.distribute_samples()
                if len(self._generations[0]) > 0:
                    yield from yield_samples()
            self._sequence.append(x)

        if len(self._sequence) > 0:
            self.distribute_samples()

        # empty pipeline
        while len(self._generations[0]) > 0:
            yield from yield_samples()

    def __len__(self):
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)
        raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))

    def reset(self) -> None:
        self._sequence.clear()
        self._generations = deque(maxlen=self.max_concurrent_sequences)
        for _ in range(self.max_concurrent_sequences):
            self._generations.append([])
        self._current_sequence_name = ""

    def __getstate__(self) -> tuple[IterDataPipe[T_co], int, list, deque[list], str]:
        state = (self.datapipe,
                 self.max_concurrent_sequences,
                 self._sequence,
                 self._generations,
                 self._current_sequence_name)
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state: tuple[IterDataPipe[T_co], int, list, deque[list], str]):
        (
            self.datapipe,
            self.max_concurrent_sequences,
            self._sequence,
            self._generations,
            self._current_sequence_name
        ) = state


def _test():
    from torch.utils.data.datapipes.iter import IterableWrapper
    print("Example use of the temporal sliding window pipeline operator.\n"
          "Samples are represented by integer values.")
    sequence_len = 20
    dp = IterableWrapper(zip(["seq_a.smpl.comp.ext"] * 10 + ["seq_b.smpl.comp.ext"] * 10 + ["seq_c.smpl.comp.ext"] * 10,
                             range(sequence_len)))
    dp = SlidingWindowIterDataPipe(dp, window_size=3)
    dp = dp.seq_aware_shuffle(2)
    for i, sample in enumerate(dp):
        print(i, ":", sample)


if __name__ == "__main__":
    _test()
