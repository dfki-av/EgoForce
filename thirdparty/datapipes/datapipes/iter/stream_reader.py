##############################################################################
# Copyright (c) 2022 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, December 2022
##############################################################################
import io
from typing import Sized, TypeVar

from torch.utils.data import functional_datapipe
from torch.utils.data.datapipes.datapipe import DataChunk, IterDataPipe
from torch.utils.data.datapipes.utils.common import StreamWrapper


T_co = TypeVar('T_co', covariant=True)


@functional_datapipe("read_streams")
class StreamReaderIterDataPipe(IterDataPipe[T_co]):
    def __init__(self, source_dp: IterDataPipe) -> None:
        """ Reads an entire (file) stream into a byte array and closes the stream.

        This is usually done automatically as part of the file decoding process. In certain cases it can be necessary
        to do this explicitly as a separate step. One example is the application of a temporary sliding window filter
        to the stream of samples. In this case the stream needs to be read explicitly and passed on, because it is
        needed multiple times.

        :param source_dp: The source pipeline from which to read elements.
        """
        super().__init__()
        self.datapipe = source_dp

    @staticmethod
    def _is_stream_handle(data):
        obj_to_check = data.file_obj if isinstance(data, StreamWrapper) else data
        return isinstance(obj_to_check, io.BufferedIOBase) or isinstance(obj_to_check, io.RawIOBase)

    def __iter__(self):
        for elements in self.datapipe:
            sample = []
            for element in elements:
                file_path, data = element
                if StreamReaderIterDataPipe._is_stream_handle(data):
                    ds = data
                    # The behavior of .read can differ between streams (e.g. HTTPResponse), hence this is used instead
                    data = b"".join(data)
                    # close (file) stream
                    ds.close()
                sample.append((file_path, data))
            yield DataChunk(tuple(sample))

    def __len__(self):
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)
        raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))

    def reset(self) -> None:
        pass

    def __getstate__(self):
        state = self.datapipe
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state: IterDataPipe):
        self.datapipe = state
