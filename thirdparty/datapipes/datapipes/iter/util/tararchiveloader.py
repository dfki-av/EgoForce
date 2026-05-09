# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of the torchdata repository's source tree.

import os
import tarfile
import warnings
from io import BufferedIOBase, IOBase
from typing import cast, IO, Iterable, Iterator, Optional, Tuple

from torch.utils.data import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import StreamWrapper


def validate_pathname_binary_tuple(data: Tuple[str, IOBase]):
    if not isinstance(data, tuple):
        raise TypeError(f"pathname binary data should be tuple type, but it is type {type(data)}")

    if len(data) != 2:
        raise TypeError(f"pathname binary stream tuple length should be 2, but got {len(data)}")

    if not isinstance(data[0], str):
        raise TypeError(f"pathname within the tuple should have string type pathname, but it is type {type(data[0])}")

    if not isinstance(data[1], IOBase) and not isinstance(data[1], StreamWrapper):
        raise TypeError(f"binary stream within the tuple should have IOBase or"
                        f"its subclasses as type, but it is type {type(data[1])}")


@functional_datapipe("load_from_tar")
class TarArchiveLoaderIterDataPipe(IterDataPipe[Tuple[str, BufferedIOBase]]):
    r"""
    Opens/decompresses tar binary streams from an Iterable DataPipe which contains tuples of path name and
    tar binary stream, and yields a tuple of path name and extracted binary stream (functional name: ``load_from_tar``).

    Args:
        datapipe: Iterable DataPipe that provides tuples of path name and tar binary stream
        mode: File mode used by `tarfile.open` to read file object.
            Mode has to be a string of the form `'filemode[:compression]'`
        length: a nominal length of the DataPipe

    Note:
        The opened file handles will be closed automatically if the default ``DecoderDataPipe``
        is attached. Otherwise, user should be responsible to close file handles explicitly
        or let Python's GC close them periodically.
    """

    def __init__(self, datapipe: Iterable[Tuple[str, BufferedIOBase]], mode: str = "r:*", length: int = -1) -> None:
        super().__init__()
        self.datapipe: Iterable[Tuple[str, BufferedIOBase]] = datapipe
        self.mode: str = mode
        self.length: int = length

    def __iter__(self) -> Iterator[Tuple[str, BufferedIOBase]]:
        for data in self.datapipe:
            validate_pathname_binary_tuple(data)
            pathname, data_stream = data
            try:
                if isinstance(data_stream, StreamWrapper) and isinstance(data_stream.file_obj, tarfile.TarFile):
                    tar = data_stream.file_obj
                else:
                    reading_mode = (
                        self.mode
                        if hasattr(data_stream, "seekable") and data_stream.seekable()
                        else self.mode.replace(":", "|")
                    )
                    # typing.cast is used here to silence mypy's type checker
                    tar = tarfile.open(fileobj=cast(Optional[IO[bytes]], data_stream), mode=reading_mode)
                for tarinfo in tar:
                    if not tarinfo.isfile():
                        continue
                    extracted_fobj = tar.extractfile(tarinfo)
                    if extracted_fobj is None:
                        warnings.warn(f"failed to extract file {tarinfo.name} from source tarfile {pathname}")
                        raise tarfile.ExtractError
                    inner_pathname = os.path.normpath(os.path.join(pathname, tarinfo.name))
                    yield inner_pathname, StreamWrapper(extracted_fobj, data_stream, name=inner_pathname)
            except Exception as e:
                warnings.warn(f"Unable to extract files from corrupted tarfile stream {pathname} due to: {e}, abort!")
                raise e
            finally:
                if isinstance(data_stream, StreamWrapper):
                    data_stream.autoclose()

    def __len__(self) -> int:
        if self.length == -1:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
        return self.length
