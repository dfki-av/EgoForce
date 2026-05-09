##############################################################################
# Copyright (c) 2023 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, October 2023
##############################################################################
"""doc
# datapipes.in_memory_dataset: random access to datasets that fit into memory

> Provides index-based access to the samples of datasets stored in main memory.
> The dataset is read only once and kept in a separate process which serves the sample access requests.
> This means that the dataset must fit into main memory.

"""
from __future__ import annotations

import hashlib
import os
import multiprocessing as mp
import socket
import sys
import time

from threading import Thread
from typing import Any, Callable, NoReturn, Sequence

import msgpack

from datapipes.base_pipeline import BasePipelineCreator
from datapipes.iter.stream_reader import StreamReaderIterDataPipe
from datapipes.utils.collation_functions import sample_to_dict
from datapipes.utils.dispatcher import Dispatcher


class InMemoryDataset(Sequence):
    """
    The InMemoryDataset class handles the logic needed to orchestrate the interactions between the workers of a
    dataloader and the in-memory database holding a specific subset of a dataset in memory.
    """
    def __init__(self,
                 dataset_path: str,
                 subset: str | list[str],
                 decoder_map: dict[str, Callable] | None = None,
                 components: list[str] | None = None,
                 max_samples: int | None = None,
                 add_component_fn: Callable[[str, str, str], list[tuple]] | None = None):
        """ Creates a new InMemoryDataset which provides index based access to a dataset.

        The dataset is loaded into memory only once in a separate process. Each sample is accessed via socket-based IPC.

        :param dataset_path: The path to the sharded dataset that shall be loaded.
        :param subset: The subset(s)/split(s) of the dataset that shall be loaded. If multiple subsets are specified,
        they will be merged into a single set of samples.
        :param decoder_map: A dictionary that maps component names to functions that decode the content of the file
        containing the data of the respective component.
        :param components: The components that shall be loaded. The default None results in all components being loaded.
        :param max_samples: The maximum number of samples that shall be loaded. The default is to load all samples.
        :param add_component_fn: An optional function or callable class that adds additional components to a sample.
         The function will be called with three parameters: subset (data split), sequence id and sample id.
         It must return a list of tuples with the component ID and file extension as the first element and the
         component data as the second, e.g. ("depth.png", <image data>).
        """
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        subset_path = os.path.join(dataset_path, subset)
        if not os.path.exists(subset_path):
            raise ValueError(f"The subset {subset} does not exist in {dataset_path}.")
        hash_fn = hashlib.blake2b(subset_path.encode('utf-8'), digest_size=4)
        self.socket_path = f"/tmp/imds_{hash_fn.hexdigest()}"

        local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
        if local_rank == 0:
            if os.path.exists(self.socket_path):
                os.remove(self.socket_path)
            command_interface = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            command_interface.bind(self.socket_path)
            command_interface.listen(256)
            dataset_server_process = mp.Process(target=_dataset_server,
                                                args=(command_interface, dataset_path, subset, components,
                                                      max_samples, add_component_fn),
                                                daemon=True)
            dataset_server_process.start()
        self.decoder_map = decoder_map
        self.last_pid = None

    def __getitem__(self, item: int) -> dict[str, Any]:
        """ Returns the sample with the specified index.

        Each sample is a dictionary mapping from sample component names to the sample components. If a decoder map was
        provided, then each sample component will be decoded on access in the process that has requested this item.
        Otherwise, the file contents for each sample component are provided as bytes.

        :param item: The index of the sample that shall be retrieved.
        """
        # ensure that each (worker) process gets its own client
        if self.last_pid is None or self.last_pid != os.getpid():
            self.last_pid = os.getpid()
            self.dataset_accessor = _DatasetAccessClient(self.socket_path, self.decoder_map)
        return self.dataset_accessor[item]

    def __len__(self) -> int:
        """ Returns the number of samples in the dataset split. """
        # ensure that each (worker) process gets its own client
        if self.last_pid is None or self.last_pid != os.getpid():
            self.last_pid = os.getpid()
            self.dataset_accessor = _DatasetAccessClient(self.socket_path, self.decoder_map)
        return len(self.dataset_accessor)


def _identity(x):
    """ The identity function. """
    return x


class _DatasetAccessClient(Sequence):
    """
    An internal helper class for accessing samples stored in another process. This class should not be instantiated
    directly.
    """
    __slots__ = ["socket", "decoding_dispatcher", "ds_length"]

    def __init__(self, socket_path: str, decoder_map: dict[str, Callable] | None = None):
        """ Creates a new DatasetAccessClient.

        DatasetAccessClient are created automatically by the InMemoryDataset. There is no need to manually create any.
        A DatasetAccessClient provides access to the process that keeps the dataset in memory. Each DatasetAccessClient
        instance may not be used by more than one worker process.
        :param socket_path: The path to the UNIX domain socket that shall be used for communication with the dataset
        server process.
        :param decoder_map: An optional decoder map holding a decoding function for each component.
        """
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        count = 0
        while not os.path.exists(socket_path) and count < 300:
            time.sleep(0.1)
            count += 1
        self.socket.connect(socket_path)
        self.decoding_dispatcher = Dispatcher(decoder_map, key_fn=_identity) if decoder_map is not None else None
        self.ds_length = int.from_bytes(self.socket.recv(4), byteorder=sys.byteorder, signed=False)

    def __getitem__(self, item: int) -> dict[str, Any]:
        """ Returns the sample with the specified index.

        Each sample is a dictionary mapping from sample component names to sample components. If a decoder map was
        provided, then each sample component will be decoded on access in the process that has requested this item.
        Otherwise, the file contents for each sample component are provided as bytes.

        :param item: The index of the sample that shall be retrieved.
        :return: A dictionary containing the components of the requested sample. For components for which no decoder was
        specified the file contents will be returned as bytes.
        """
        if item < self.ds_length:
            # request sample with index "item"
            sample_idx = item.to_bytes(4, byteorder=sys.byteorder, signed=False)
            self.socket.sendall(sample_idx)
            # receive sample
            encoded_length = int.from_bytes(self.socket.recv(4), byteorder=sys.byteorder, signed=False)
            received_length = 0
            sample = b""
            # the loop is required because we use a streaming socket and the data may not have been sent completely, yet
            while received_length < encoded_length:
                chunk = self.socket.recv(encoded_length - received_length)
                sample += chunk
                chunk_len = len(chunk)
                received_length += chunk_len
            sample = msgpack.loads(sample)
            if self.decoding_dispatcher is not None:
                sample = self.decoding_dispatcher(sample)
            return sample
        else:
            raise IndexError

    def __len__(self) -> int:
        """ Returns the number of samples in the dataset. """
        return self.ds_length


def _dataset_server(command_socket: socket.socket,
                    path: str,
                    subset: str | list[str],
                    components: list[str] | None = None,
                    max_samples: int | None = None,
                    add_component_fn: Callable[[str, str, str], list[tuple]] | None = None) -> NoReturn:
    """ The logic of the in-memory dataset storage process. Do not call this directly!"""
    dataset = _load_dataset(path, subset, components, max_samples, add_component_fn)

    def __thread_worker(connection: socket.SocketType):
        try:
            # let the worker know about the size of the dataset
            enc_db_length = len(dataset).to_bytes(4, byteorder=sys.byteorder, signed=False)
            connection.sendall(enc_db_length)
            # process requests for individual samples
            while True:
                received_bytes = connection.recv(4)
                # terminate if connection was closed
                if not received_bytes:
                    break
                sample_idx = int.from_bytes(received_bytes, byteorder=sys.byteorder, signed=False)
                # send response back to client
                connection.sendall(dataset[sample_idx])
        finally:
            connection.close()

    # Event loop
    while True:
        connection, client_address = command_socket.accept()
        t = Thread(target=__thread_worker, args=(connection,))
        t.start()


def _load_dataset(dataset_dir: str,
                  subset: str | list[str],
                  components: list[str] | None = None,
                  max_samples: int | None = None,
                  add_component_fn: Callable[[str, str, str], list[tuple]] | None = None) -> list[bytes]:
    """ Reads one or more subsets of a dataset and stores them in a list.

    If multiple subsets are specified, then the samples of these subsets are merged into a single set. To serve the
    subsets separately, separate databases must be created.

    :param dataset_dir: The path to the directory that contains the dataset.
    :param subset: The subset(s) of the dataset that shall be loaded (typically train, val or test).
    :param components: The components that shall be loaded. The default is to load all components.
    :param max_samples: Maximum number of samples that shall be loaded. The default is to load all samples.
    :param add_component_fn: An optional function or callable class that adds additional components to a sample.
    The function will be called with three parameters: subset (data split), sequence id and sample id.
    :return: A list containing the samples of the subset(s) of the dataset.
    """
    factory = BasePipelineCreator(dataset_dir)
    pipe = factory.create_datapipe(subset, 0, components, False,
                                   max_samples=max_samples, add_component_fn=add_component_fn)
    pipe = StreamReaderIterDataPipe(pipe)
    pipe = pipe.map(fn=sample_to_dict)
    dataset = []
    for sample in pipe:
        sample_bytes = msgpack.dumps(sample[1])
        len_bytes = len(sample_bytes).to_bytes(4, byteorder=sys.byteorder, signed=False)
        dataset.append(len_bytes + sample_bytes)
    return dataset
