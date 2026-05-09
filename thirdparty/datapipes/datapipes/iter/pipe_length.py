##############################################################################
# Copyright (c) 2023 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, December 2023
##############################################################################
from __future__ import annotations

import os.path
from typing import Any, Iterator, TypeVar

import torch.distributed as dist
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter.combining import _ChildDataPipe
from torch.utils.data.datapipes.iter.sharding import ShardingFilterIterDataPipe

T_co = TypeVar('T_co', covariant=True)


class TarShardingFilter(ShardingFilterIterDataPipe):
    def __init__(self,
                 source_datapipe: IterDataPipe,
                 metadata: dict,
                 metadata_version: int,
                 temporal_sliding_window_size: int | None,
                 sharding_group_filter=None,):
        super().__init__(source_datapipe, sharding_group_filter)
        self._metadata = metadata
        self._metadata_version = metadata_version
        self._temporal_sliding_window_size = temporal_sliding_window_size
        self.lengths = {}

    @staticmethod
    def extract_ids(path: str | tuple) -> tuple[str, str]:
        if isinstance(path, tuple):
            path = path[0]
        subset_id = os.path.basename(os.path.dirname(path))
        shard_id = os.path.basename(path).split(".")[0]
        return subset_id, shard_id

    def __iter__(self) -> Iterator[T_co]:
        tar_list = list(self.source_datapipe)

        for idx in range(self.num_of_instances):
            self.lengths[idx] = 0

        # determine length of each pipeline instance
        for i, item in enumerate(tar_list):
            instance_id = i % self.num_of_instances

            if self._temporal_sliding_window_size is None or self._temporal_sliding_window_size == 1:
                subset_id, shard_id = TarShardingFilter.extract_ids(tar_list[i])
                self.lengths[instance_id] += self._metadata[subset_id]["samples_per_shard"][shard_id]
            else:
                if self._metadata_version == 1:
                    break
                # compute number of samples that can be generated with the giving window size
                subset_id, shard_id = TarShardingFilter.extract_ids(tar_list[i])
                for sequence_len in self._metadata[subset_id]["sequence_lengths_per_shard"][shard_id]:
                    old_len = self.lengths.get(instance_id, 0)
                    self.lengths[instance_id] = old_len + sequence_len - self._temporal_sliding_window_size + 1

        self.lengths["worker_id"] = self.instance_id
        self.lengths["total_num_workers"] = self.num_of_instances

        # filter out tars that belong to other pipeline instances
        tar_list = [x for i, x in enumerate(tar_list) if i % self.num_of_instances == self.instance_id]

        # yield tars for next step in pipeline
        for item in tar_list:
            yield item

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state: dict[str, Any]):
        self.__dict__ = state


class SequenceShardingFilter(ShardingFilterIterDataPipe):
    def __init__(self,
                 source_datapipe: IterDataPipe,
                 metadata: dict,
                 metadata_version: int,
                 temporal_sliding_window_size: int | None,
                 sharding_group_filter=None):
        super().__init__(source_datapipe, sharding_group_filter)
        self._metadata = metadata
        self._metadata_version = metadata_version
        self._temporal_sliding_window_size = temporal_sliding_window_size
        self.lengths = {}

    @staticmethod
    def extract_ids(path: str | tuple) -> tuple[str, str, str]:
        if isinstance(path, tuple):
            path = path[0]
        subset_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
        shard_id = os.path.basename(os.path.dirname(path)).split(".")[0]
        sequence_id = os.path.basename(path).split(".")[0]
        return subset_id, shard_id, sequence_id

    def __iter__(self) -> Iterator[T_co]:
        # yield tars for next step in pipeline
        self.lengths["worker_id"] = self.instance_id
        self.lengths["total_num_workers"] = self.num_of_instances

        last_seq_id = None
        i = -1
        for item in self.source_datapipe:
            path, data = item[0]
            subset_id, shard_id, sequence_id = SequenceShardingFilter.extract_ids(path)

            if last_seq_id is None:
                # determine length of each pipeline instance
                for j, sequence_len in enumerate(self._metadata[subset_id]["sequence_lengths_per_shard"][shard_id]):
                    instance_id = j % self.num_of_instances
                    old_len = self.lengths.get(instance_id, 0)
                    # compute number of samples that can be generated with the giving window size
                    self.lengths[instance_id] = old_len + sequence_len - self._temporal_sliding_window_size + 1

            if sequence_id != last_seq_id:
                last_seq_id = sequence_id
                i += 1
            if i % self.num_of_instances == self.instance_id:
                yield item

    def __getstate__(self) -> tuple[tuple, dict, int, int, dict[int | str, int]]:
        base_state = super().__getstate__()
        state = (base_state,
                 self._metadata,
                 self._metadata_version,
                 self._temporal_sliding_window_size,
                 self.lengths)
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state: tuple[tuple, dict, int, int, dict[int | str, int]]) -> None:
        (
            base_state,
            self._metadata,
            self._metadata_version,
            self._temporal_sliding_window_size,
            self.lengths
        ) = state
        super().__setstate__(base_state)


class LengthLimiter(IterDataPipe[T_co]):
    def __init__(self,
                 source_datapipe: IterDataPipe[T_co],
                 batch_size: int | None = None,
                 max_samples: int | None = None) -> None:
        self.datapipe: IterDataPipe[T_co] = source_datapipe
        self.multi_gpu = dist.is_initialized()
        self.num_gpus = int(os.environ["WORLD_SIZE"]) if self.multi_gpu else 1
        self.limit = None
        self.batch_size = batch_size if batch_size is not None else 1
        self.max_samples = max_samples

    @staticmethod
    def _get_lengths(datapipe) -> dict[int | str, int] | None:
        def _helper(members: dict) -> dict[int | str, int] | None:
            for member in members.values():
                if isinstance(member, (TarShardingFilter, SequenceShardingFilter)):
                    return member.lengths
                if isinstance(member, ShardingFilterIterDataPipe):
                    return {"worker_id": member.instance_id,
                            "total_num_workers": member.num_of_instances}
                if isinstance(member, (IterDataPipe, _ChildDataPipe)):
                    return _helper(member.__dict__)
                if isinstance(member, (tuple, list)) and isinstance(member[0], IterDataPipe):
                    return _helper(member[0].__dict__)
            return None

        if isinstance(datapipe, TarShardingFilter):
            return datapipe.lengths
        if isinstance(datapipe, IterDataPipe):
            lengths = _helper(datapipe.__dict__)
            if len(lengths) == 2:
                num_workers = lengths["total_num_workers"]
                for i in range(num_workers):
                    lengths[i] = datapipe.length // num_workers + int(i < datapipe.length % num_workers)
            return lengths
        return None

    def _compute_limit(self) -> int | None:
        if (self.multi_gpu and self.num_gpus > 1) or self.max_samples is not None:
            lengths = LengthLimiter._get_lengths(self.datapipe)
            # silently fail if length data is not available
            if lengths is None:
                return None

            gpu_id = int(os.environ["RANK"]) if self.multi_gpu else 0
            worker_id = lengths["worker_id"]
            num_workers = lengths["total_num_workers"] // self.num_gpus

            # determine number of samples available to each GPU
            worker_group_lens = []
            for gpu_idx in range(self.num_gpus):
                group_limit = 0
                for worker_idx in range(num_workers):
                    group_limit += lengths[worker_idx * self.num_gpus + gpu_idx] // self.batch_size
                worker_group_lens.append(group_limit)
            # determine the minimum number of available samples across GPUs
            group_limit = min(worker_group_lens)
            # incorporate user-defined limit
            if self.max_samples is not None:
                group_limit = min(group_limit, self.max_samples // self.batch_size)

            # derive limits for each individual worker (pipeline) of the current GPU
            pipe_limits = [(i, x // self.batch_size) for i, x in lengths.items()
                           if isinstance(i, int) and (i % self.num_gpus == gpu_id)]
            # sort by pipeline lengths in descending order
            pipe_limits = list(sorted(pipe_limits, key=lambda x: x[1], reverse=True))
            worker_ids = [x[0] for x in pipe_limits]
            pipe_limits = [x[1] for x in pipe_limits]

            # determine how many batches need to be dropped to conform to group limit
            excess_batches = worker_group_lens[gpu_id] - group_limit
            # pad a 0 value in case we need to remove batches from all pipes / workers
            pipe_limits.append(0)
            for step in range(1, num_workers + 1):
                if excess_batches < step:
                    break
                max_reduction = pipe_limits[step - 1] - pipe_limits[step]
                cur_reduction = min(excess_batches // step, max_reduction)
                for idx in range(step):
                    pipe_limits[idx] -= cur_reduction
                excess_batches -= cur_reduction * step
            # remove remaining excess batches where we could not remove batches from all pipelines equally
            for idx in range(excess_batches):
                pipe_limits[idx] -= 1
            # determine which limit is the one for the current pipeline
            own_pipe_idx = worker_ids.index(worker_id)
            return pipe_limits[own_pipe_idx] * self.batch_size
        return None

    def __iter__(self) -> Iterator[T_co]:
        for i, value in enumerate(self.datapipe):
            if i == 0:
                # computing the pipeline length limit needs to be done after iteration has started
                self.limit = self._compute_limit()
            if self.limit is None or i < self.limit:
                yield value
            else:
                break

    def __len__(self) -> int:
        if self.limit is not None:
            return self.limit
        return len(self.datapipe)

    def __getstate__(self) -> tuple[IterDataPipe[T_co], int, int, int | None, int | None, int | None]:
        state = (self.datapipe, self.multi_gpu, self.num_gpus, self.limit, self.batch_size, self.max_samples)
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state: tuple[IterDataPipe[T_co], int, int, int | None, int | None, int | None]) -> None:
        self.datapipe, self.multi_gpu, self.num_gpus, self.limit, self.batch_size, self.max_samples = state

    def reset(self) -> None:
        self.limit = None
