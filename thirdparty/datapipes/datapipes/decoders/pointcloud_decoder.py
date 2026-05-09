##############################################################################
# Copyright (c) 2022 DFKI GmbH - All Rights Reserved
# Written by David Michael Fürst <david_michael.fuerst@dfki.de>, October 2022
##############################################################################
"""doc
# datapipes.decoders.pointcloud_decoder

> Decode pointcloud files for torch.

"""
from __future__ import annotations
import numpy as np
import torch


class PointcloudDecoder(object):
    def __init__(self, data_shape: tuple[int, int] = (-1, 5), transposed: bool = True, keep_channels: int = 4,
                 output_torch: bool = False) -> None:
        """
        Decode a byte buffer to pointcloud.

        The pointcloud will be loaded from a numpy buffer and reshaped into the correct format.

        :param data_shape: The shape of the original data in the buffer.
        :param transposed: If the buffer is in channel last and needs to be converted to channel first.
            E.g. a buffer of shape (N, 5) would be transposed and must be converted to (5, N).
        :param keep_channels: Selects the number of channels to keep so that the output is [C, N],
            where C is the number of channels and N the number of points.
        :param output_torch: If the output shouldbe a torch tensor or numpy.
        """
        self.data_shape = data_shape
        self.keep_channels = keep_channels
        self.transposed = transposed
        self.output_torch = output_torch

    def __call__(self, file_path: str, data: bytes) -> np.ndarray | torch.Tensor:
        """
        Convert the data based on the specification given to the init function.

        :param file_path: The path to the file in the dataset. Ignored.
        :param data: The byte buffer retrieved from the tar.
        :return: The pointcloud as an array of shape (C, N).
        """
        # Conversion from:
        # https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L256
        scan = np.frombuffer(data, dtype=np.float32)
        points = scan.reshape(self.data_shape)
        if self.transposed:
            points = points.T
        points = points[:self.keep_channels]
        if self.output_torch:
            return torch.tensor(points, dtype=torch.float32)
        else:
            return np.array(points, dtype=np.float32)
