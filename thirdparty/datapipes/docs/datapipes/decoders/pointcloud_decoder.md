[Back to Overview](../../README.md)

# datapipes.decoders.pointcloud_decoder

> Decode pointcloud files for torch.

## *class* **PointcloudDecoder**(object) [[src]](../../../datapipes/decoders/pointcloud_decoder.py#L16)
Decode a byte buffer to pointcloud.

The pointcloud will be loaded from a numpy buffer and reshaped into the correct format.

* **data_shape**: The shape of the original data in the buffer.
* **transposed**: If the buffer is in channel last and needs to be converted to channel first.
    E.g. a buffer of shape (N, 5) would be transposed and must be converted to (5, N).
* **keep_channels**: Selects the number of channels to keep so that the output is [C, N],
    where C is the number of channels and N the number of points.
* **output_torch**: If the output shouldbe a torch tensor or numpy.
