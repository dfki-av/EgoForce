##############################################################################
# Copyright (c) 2022 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, June 2022
##############################################################################
from __future__ import annotations
from argparse import ArgumentParser

import torch
from torch.utils.data.datapipes.datapipe import IterDataPipe

from datapipes.base_pipeline import BasePipelineCreator
from datapipes.decoders.image_decoder import ImageDecoder
from datapipes.utils.dispatcher import Dispatcher


def landmark_decoder(file_path: str, data: bytes):
    """ Decodes text files containing landmark point files.

    :param file_path: The path to the file that shall be decoded.
    :param data: A stream of bytes representing the content of the file.
    :return: The decoded landmarks.
    """
    # Decode byte stream to UTF-8 text
    lines = [line.split(' ') for line in data.decode("utf-8").splitlines()]
    # Convert to float
    for line_idx in range(len(lines)):
        for val_idx in range(len(lines[line_idx])):
            lines[line_idx][val_idx] = float(lines[line_idx][val_idx])
    # Convert to tensor
    values = torch.tensor(lines)
    return values


def decode_face_synthetics(pipe: IterDataPipe):
    """ Decodes the components of the dataset "dataset". """
    pipe = pipe.map(Dispatcher({"ldmks": landmark_decoder,
                                "img": ImageDecoder('torchrgb'),
                                "seg": ImageDecoder('torchl8')}))
    return pipe


def main():
    parser = ArgumentParser()
    parser.add_argument("--in_dir",
                        type=str,
                        default="/ds-av/public_datasets/face_synthetics/td",
                        help="The path to the directory containing the sharded dataset.")
    args = parser.parse_args()

    # Make sure to create the factory only once. It reads the metadata file at construction time.
    factory = BasePipelineCreator(args.in_dir)

    print("Here are some examples of the dataset statistics that you can query:\n")
    for subset in factory.get_subsets():
        groups = factory.get_component_groups(subset)
        stats = factory.get_component_groups_stats(subset)
        shard_len = factory.get_average_shard_sample_count(subset)
        print(f"Subset: {subset}")
        print(f" average number of samples per shard: {shard_len}")
        for group in groups:
            print(f" component group {group}: {stats[group]}")

    # Choose the subset for which you wish to create a data streaming pipeline.
    subset = "train"
    # Make the shuffle buffer a multiple of the shard size. The multiplier may be chosen according to the batch size.
    multiplier = 2
    shard_size = factory.get_average_shard_sample_count(subset)
    # Make an educated guess on a good size for the shuffle buffer using the meta-data.
    shuffle_buffer_size = int(multiplier * shard_size)
    # Using the metadata created in the conversion process, the streaming pipeline can be created automatically.
    pipe = factory.create_datapipe(subset, shuffle_buffer_size, shuffle_shards=subset == "train", max_samples=5)
    # Decode the components of the dataset. Placing it in a function makes it reusable.
    pipe = decode_face_synthetics(pipe)

    print("\nThis is an example of what the pipeline output looks like."
          "\nOnly the file paths are printed, not the actual data.\n")

    for sample in pipe:
        paths = [component[0] for component in sample]
        print("Sample:", paths)


if __name__ == "__main__":
    main()
