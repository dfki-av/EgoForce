##############################################################################
# Copyright (c) 2022 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, December 2022
##############################################################################
from argparse import ArgumentParser

import numpy as np
from torch.utils.data import DataLoader

from datapipes.base_pipeline import BasePipelineCreator
from datapipes.decoders.image_decoder import ImageDecoder
from datapipes.decoders.json_decoder import decode_json
from datapipes.utils.collation_functions import collate_batch_as_dict
from datapipes.utils.dispatcher import Dispatcher
from datapipes.versions import api_version, metadata_file_format_version


def main():
    parser = ArgumentParser()
    parser.add_argument("--in_dir",
                        type=str,
                        default="/ds-av/public_datasets/unreal-ego/td/randomized",
                        help="The path to the directory containing the sharded dataset.")
    args = parser.parse_args()

    print(f"API-Version: {api_version}")
    print(f"File Format Version: {metadata_file_format_version}")

    # Make sure to create the factory only once. It reads the metadata file at construction time.
    factory = BasePipelineCreator(args.in_dir)

    print("Here are some examples of the dataset statistics that you can query:\n")
    for subset in factory.get_subsets():
        groups = factory.get_component_groups(subset)
        stats = factory.get_component_groups_stats(subset)
        shard_len = factory.get_average_shard_sample_count(subset)
        print(f"Subset: {subset}")
        print(f" avg shard len: {shard_len}")
        for group in groups:
            print(f" component group {group}: {stats[group]['all_components']}")

    # Choose the subset for which you wish to create a data streaming pipeline.
    subset = "train"
    # Make the shuffle buffer a multiple of the shard size. The multiplier may be chosen according to the batch size.
    multiplier = 2
    shard_size = factory.get_average_shard_sample_count(subset)
    # Make an educated guess on a good size for the shuffle buffer using the meta-data.
    num_workers = 1
    shuffle_buffer_size = int(multiplier * shard_size / num_workers)
    # Using the metadata created in the conversion process, the streaming pipeline can be created automatically.
    pipe = factory.create_datapipe(subset, shuffle_buffer_size,
                                   components=["left_rgb_256", "right_rgb_256", "annotations"],
                                   shuffle_shards=subset == "train", max_samples=6)
    # Decode the components of the dataset.
    pipe = pipe.map(Dispatcher({"left_depth": ImageDecoder('l'),
                                "right_depth": ImageDecoder('l'),
                                "left_rgb_256": ImageDecoder('rgb'),
                                "right_rgb_256": ImageDecoder('rgb'),
                                "annotations": decode_json}))

    print("\nThis is an example of what the pipeline output looks like."
          "\nFor each sample component the shape of the tensor is printed.\n")

    for batch in DataLoader(pipe, batch_size=2, num_workers=num_workers, collate_fn=collate_batch_as_dict):
        for key, val in batch.items():
            if isinstance(val, np.ndarray):
                print(key, " shape:", val.shape)
            else:
                print(key, " len:", len(val))


if __name__ == "__main__":
    main()
