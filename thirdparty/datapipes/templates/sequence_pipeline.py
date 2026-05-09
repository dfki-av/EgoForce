##############################################################################
# Copyright (c) 2022 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, December 2022
##############################################################################
from argparse import ArgumentParser

from torch.utils.data.datapipes.datapipe import IterDataPipe

from datapipes.sequence_pipeline import SequencePipelineCreator
from datapipes.utils.dispatcher import Dispatcher


def decode_dataset(pipe: IterDataPipe):
    """ Decodes the components of the dataset "dataset". """
    # TODO map components to decoding functions
    decoder_map = {}
    # apply the decoding functions to each sample
    pipe = pipe.map(Dispatcher(decoder_map))
    return pipe


def main():
    parser = ArgumentParser()
    parser.add_argument("--in_dir",
                        type=str,
                        required=True,
                        help="The path to the directory containing the sharded dataset.")
    args = parser.parse_args()

    # Make sure to create the factory only once. It reads the metadata file at construction time.
    factory = SequencePipelineCreator(args.in_dir)

    print("Here are some examples of the dataset statistics that you can query:\n")
    for subset in factory.get_subsets():
        groups = factory.get_component_groups(subset)
        stats = factory.get_component_groups_stats(subset)
        shard_len = factory.get_average_shard_sample_count(subset)
        print(f"Subset: {subset}")
        print(f" avg shard len: {shard_len}")
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
    pipe = factory.create_datapipe(subset, shuffle_buffer_size, shuffle_shards=True, temporal_sliding_window_size=3)
    # Decode the components of the dataset. Placing it in a function makes it reusable.
    pipe = decode_dataset(pipe)

    print("\nThis is an example of what the pipeline output looks like."
          "\nOnly the file paths are printed, not the actual data.\n")

    for samples in pipe:
        paths = [component[0] for sample in samples for component in sample]
        print("Sample:", paths)


if __name__ == "__main__":
    main()
