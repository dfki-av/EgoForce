##############################################################################
# Copyright (c) 2024 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, August 2024
##############################################################################
from argparse import ArgumentParser

from torch.utils.data.dataset import Dataset

from datapipes.decoders.image_decoder import ImageDecoder
from datapipes.decoders.json_decoder import decode_json
from datapipes.in_memory_dataset import InMemoryDataset


class UnrealEgoDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 subset: str | list[str],
                 components: list[str]):
        # a decoder for each sample component
        decoders = {"left_depth": ImageDecoder('l'),
                    "right_depth": ImageDecoder('l'),
                    "left_rgb_256": ImageDecoder('rgb'),
                    "right_rgb_256": ImageDecoder('rgb'),
                    "annotations": decode_json}
        self.dataset = InMemoryDataset(dataset_path, subset, decoders, components)

    def __getitem__(self, index: int):
        # each sample is a dict mapping from sample component name to the (decoded) data of the sample component
        # decoding is done on access to reduce the memory footprint
        sample = self.dataset[index]
        # add more operations here, e.g., pre-processing, data augmentation, etc.
        return sample

    def __len__(self):
        return len(self.dataset)


def main():
    parser = ArgumentParser()
    parser.add_argument("--in_dir",
                        type=str,
                        default="/ds-av/public_datasets/unreal-ego/td",
                        help="The path to the directory containing the sharded dataset.")
    args = parser.parse_args()
    # the data split that you wish to load (typically one of "train", "val" or "test")
    split = "val"
    # the sample components that you wish to load
    components = ["left_rgb_256", "right_rgb_256", "annotations"]
    # warning: this loads the *entire* split of this dataset into memory
    unreal_ego_dataset = UnrealEgoDataset(args.in_dir, split, components)

    # samples may be accessed arbitrarily
    index = 7
    sample = unreal_ego_dataset[index]
    print(f"sample {index}:\n")
    print(sample)


if __name__ == "__main__":
    main()
