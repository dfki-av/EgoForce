##############################################################################
# Copyright (c) 2023 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, October 2023
##############################################################################
import time
from argparse import ArgumentParser

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from datapipes.in_memory_dataset import InMemoryDataset


class MyDataset(Dataset):
    """
    A classic map-style PyTorch dataset that reads all samples into memory at creation time.

    Since the entire dataset is stored in memory, random access to any sample is possible.
    """
    def __init__(self, dataset_path: str, subset: str):
        """ Creates a new custom dataset.

        :param dataset_path: Path to the directory that holds the sharded dataset that shall be loaded.
        :param subset: The subset(s) that shall be loaded into memory. If multiple subsets are specified, then they
        will be merged into a single set.
        """
        # Provides access to the samples of the dataset split via the Sequence interface.
        self.dataset = InMemoryDataset(dataset_path, subset)

    def __getitem__(self, item: int):
        # retrieve the requested sample from the in-memory dataset
        sample = self.dataset[item]
        # TODO implement further transformations like data augmentation here
        return sample

    def __len__(self):
        return len(self.dataset)


def main(dataset_path: str):
    my_train_dataset = MyDataset(dataset_path, "train")
    train_dataloader = DataLoader(my_train_dataset, num_workers=2)
    start = time.perf_counter()
    count = 0
    total_time = 0
    for sample in train_dataloader:
        delta = time.perf_counter() - start
        # do not consider the first access when computing access times as it is much slower than the rest
        if count > 0:
            total_time += delta
        count += 1
        start = time.perf_counter()
    print("Average data access (and decoding) time:", round(total_time / (count - 1) * 1000, 2), "ms.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir",
                        type=str,
                        required=True,
                        help="The path to the directory containing a sharded dataset.")
    args = parser.parse_args()
    main(dataset_path=args.dataset_dir)
