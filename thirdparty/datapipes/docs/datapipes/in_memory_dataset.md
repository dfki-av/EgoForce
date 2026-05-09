[Back to Overview](../README.md)

# datapipes.in_memory_dataset: random access to datasets that fit into memory

> Provides index-based access to the samples of datasets stored in main memory.
> The dataset is read only once and kept in a separate process which serves the sample access requests.
> This means that the dataset must fit into main memory.

## *class* **InMemoryDataset**(Sequence) [[src]](../../datapipes/in_memory_dataset.py#L33)
The InMemoryDataset class handles the logic needed to orchestrate the interactions between the workers of a
dataloader and the in-memory database holding a specific subset of a dataset in memory.
