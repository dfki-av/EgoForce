[Back to Overview](../../README.md)

# datapipes.utils.collation_functions

> Functions for collating multiple samples into batches. The individual components are provided in tuples or dicts.

## *def* **collate_batch_as_tuples** [[src]](../../../datapipes/utils/collation_functions.py#L31)
Collates individual samples (stored in a list of tuples) into batches.

Combines corresponding elements of individual samples. Tensors and numpy arrays will be stacked in the batch
dimension. Other data types are simply placed in lists. All aggregated components are then placed in a tuple in the
same order as they appear in the sample tuples. Additionally, a tuple with the file paths for each file in the
batch is returned as well.

* **samples**: A list of the samples that shall be combined into a batch.
* **returns**: The batched data in two tuples: The first tuple contains the filenames for each component.
 The second tuple contains the actual data of each component.

## *def* **collate_batch_as_dict** [[src]](../../../datapipes/utils/collation_functions.py#L63)
Collates individual samples (stored in a list of tuples) into batches.

Combines corresponding elements of individual samples. Tensors and numpy arrays will be stacked in the batch
dimension. Other data types are simply placed in lists.  All aggregated components are then placed in a dictionary
where the keys are the component names and the values are the stacked tensors (or lists in case of other types of
objects).

* **samples**: A list of the samples that shall be combined into a batch.
* **returns**: The batched data in a dictionary that maps the component names to the actual data for each component.

## *def* **collate_batch_as_dicts** [[src]](../../../datapipes/utils/collation_functions.py#L94)
Collates individual samples (stored in a list of tuples) into batches.

Combines corresponding elements of individual samples. Tensors and numpy arrays will be stacked in the batch
dimension. Other data types are simply placed in lists.  All aggregated components are then placed in a dictionary
where the keys are the component names and the values are the stacked tensors (or lists in case of other types of
objects).

* **samples**: A list of the samples that shall be combined into a batch.
* **returns**: The batched data in two dicts: The first dict contains a mapping of component names to filenames.
 The second dict maps the component names to the actual data for each component.

## *def* **collate_dict** [[src]](../../../datapipes/utils/collation_functions.py#L128)
Collates individual samples from a list of dicts into a single dict containing batches.

Combines corresponding elements of individual samples. Tensors and numpy arrays will be stacked in the batch
dimension. Other data types are simply placed in lists. All elements are kept with their original keys.

* **samples**: The samples that shall be combined into a batch.
* **returns**: The dictionary, where all elements with the same key in the input dicts have been stacked/batched.

## *def* **collate_sequences_as_tuples** [[src]](../../../datapipes/utils/collation_functions.py#L153)
Collates individual samples (stored in lists of sequences) into batches.

Combines corresponding elements of individual sequences. Tensors and numpy arrays will be stacked in the batch and
sequence dimensions. Other data types are simply placed in nested lists. All aggregated components are then placed
in a tuple in the same order as they appear in the sequence tuples.

* **sequences**: The sequences that shall be combined into a batch.
* **returns**: The batched data in two tuples: The first tuple contains the filenames for each component.
 The second tuple contains the actual data of each component.

## *def* **collate_sequences_as_dicts** [[src]](../../../datapipes/utils/collation_functions.py#L208)
Collates individual samples (stored in lists of sequences) into batches.

Combines corresponding elements of individual sequences. Tensors and numpy arrays will be stacked in the batch and
sequence dimensions. Other data types are simply placed in nested lists. All aggregated components are then placed
in a dictionary where the keys are the component names and the values are the stacked tensors (or nested lists in
case of other types of objects).

* **sequences**: The samples that shall be combined into a batch.
* **returns**: The batched data in two dicts: The first dict maps from component names to the filenames.
 The second dict maps from the component names to the actual data for each component.

## *def* **sample_to_dict** [[src]](../../../datapipes/utils/collation_functions.py#L265)
Collates a single sample into a dictionary.

* **sample**: A tuple containing the sample components.
* **returns**: A dictionary mapping from component IDs to sample components.
