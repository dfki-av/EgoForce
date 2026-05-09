[Back to Overview](../../README.md)

# datapipes.utils.dispatcher

> The dispatcher can be used to map transformation functions to individual elements of a sample, e.g. sample components.

## *class* **Dispatcher** [[src]](../../../datapipes/utils/dispatcher.py#L22)
Creates an object that applies the specified transformations to the respective components of samples.

* **transformations**: A dictionary mapping a key to a transformation function. By default, the key is the
 component identifier.
* **in_state**: The expected instate of the data if it is encapsulated in a StateWrapper.
* **out_state**: The state of the data in the StateWrapper after it has been processed by the specified
 transformation function.
* **key_fn**: The function that shall be used to extract the key for indexing into the transformations
 dictionary from the file_path of each component. Defaults to a function that extracts the component identifier.
