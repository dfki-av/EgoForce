[Back to Overview](../../README.md)

# datapipes.utils.state_wrapper

> The StateWrapper allows to wrap data along with information about the state the data is in. It can be used to avoid
> duplication of data in sequence pipelines which can occur in sequence pipelines if unwrapped data is transformed.

## *class* **StateWrapper** [[src]](../../../datapipes/utils/state_wrapper.py#L16)
Wraps data and attaches state information to it. The wrapper can be used to avoid duplication of samples in
sequence pipelines where the data for one frame (point in time) is incorporated into multiple samples via a
temporal sliding window filter.
