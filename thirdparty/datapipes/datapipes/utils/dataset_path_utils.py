from __future__ import annotations

import os.path

from torch.utils.data.datapipes.datapipe import DataChunk


def get_sample_id(sample: DataChunk | tuple) -> str:
    """ Extracts the part of the file path that uniquely identifies a sample within a subset.

    This will work with any dataset that follows the canonical file name structure:
    {sequence.}sample.sample_component.file_extension

    The results will be: {sequence.}sample

    If the file name includes a sequence ID then the combination of sequence ID and sample name must identify a sample
    uniquely. Otherwise, the sample name must be unique on its own.

    :param sample: A tuple consisting of file path and file content. The file path is used to determine to which sample
     the file belongs to.
    :return: The identifier used to group the files into samples.
    """
    bn = os.path.basename(sample[0])
    return ".".join(bn.split(".")[:-2])


def get_sequence_id(sample: DataChunk | tuple) -> str:
    """ Extracts the part of the file path that uniquely identifies a sequence within a subset.

    This will work with any dataset that follows the canonical file name structure:
    {sequence.}sample.sample_component.file_extension

    :param sample: A tuple consisting of file path and file content. The file path is used to determine to which sequence
     the file belongs to.
    :return: The identifier used to group the files into sequences or an empty string if not sequence IDs are present.
    """
    first_element = sample[0]
    if isinstance(first_element, tuple):
        first_element = first_element[0]
    name_elements = os.path.basename(first_element).split(".")
    if len(name_elements) > 3:
        return name_elements[0]
    return ""


def get_ids_from_path(file_path: str) -> list[str]:
    """
    Extracts the all identifiers needed to uniquely identify the sample component in the file specified by file_path.

    :param file_path: A full path or filename.
    :return: The subset ID, sequence ID, sample ID, component ID and file extension of the file specified by file_path.
    """
    path_elements = file_path.split(os.path.sep)
    filename_elements = path_elements[-1].split(".")
    if len(filename_elements) < 4:
        return [path_elements[-3], "", *filename_elements]
    return [path_elements[-3], *filename_elements]


def get_component_name(file_path: str) -> str:
    """
    Extracts the component identifier from the path of a sample component.

    :param file_path: A full path or filename.
    :return: The name of the component stored in the file specified by file_path.
    """
    return get_ids_from_path(file_path)[-2]


def get_file_extension(file_path: str) -> str:
    """
    Extracts the file extension from the path of a sample component.

    :param file_path: A full path or filename.
    :return: The file extension of the file specified by file_path.
    """
    return get_ids_from_path(file_path)[-1]
