##############################################################################
# Copyright (c) 2024 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, June 2024
##############################################################################
"""doc
# datapipes.decoders.pickle_decoder

> Decode pickle files.
"""
import pickle


def decode_pickle(file_path: str, data: bytes):
    """ Decode a pickled object into a Python object.

    :param file_path: The path to the file in the dataset. Ignored.
    :param data: The pickled object as an array of bytes.
    :return: The decoded object.
    """
    return pickle.loads(data)
