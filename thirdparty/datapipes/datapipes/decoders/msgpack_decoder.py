##############################################################################
# Copyright (c) 2024 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, June 2024
##############################################################################
"""doc
# datapipes.decoders.msgpack_decoder

> Decode message pack files.

"""
from __future__ import annotations

import msgpack


def decode_msgpack(file_path: str, data: bytes) -> dict | list:
    """ Decode a message pack object into a Python dictionary or list.

    :param file_path: The path to the file in the dataset. Ignored.
    :param data: The message pack object as a byte array.
    :return: The decoded message pack as a dictionary or list.
    """
    decoded_msgpack = msgpack.loads(data)
    return decoded_msgpack
