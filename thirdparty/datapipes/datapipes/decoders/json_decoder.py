##############################################################################
# Copyright (c) 2022 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, December 2022
##############################################################################
"""doc
# datapipes.decoders.json_decoder

> Decode JSON files.

"""
from __future__ import annotations

import json


def decode_json(file_path: str, data: bytes) -> dict | list:
    """ Decode a JSON document into a Python dictionary or list.

    :param file_path: The path to the file in the dataset. Ignored.
    :param data: The JSON document as a byte array.
    :return: The decoded JSON as a dictionary or list.
    """
    decoded_json = json.loads(data)
    return decoded_json
