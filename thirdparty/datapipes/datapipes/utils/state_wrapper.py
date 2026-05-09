##############################################################################
# Copyright (c) 2023 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, July 2023
##############################################################################
"""doc
# datapipes.utils.state_wrapper

> The StateWrapper allows to wrap data along with information about the state the data is in. It can be used to avoid
> duplication of data in sequence pipelines which can occur in sequence pipelines if unwrapped data is transformed.
"""
from __future__ import annotations

from typing import Any


class StateWrapper:
    """
    Wraps data and attaches state information to it. The wrapper can be used to avoid duplication of samples in
    sequence pipelines where the data for one frame (point in time) is incorporated into multiple samples via a
    temporal sliding window filter.
    """
    __slots__ = "data", "state"

    def __init__(self, data: Any, state: str):
        """ Creates a new wrapper for the data and attaches a state descriptor to it.

        :param data: The data that shall be wrapped.
        :param state: The state the data is in.
        """
        self.data = data
        self.state = state
