##############################################################################
# Copyright (c) 2024 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, January 2024
##############################################################################
from __future__ import annotations

import sys


class Color:
    BLUE = "\033[94m" if sys.platform == "linux" else ""
    CYAN = "\033[96m" if sys.platform == "linux" else ""
    GREEN = "\033[92m" if sys.platform == "linux" else ""
    YELLOW = "\033[93m" if sys.platform == "linux" else ""
    RED = "\033[91m" if sys.platform == "linux" else ""


class Format:
    NORMAL = "\033[0m" if sys.platform == "linux" else ""
    BOLD = "\033[1m" if sys.platform == "linux" else ""
    UNDERLINE = "\033[4m" if sys.platform == "linux" else ""


def print_c(message: str, color: Color | Format, formatting: Format = "", *, end: str = "\n", flush: bool = False):
    print(f"{formatting}{color}{message}{Format.NORMAL}", end=end, flush=flush)
