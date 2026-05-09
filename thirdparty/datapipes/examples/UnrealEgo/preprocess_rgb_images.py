##############################################################################
# Copyright (c) 2022 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, March 2023
##############################################################################
from __future__ import annotations

import io
import os.path
from argparse import ArgumentParser
from typing import IO

from PIL import Image

from datapipes.archive_processor import ArchiveProcessor


class MyArchiveProcessor(ArchiveProcessor):
    def include_archive(self, archive_path: str) -> bool:
        # avoid recursion in case preprocessed archives already exist
        return "preprocessed" not in archive_path

    def transform_archive_name(self, src_archive_name: str) -> str:
        return f"{os.path.splitext(src_archive_name)[0]}_rgb_256.zip"

    def include_file(self, archive_path: str, filepath: str) -> bool:
        return os.path.basename(filepath).startswith("final")

    def process_file(self,
                     src_archive_path: str,
                     filepath: str,
                     file_data: IO[bytes]) -> list[tuple[str, io.BytesIO]]:
        out_name = f"{os.path.splitext(filepath)[0]}.webp"
        out_data = io.BytesIO()

        with Image.open(file_data, formats=["PNG"]) as pil_img:
            resized_img = pil_img.resize((256, 256), Image.Resampling.LANCZOS)
            resized_img.save(out_data, format="webp", lossless=True)
            resized_img.close()
        return [(out_name, out_data)]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in_dir",
                        type=str,
                        required=True,
                        help="The path to the input directory containing the source dataset.")
    parser.add_argument("--out_dir",
                        type=str,
                        required=True,
                        help="The path to the output directory. The preprocessed files will be stored there.")
    args = parser.parse_args()
    # Run the dataset conversion.
    processor = MyArchiveProcessor()
    processor.process_dataset(args.in_dir, args.out_dir)
