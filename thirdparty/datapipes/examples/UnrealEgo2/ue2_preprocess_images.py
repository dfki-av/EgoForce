##############################################################################
# Copyright (c) 2024 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, April 2024
##############################################################################
from __future__ import annotations

import io
import os.path
from argparse import ArgumentParser
from typing import IO

from PIL import Image

from datapipes.archive_processor import ArchiveProcessor


class MyArchiveProcessor(ArchiveProcessor):
    def __init__(self, resolution: int):
        """ Creates a custom archive processor that scales images to a specific square resolution and converts them to
        the WebP format.

        :param resolution: The length of a single side of the square resolution to which the images will be rescaled to.
        """
        super().__init__(8, continue_on_errors=True)
        self.resolution = resolution

    def include_archive(self, archive_path: str) -> bool:
        # avoid recursion in case preprocessed archives already exist
        return "preprocessed" not in archive_path

    def transform_archive_name(self, src_archive_name: str) -> str:
        if "depth" in self.in_dir:
            image_type = "depth"
        elif "rgb" in self.in_dir:
            image_type = "rgb"
        else:
            raise RuntimeError("Could not determine image type.")
        return f"{os.path.splitext(src_archive_name)[0]}_{image_type}_{self.resolution}.zip"

    def include_file(self, archive_path: str, filepath: str) -> bool:
        # Process only image files. (The dataset contains only PNGs.)
        return os.path.basename(filepath).endswith(".png")

    def process_file(self,
                     src_archive_path: str,
                     filepath: str,
                     file_data: IO[bytes]) -> list[tuple[str, io.BytesIO]]:
        out_name = f"{os.path.splitext(filepath)[0]}.webp"
        out_data = io.BytesIO()
        # Use pillow to rescale and encode as WebP
        with Image.open(file_data, formats=["PNG"]) as pil_img:
            resized_img = pil_img.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
            resized_img.save(out_data, format="webp", quality=98)
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
    parser.add_argument("--resolution",
                        type=int,
                        default=256,
                        help="The the width of the output images. Images are expected to be square.")
    args = parser.parse_args()
    # Run the dataset conversion.
    processor = MyArchiveProcessor(args.resolution)
    processor.process_dataset(args.in_dir, args.out_dir)
