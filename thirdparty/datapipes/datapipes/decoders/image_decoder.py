##############################################################################
# Copyright (c) 2022 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, February 2023
# and by Suresh Guttikonda <Suresh.Guttikonda@dfki.de>, February 2023
# based on https://github.com/pytorch/pytorch/blob/master/torch/utils/data/datapipes/utils/decoder.py#L114
##############################################################################
"""doc
# datapipes.decoders.image_decoder

> Decode images to numpy arrays, torch tensors or PIL images.

"""
from __future__ import annotations
import io

import numpy as np
import torch
import cv2

_image_specs = {
    "u8": ("numpy", "uint8", "raw"),
    "l8": ("numpy", "uint8", "l"),
    "rgb8": ("numpy", "uint8", "rgb"),
    "rgba8": ("numpy", "uint8", "rgba"),

    "u": ("numpy", "float", "raw"),
    "l": ("numpy", "float", "l"),
    "i": ("numpy", "float", "i"),
    "rgb": ("numpy", "float", "rgb"),
    "rgba": ("numpy", "float", "rgba"),

    "torchu8": ("torch", "uint8", "raw"),
    "torchl8": ("torch", "uint8", "l"),
    "torchrgb8": ("torch", "uint8", "rgb"),
    "torchrgba8": ("torch", "uint8", "rgba"),

    "torchu": ("torch", "float", "raw"),
    "torchl": ("torch", "float", "l"),
    "torchi": ("torch", "float", "i"),
    "torch": ("torch", "float", "rgb"),
    "torchrgb": ("torch", "float", "rgb"),
    "torchrgba": ("torch", "float", "rgba"),

    "pilu": ("pil", None, "raw"),
    "pill": ("pil", None, "l"),
    "pili": ("pil", None, "i"),
    "pil": ("pil", None, "rgb"),
    "pilrgb": ("pil", None, "rgb"),
    "pilrgba": ("pil", None, "rgba"),
}


class ImageDecoder:
    """
    A custom variant of pyTorch's ImageHandler.

    It adds support for 16 bit single channel images and the WebP image format. Furthermore, it ensures that single
    channel images always have a channel dimension when converting them to numpy or torch. This fixes a crash when
    trying to load single channel images as torch tensors.

    How the image data shall be decoded is defined by the given `image_spec`.
    The `image_spec` specifies whether
    (1) the image is returned as a numpy array, torch tensor or PIL image,
    (2) the data type of the elements in the numpy array or torch tensor is uint8 or float32,
    (3) the image is converted to 8-bit grayscale (l), 16-bit grayscale (i), 8-bit RGB (rgb), 8-bit RGBA (rgba), or
        kept in the same format as in the image file (raw).

    The available choices are:
    - u8: numpy uint8 raw
    - l8: numpy uint8 l
    - rgb8: numpy uint8 rgb
    - rgba8: numpy uint8 rgba
    - u: numpy float raw
    - l: numpy float l
    - i: numpy float i
    - rgb: numpy float rgb
    - rgba: numpy float rgba
    - torchu8: torch uint8 raw
    - torchl8: torch uint8 l
    - torchrgb8: torch uint8 rgb
    - torchrgba8: torch uint8 rgba
    - torchu: torch float raw
    - torchl: torch float l
    - torchi: torch float i
    - torch: torch float rgb
    - torchrgb: torch float rgb
    - torchrgba: torch float rgba
    - pilu: pil None u
    - pill: pil None l
    - pil: pil None rgb
    - pilrgb: pil None rgb
    - pilrgba: pil None rgba
    """
    def __init__(self, image_spec: str, decode_truncated_images: bool = False):
        """

        :param image_spec: The specifier for the target image format.
        :param decode_truncated_images: Whether to allow (partial) decoding of incomplete images. If set to False
         (default) an exception will be raised if incomplete image data is encountered.
        """
        assert image_spec in list(_image_specs.keys()), "unknown image specification: {}".format(image_spec)
        self.image_spec = image_spec.lower()
        self.decode_truncated_images = decode_truncated_images

    def __call__(self, file_path: str, data: bytes):
        try:
            import PIL.Image
            if self.decode_truncated_images:
                import PIL.ImageFile
                PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
        except ImportError as e:
            raise ModuleNotFoundError("Package 'PIL' is required to be installed to decode images."
                                      "Please use `pip install Pillow` to install the package") from e

        a_type, e_type, mode = _image_specs[self.image_spec]

        assert a_type == 'numpy'
        assert e_type == 'uint8'
        assert mode == 'rgb'
        
        nparr = np.frombuffer(data, dtype=np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
    
        with io.BytesIO(data) as stream:
            print(f"[ImageDecoder] Decoding '{file_path}' as {a_type} {e_type} {mode}")
            image = PIL.Image.open(stream)
            try:
                image.load()
            except OSError as e:
                import sys
                print(f"[ImageDecoder] failed to decode '{file_path}'", file=sys.stderr)
                raise e

            if mode != "raw":
                image = image.convert(mode.upper())

            if a_type == "pil":
                return image

            image = np.asarray(image, dtype=np.uint16 if image.mode == "I" else np.uint8)
            # add missing channel dimension if case of single channel (grayscale) images
            if image.ndim == 2:
                image = image[:, :, np.newaxis]

            if a_type == "numpy":
                if e_type == "float":
                    image = image.astype("f") / np.iinfo(image.dtype).max
                return image

            if a_type == "torch":
                image = image.transpose((2, 0, 1))
                if e_type == "float":
                    max_val = np.iinfo(image.dtype).max
                    image = image.astype("f") / max_val
                return torch.tensor(image)
            return None
