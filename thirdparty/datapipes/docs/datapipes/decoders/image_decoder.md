[Back to Overview](../../README.md)

# datapipes.decoders.image_decoder

> Decode images to numpy arrays, torch tensors or PIL images.

## *class* **ImageDecoder** [[src]](../../../datapipes/decoders/image_decoder.py#L53)
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
