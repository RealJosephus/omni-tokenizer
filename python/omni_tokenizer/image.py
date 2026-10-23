"""Pillow preprocessing for the fixed VQSigLIP encoder."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PreparedImage:
    """Normalized CHW pixels and sizes, with all sizes in height/width order."""

    pixels: np.ndarray
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    grid: tuple[int, int]


def target_size(width: int, height: int) -> tuple[int, int]:
    """Return reference width/height after scaling and ties-to-even rounding."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    target_width, target_height = width, height
    short = min(target_width, target_height)
    if short < 384:
        target_width = target_width * (384 / short)
        target_height = target_height * (384 / short)
    long = max(target_width, target_height)
    if long > 1152:
        target_width = target_width * (1152 / long)
        target_height = target_height * (1152 / long)
    target_width = int(round(target_width / 16)) * 16
    target_height = int(round(target_height / 16)) * 16
    if not target_width or not target_height:
        raise ValueError("Image aspect ratio rounds a processed dimension to zero")
    return target_width, target_height


def prepare_image(
    source: str | os.PathLike[str] | bytes | bytearray | BinaryIO | Image.Image,
    *,
    max_pixels: int = 64_000_000,
) -> PreparedImage:
    """Decode one image with the reference RGB, bicubic, and normalization rules.

    EXIF orientation is not applied, matching the reference. Animated files use
    the selected frame for a Pillow object and the first frame for a file.
    The input pixel limit is checked before decoding compressed pixel data.
    """
    if max_pixels <= 0:
        raise ValueError("max_pixels must be positive")

    def process(image: Image.Image) -> PreparedImage:
        width, height = image.size
        if width * height > max_pixels:
            raise ValueError(f"Image exceeds the input limit of {max_pixels} pixels")
        processed_width, processed_height = target_size(width, height)
        rgb = image.convert("RGB")
        resized = rgb.resize(
            (processed_width, processed_height), resample=Image.Resampling.BICUBIC
        )
        pixels = np.array(resized, dtype=np.uint8).transpose(2, 0, 1)
        pixels = np.ascontiguousarray(pixels, dtype=np.float32)
        # Keep the same operation order as ToTensor followed by Normalize.
        pixels /= np.float32(255.0)
        pixels -= np.float32(0.5)
        pixels /= np.float32(0.5)
        return PreparedImage(
            pixels=pixels,
            original_size=(height, width),
            processed_size=(processed_height, processed_width),
            grid=(processed_height // 16, processed_width // 16),
        )

    if isinstance(source, Image.Image):
        return process(source)
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    with Image.open(source) as opened:
        return process(opened)
