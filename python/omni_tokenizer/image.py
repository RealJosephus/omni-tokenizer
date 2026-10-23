"""Pillow preprocessing for the fixed VQSigLIP encoder."""

from __future__ import annotations

import io
import os
from dataclasses import asdict, dataclass, replace
from typing import BinaryIO

import numpy as np
from PIL import Image

DEFAULT_MIN_SHORT_EDGE = 384
DEFAULT_MAX_LONG_EDGE = 1152
DEFAULT_MAX_PIXELS = 64_000_000


def validate_image_options(value: dict) -> dict:
    """Validate partial overrides without replacing instance defaults."""
    if not isinstance(value, dict) or set(value) - {
        "resize",
        "min_short_edge",
        "max_long_edge",
    }:
        raise ValueError(
            "image_options accepts resize, min_short_edge, and max_long_edge"
        )
    for name, option in value.items():
        if name == "resize":
            if type(option) is not bool:
                raise ValueError("resize must be a boolean")
        elif (
            type(option) is not int or option < 16 or option % 16 or option > 0xFFFFFFF0
        ):
            raise ValueError(
                f"{name} must be a positive multiple of 16 within uint32 range"
            )
    return dict(value)


@dataclass(frozen=True)
class ImageOptions:
    resize: bool = True
    min_short_edge: int = DEFAULT_MIN_SHORT_EDGE
    max_long_edge: int = DEFAULT_MAX_LONG_EDGE

    def __post_init__(self):
        validate_image_options(asdict(self))

    def updated(self, values: dict) -> ImageOptions:
        return replace(self, **validate_image_options(values))

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def profile(self) -> str:
        if not self.resize:
            return "image-rgb-native-size-v1"
        if (self.min_short_edge, self.max_long_edge) == (
            DEFAULT_MIN_SHORT_EDGE,
            DEFAULT_MAX_LONG_EDGE,
        ):
            return "image-pillow-v1"
        return f"image-pillow-v1:min={self.min_short_edge}:max={self.max_long_edge}"


@dataclass(frozen=True)
class PreparedImage:
    """Normalized CHW pixels and sizes, with all sizes in height/width order."""

    pixels: np.ndarray
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    grid: tuple[int, int]


def target_size(
    width: int,
    height: int,
    *,
    min_short_edge: int = DEFAULT_MIN_SHORT_EDGE,
    max_long_edge: int = DEFAULT_MAX_LONG_EDGE,
) -> tuple[int, int]:
    """Scale the short edge, cap the long edge, then round to whole patches."""
    ImageOptions(min_short_edge=min_short_edge, max_long_edge=max_long_edge)
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    target_width, target_height = width, height
    short = min(target_width, target_height)
    if short < min_short_edge:
        target_width = target_width * (min_short_edge / short)
        target_height = target_height * (min_short_edge / short)
    long = max(target_width, target_height)
    if long > max_long_edge:
        target_width = target_width * (max_long_edge / long)
        target_height = target_height * (max_long_edge / long)
    target_width = int(round(target_width / 16)) * 16
    target_height = int(round(target_height / 16)) * 16
    if not target_width or not target_height:
        raise ValueError("Image aspect ratio rounds a processed dimension to zero")
    return target_width, target_height


def prepare_image(
    source: str | os.PathLike[str] | bytes | bytearray | BinaryIO | Image.Image,
    *,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    resize: bool = True,
    min_short_edge: int = DEFAULT_MIN_SHORT_EDGE,
    max_long_edge: int = DEFAULT_MAX_LONG_EDGE,
) -> PreparedImage:
    """Decode one image with the reference RGB, bicubic, and normalization rules.

    EXIF orientation is not applied, matching the reference. Animated files use
    the selected frame for a Pillow object and the first frame for a file.
    The input pixel limit is checked before decoding compressed pixel data.
    With resize=False, preserve pixels and require whole 16-pixel patches,
    at most max_long_edge pixels per side. max_pixels bounds both decoded and
    processed images, independently of the resize policy.
    """
    ImageOptions(resize, min_short_edge, max_long_edge)
    if type(max_pixels) is not int or max_pixels <= 0:
        raise ValueError("max_pixels must be positive")

    def process(image: Image.Image) -> PreparedImage:
        width, height = image.size
        if width * height > max_pixels:
            raise ValueError(f"Image exceeds the input limit of {max_pixels} pixels")
        if resize:
            processed_width, processed_height = target_size(
                width,
                height,
                min_short_edge=min_short_edge,
                max_long_edge=max_long_edge,
            )
        else:
            if any(
                size < 16 or size > max_long_edge or size % 16
                for size in (width, height)
            ):
                raise ValueError(
                    f"Image dimensions must be multiples of 16 between 16 and {max_long_edge}"
                )
            processed_width, processed_height = width, height
        if processed_width * processed_height > max_pixels:
            raise ValueError(
                f"Processed image exceeds the limit of {max_pixels} pixels"
            )
        rgb = image.convert("RGB")
        resized = (
            rgb.resize(
                (processed_width, processed_height), resample=Image.Resampling.BICUBIC
            )
            if resize
            else rgb
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
