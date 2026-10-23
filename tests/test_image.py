import io
import unittest

import numpy as np
from omni_tokenizer.image import ImageOptions, prepare_image, target_size
from PIL import Image


class ImagePreprocessingTests(unittest.TestCase):
    def test_resize_rules_and_ties_to_even(self):
        for original, expected in (
            ((384, 384), (384, 384)),
            ((192, 288), (384, 576)),
            ((1500, 500), (1152, 384)),
            ((392, 392), (384, 384)),
            ((408, 408), (416, 416)),
            ((100, 1000), (112, 1152)),
        ):
            self.assertEqual(target_size(*original), expected)
        with self.assertRaisesRegex(ValueError, "zero"):
            target_size(1, 100_000)
        with self.assertRaisesRegex(ValueError, "positive"):
            target_size(0, 100)

    def test_channel_order_and_normalization(self):
        image = Image.new("RGB", (384, 384), (0, 127, 255))
        prepared = prepare_image(image)
        self.assertEqual(prepared.original_size, (384, 384))
        self.assertEqual(prepared.grid, (24, 24))
        self.assertEqual(prepared.pixels.dtype, np.float32)
        self.assertTrue(prepared.pixels.flags.c_contiguous)
        expected = (np.array([0, 127, 255], np.float32) / 255 - 0.5) / 0.5
        np.testing.assert_array_equal(prepared.pixels[:, 100, 100], expected)

    def test_rgb_conversion_bytes_and_pillow_resize(self):
        values = np.arange(113 * 169, dtype=np.uint8).reshape(113, 169)
        image = Image.fromarray(values)
        output = io.BytesIO()
        image.save(output, format="PNG")
        prepared = prepare_image(output.getvalue())
        self.assertEqual(prepared.original_size, (113, 169))
        expected_image = image.convert("RGB").resize(
            target_size(169, 113), resample=Image.Resampling.BICUBIC
        )
        expected = np.asarray(expected_image).transpose(2, 0, 1).astype(np.float32)
        expected = (expected / np.float32(255) - np.float32(0.5)) / np.float32(0.5)
        np.testing.assert_array_equal(prepared.pixels, expected)

    def test_pixel_limit(self):
        with self.assertRaisesRegex(ValueError, "limit"):
            prepare_image(Image.new("RGB", (100, 100)), max_pixels=9999)

    def test_native_size_preserves_cropped_pixels_without_upscaling(self):
        values = np.arange(80 * 112 * 3, dtype=np.uint8).reshape(80, 112, 3)
        image = Image.fromarray(values).crop((7, 9, 103, 73))
        prepared = prepare_image(image, resize=False)
        self.assertEqual(prepared.original_size, (64, 96))
        self.assertEqual(prepared.processed_size, (64, 96))
        self.assertEqual(prepared.grid, (4, 6))
        expected = np.asarray(image).transpose(2, 0, 1).astype(np.float32)
        expected = (expected / np.float32(255) - np.float32(0.5)) / np.float32(0.5)
        np.testing.assert_array_equal(prepared.pixels, expected)
        for size in ((17, 16), (16, 15), (1168, 16)):
            with self.assertRaisesRegex(ValueError, "multiples of 16"):
                prepare_image(Image.new("RGB", size), resize=False)

    def test_custom_resize_targets_and_long_edge_precedence(self):
        self.assertEqual(target_size(1800, 1200), (1152, 768))
        self.assertEqual(target_size(1800, 1200, max_long_edge=1536), (1536, 1024))
        self.assertEqual(
            target_size(192, 288, min_short_edge=256, max_long_edge=2048), (256, 384)
        )
        self.assertEqual(
            target_size(192, 288, min_short_edge=512, max_long_edge=96), (64, 96)
        )
        prepared = prepare_image(Image.new("RGB", (1280, 1296)), max_long_edge=2048)
        self.assertEqual(prepared.processed_size, (1296, 1280))
        raw = prepare_image(
            Image.new("RGB", (1536, 32)), resize=False, max_long_edge=1536
        )
        self.assertEqual(raw.processed_size, (32, 1536))

    def test_pixel_budget_checks_upscaled_output_before_allocation(self):
        with self.assertRaisesRegex(ValueError, "Processed image"):
            prepare_image(
                Image.new("RGB", (16, 16)),
                min_short_edge=1024,
                max_long_edge=2048,
                max_pixels=4096,
            )

    def test_resize_options_are_strict_and_have_distinct_profiles(self):
        for invalid in (True, 0, 15, 17, 384.0, "384", 2**32):
            with self.assertRaises(ValueError):
                ImageOptions(min_short_edge=invalid)
            with self.assertRaises(ValueError):
                ImageOptions(max_long_edge=invalid)
        with self.assertRaises(ValueError):
            ImageOptions(resize=1)
        with self.assertRaises(ValueError):
            ImageOptions().updated({"max_image_pixels": 100})
        self.assertEqual(ImageOptions().profile, "image-pillow-v1")
        self.assertEqual(
            ImageOptions(max_long_edge=2048).profile, "image-pillow-v1:min=384:max=2048"
        )


if __name__ == "__main__":
    unittest.main()
