import io
import unittest

import numpy as np
from PIL import Image

from omni_tokenizer.image import prepare_image, target_size


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


if __name__ == "__main__":
    unittest.main()
