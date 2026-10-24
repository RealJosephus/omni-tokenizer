import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.convert import HEADER, check_package, inspect_model
from tools.make_fixture import bf16, generate


class BFloat16FixtureTests(unittest.TestCase):
    def test_scalar_and_array_rounding_preserve_float32_and_ties_to_even(self):
        values = np.array([0, 1.00390625, 1.01171875, -1.00390625], np.float32)
        expected = np.array([0, 1, 1.015625, -1], np.float32)
        rounded = bf16(values)
        self.assertEqual(rounded.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(rounded, expected)
        for value, wanted in zip(values, expected):
            scalar = bf16(value)
            self.assertEqual(scalar.shape, ())
            self.assertEqual(scalar.dtype, np.dtype(np.float32))
            self.assertEqual(scalar, wanted)


class ModelPackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        generate(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_synthetic_packages_pass_integrity_check(self):
        for kind, tensors in (("audio", 36), ("image", 30)):
            result = check_package(self.root / kind)
            self.assertEqual(result["kind"], kind)
            self.assertEqual(result["tensor_count"], tensors)
            self.assertEqual(result["status"], "ok")

    def test_data_corruption_is_detected(self):
        path = self.root / "image" / "model.otm"
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            check_package(path)

    def test_explicit_model_filename_is_checked(self):
        directory = self.root / "image"
        path = directory / "renamed.otm"
        data = bytearray((directory / "model.otm").read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            check_package(path)

    def test_tensor_index_cannot_point_into_header(self):
        path = self.root / "audio" / "model.otm"
        data = bytearray(path.read_bytes())
        struct.pack_into("<Q", data, HEADER.size + 16, 0)
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "overlap"):
            inspect_model(path)

    def test_truncated_payload_is_rejected(self):
        path = self.root / "image" / "model.otm"
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "outside"):
            inspect_model(path)

    def test_manifest_must_describe_actual_index(self):
        path = self.root / "audio" / "manifest.json"
        data = json.loads(path.read_text())
        data["tensors"][0]["shape"][0] += 1
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "index"):
            check_package(path.parent)


if __name__ == "__main__":
    unittest.main()
