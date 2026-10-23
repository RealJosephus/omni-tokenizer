import hashlib
import io
from pathlib import Path
import tempfile
import unittest

from tools.fetch_checkpoints import CheckpointFile, fetch_file


class Response(io.BytesIO):
    def __init__(self, content, status=200, content_range=None):
        super().__init__(content)
        self.status = status
        self.headers = {"Content-Range": content_range} if content_range else {}


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data = bytes(range(256)) * 100
        self.spec = CheckpointFile("example/model", "a" * 40, "weights.bin", "model/weights.bin",
                                   len(self.data), hashlib.sha256(self.data).hexdigest())

    def test_download_verify_and_no_second_request(self):
        calls = []
        def open_url(request, timeout):
            calls.append(request.full_url)
            return Response(self.data)
        result = fetch_file(self.spec, self.root, opener=open_url)
        self.assertEqual(result["status"], "downloaded")
        self.assertEqual((self.root / self.spec.target).read_bytes(), self.data)
        self.assertEqual(fetch_file(self.spec, self.root, opener=open_url)["status"], "verified")
        self.assertEqual(len(calls), 1)
        self.assertFalse(list(self.root.rglob("*.part")))
        self.assertFalse(list(self.root.rglob("*.lock")))

    def test_resume_and_range_validation(self):
        destination = self.root / self.spec.target
        destination.parent.mkdir()
        partial = destination.with_name(destination.name + ".part")
        partial.write_bytes(self.data[:1234])
        def open_url(request, timeout):
            self.assertEqual(request.get_header("Range"), "bytes=1234-")
            return Response(self.data[1234:], 206, f"bytes 1234-{len(self.data)-1}/{len(self.data)}")
        self.assertEqual(fetch_file(self.spec, self.root, opener=open_url)["resumed_bytes"], 1234)
        self.assertEqual(destination.read_bytes(), self.data)

    def test_ignored_range_restarts_without_appending(self):
        destination = self.root / self.spec.target
        destination.parent.mkdir()
        destination.with_name(destination.name + ".part").write_bytes(self.data[:1234])
        fetch_file(self.spec, self.root, opener=lambda *args, **kwargs: Response(self.data))
        self.assertEqual(destination.read_bytes(), self.data)

    def test_hash_mismatch_preserves_final_and_partial(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            fetch_file(self.spec, self.root, opener=lambda *args, **kwargs: Response(b"X" * len(self.data)))
        destination = self.root / self.spec.target
        self.assertFalse(destination.exists())
        self.assertTrue(destination.with_name(destination.name + ".part").exists())
        destination.write_bytes(b"do not replace")
        with self.assertRaisesRegex(ValueError, "will not be overwritten"):
            fetch_file(self.spec, self.root, opener=lambda *args, **kwargs: self.fail("Unexpected network request"))
        self.assertEqual(destination.read_bytes(), b"do not replace")

    def test_invalid_range_does_not_modify_prefix(self):
        destination = self.root / self.spec.target
        destination.parent.mkdir()
        partial = destination.with_name(destination.name + ".part")
        partial.write_bytes(self.data[:100])
        with self.assertRaisesRegex(ValueError, "invalid checkpoint byte range"):
            fetch_file(self.spec, self.root, opener=lambda *args, **kwargs: Response(self.data[100:], 206, f"bytes 99-{len(self.data)-1}/{len(self.data)}"))
        self.assertEqual(partial.read_bytes(), self.data[:100])
        self.assertFalse(destination.exists())

    def test_atomic_publication_does_not_overwrite_racing_file(self):
        destination = self.root / self.spec.target
        def open_url(request, timeout):
            destination.write_bytes(b"other process output")
            return Response(self.data)
        with self.assertRaisesRegex(ValueError, "will not be overwritten"):
            fetch_file(self.spec, self.root, opener=open_url)
        self.assertEqual(destination.read_bytes(), b"other process output")

    def test_complete_partial_needs_no_network(self):
        destination = self.root / self.spec.target
        destination.parent.mkdir()
        destination.with_name(destination.name + ".part").write_bytes(self.data)
        fetch_file(self.spec, self.root, opener=lambda *args, **kwargs: self.fail("Unexpected network request"))
        self.assertEqual(destination.read_bytes(), self.data)

    def test_interrupted_transfer_can_resume(self):
        class Interrupted(Response):
            def read(self, count=-1):
                if self.tell():
                    raise OSError("connection interrupted")
                return super().read(1000)
        with self.assertRaisesRegex(OSError, "interrupted"):
            fetch_file(self.spec, self.root, opener=lambda *args, **kwargs: Interrupted(self.data))
        destination = self.root / self.spec.target
        partial = destination.with_name(destination.name + ".part")
        self.assertEqual(partial.read_bytes(), self.data[:1000])
        self.assertFalse(destination.with_name(destination.name + ".lock").exists())
        fetch_file(self.spec, self.root, opener=lambda *args, **kwargs: Response(self.data[1000:], 206, f"bytes 1000-{len(self.data)-1}/{len(self.data)}"))
        self.assertEqual(destination.read_bytes(), self.data)

    def test_path_traversal_and_active_lock_are_rejected(self):
        for target in ("../outside", "..\\outside", "C:\\outside", "/outside"):
            invalid = CheckpointFile("example/model", "a" * 40, "x", target, 0, "0" * 64)
            with self.assertRaises(ValueError):
                fetch_file(invalid, self.root)
        destination = self.root / self.spec.target
        destination.parent.mkdir()
        destination.with_name(destination.name + ".lock").write_text("active")
        with self.assertRaisesRegex(ValueError, "locked"):
            fetch_file(self.spec, self.root)


if __name__ == "__main__":
    unittest.main()
