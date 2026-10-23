import hashlib
import io
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from omni_tokenizer import cli
from omni_tokenizer.protocol import FrameDecoder


class Engine:
    def encode_image(self, source, cancel=None):
        if str(source).endswith("bad.png"):
            raise ValueError("Invalid image")
        return {"type": "image", "status": "ok", "shape": [1, 2], "codes": [7, 8]}


@pytest.mark.parametrize("format", ["stream", "jsonl"])
def test_streaming_encode_returns_nonzero_for_media_failure(
    tmp_path, monkeypatch, format
):
    monkeypatch.setattr(cli, "Tokenizer", lambda **_: nullcontext(Engine()))
    output = tmp_path / "output"
    assert (
        cli.main(["encode", "image", "bad.png", "--format", format, "-o", str(output)])
        == 1
    )
    if format == "stream":
        decoder = FrameDecoder()
        frames = decoder.feed(output.read_bytes())
        decoder.finish()
    else:
        frames = [json.loads(line) for line in output.read_text().splitlines()]
    assert [frame["frame"] for frame in frames] == ["start", "error", "done"]


def test_batch_media_failure_preserves_valid_stream(tmp_path):
    source = tmp_path / "batch.jsonl"
    source.write_text(
        "\n".join(
            json.dumps({"item_id": str(i), "type": "image", "path": path})
            for i, path in enumerate(["good.png", "bad.png", "good.png"])
        )
    )
    output = io.BytesIO()
    assert (
        cli._batch(Engine(), SimpleNamespace(input=source, format="stream"), output)
        == 1
    )
    decoder = FrameDecoder()
    frames = decoder.feed(output.getvalue())
    decoder.finish()
    assert [frame["item_id"] for frame in frames if frame["frame"] == "data"] == [
        "0",
        "2",
    ]
    assert frames[-1]["item_count"] == 3


def test_duplicate_batch_id_aborts_without_false_completion(tmp_path):
    source = tmp_path / "batch.jsonl"
    item = json.dumps({"item_id": "same", "type": "image", "path": "good.png"})
    source.write_text(item + "\n" + item)
    output = io.BytesIO()
    with pytest.raises(ValueError, match="unique"):
        cli._batch(Engine(), SimpleNamespace(input=source, format="stream"), output)
    decoder = FrameDecoder()
    frames = decoder.feed(output.getvalue())
    assert [frame["frame"] for frame in frames] == ["start", "data", "end"]
    with pytest.raises(ValueError, match="Incomplete"):
        decoder.finish()


def test_inspect_verifies_the_explicit_file(tmp_path):
    explicit = tmp_path / "renamed.otm"
    explicit.write_bytes(b"expected")
    (tmp_path / "model.otm").write_bytes(b"wrong sibling")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "model_sha256": hashlib.sha256(explicit.read_bytes()).hexdigest(),
                "tensors": [],
            }
        )
    )
    assert cli._inspect(explicit, True)["verified"]
    with pytest.raises(ValueError, match="SHA-256"):
        cli._inspect(tmp_path, True)
