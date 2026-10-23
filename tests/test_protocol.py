import struct

import pytest

from omni_tokenizer.protocol import FrameDecoder, decode_binary, encode_binary, encode_frame, token_text


def test_exact_token_text():
    assert token_text([17, 230], "audio") == "<|audio_17|><|audio_230|>"
    text = token_text([7, 8, 9, 10, 11, 12], "image", [2, 3])
    assert text == "<|image_7|><|image_8|><|image_9|>\n<|image_10|><|image_11|><|image_12|>"
    assert not text.endswith("\n") and "\r" not in text and "im_end" not in text
    for values in ([16384], [-1], [1.5], [True]):
        with pytest.raises(ValueError):
            token_text(values, "audio")


def test_binary_roundtrip_and_payload():
    items = [{"item_id": "image", "type": "image", "status": "ok", "shape": [1, 3], "codes": [0, 255, 16383]}, {"item_id": "bad", "status": "error", "error": {"code": "invalid_media"}}, {"item_id": "audio", "type": "audio", "status": "ok", "codes": [257]}]
    wire = encode_binary(items)
    assert wire[:8] == b"OVQ1\x01\0\0\0"
    assert wire[-8:] == b"\0\0\xff\0\xff\x3f\x01\x01"
    parsed = decode_binary(wire)
    assert parsed["items"][0]["codes"] == [0, 255, 16383]
    assert parsed["items"][1]["offset"] == 6
    assert parsed["items"][2]["offset"] == 6
    assert parsed["items"][2]["codes"] == [257]
    for broken in (wire[:15], wire[:-1], wire+b"x", b"BAD!"+wire[4:]):
        with pytest.raises(ValueError):
            decode_binary(broken)
    with pytest.raises(ValueError):
        decode_binary(wire, max_codes=2)


def test_stream_one_byte_chunks_large_offsets_and_interruption():
    frames = [{"frame": "start", "item_id": "x", "type": "audio"}, {"frame": "data", "item_id": "x", "segment_index": 9000, "start_sample": 4_320_000_000, "valid_samples": 480000, "code_offset": 3375000, "codes": [42, 16383]}, {"frame": "end", "item_id": "x", "code_count": 2}, {"frame": "done", "item_count": 1}]
    wire = b"".join(map(encode_frame, frames))
    decoder = FrameDecoder()
    output = []
    for byte in wire:
        output.extend(decoder.feed(bytes([byte])))
    decoder.finish()
    assert output[1]["start_sample"] == 4_320_000_000
    assert output[1]["codes"] == [42, 16383]
    assert output[2]["code_count"] == 2
    with pytest.raises(ValueError):
        decoder.feed(b"x")
    decoder = FrameDecoder()
    decoder.feed(wire[:-1])
    with pytest.raises(ValueError):
        decoder.finish()
    with pytest.raises(ValueError):
        FrameDecoder().feed(struct.pack("<4sHHII", b"OVS1", 1, 2, 100_000_000, 0))
    with pytest.raises(ValueError):
        FrameDecoder().feed(encode_frame(frames[0]) + encode_frame(frames[-1]))
