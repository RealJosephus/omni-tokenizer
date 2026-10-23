"""Version 1 wire formats. All binary integers use little endian."""

from __future__ import annotations

import json
import struct
from collections.abc import Iterable

import numpy as np

VERSION = 1
BINARY_MEDIA_TYPE = "application/vnd.omni-tokenizer+binary"
STREAM_MEDIA_TYPE = "application/vnd.omni-tokenizer.stream"
HEADER = struct.Struct("<4sHHII")
FRAME_TYPES = {"start": 1, "data": 2, "end": 3, "error": 4, "done": 5}
MAX_METADATA = 8 * 1024 * 1024


def _json(value):
    result = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(result) > MAX_METADATA:
        raise ValueError("Protocol metadata exceeds 8 MiB")
    return result


def _codes(value):
    arr = np.asarray(value)
    if arr.size == 0:
        return np.empty(0, dtype="<u2")
    if (
        arr.ndim != 1
        or arr.dtype.kind not in "iu"
        or np.any(arr < 0)
        or np.any(arr >= 16384)
    ):
        raise ValueError("Codes must be a flat array of integers in [0, 16383]")
    return arr.astype("<u2", copy=False)


def token_text(codes: Iterable[int], kind: str, shape=None) -> str:
    arr = _codes(list(codes) if not isinstance(codes, np.ndarray) else codes)
    if kind == "audio":
        return "".join(f"<|audio_{int(i)}|>" for i in arr)
    if kind != "image" or not isinstance(shape, (list, tuple)) or len(shape) != 2:
        raise ValueError("Images require shape [height, width]")
    h, w = shape
    if (
        type(h) is not int
        or type(w) is not int
        or h <= 0
        or w <= 0
        or h * w != len(arr)
    ):
        raise ValueError("Image shape does not match code count")
    return "\n".join(
        "".join(f"<|image_{int(i)}|>" for i in arr[y * w : (y + 1) * w])
        for y in range(h)
    )


def encode_binary(items: list[dict]) -> bytes:
    metadata = []
    payload = []
    offset = 0
    for item in items:
        entry = dict(item)
        codes = _codes(entry.pop("codes", []))
        entry.update(offset=offset, count=len(codes))
        metadata.append(entry)
        raw = codes.tobytes()
        payload.append(raw)
        offset += len(raw)
    body = _json({"version": VERSION, "items": metadata})
    return (
        HEADER.pack(b"OVQ1", VERSION, 0, len(body), 0)
        + body
        + bytes((-len(body)) % 8)
        + b"".join(payload)
    )


def decode_binary(data: bytes, *, max_codes=10_000_000) -> dict:
    if len(data) < HEADER.size:
        raise ValueError("Truncated binary header")
    magic, version, flags, length, reserved = HEADER.unpack_from(data)
    if (magic, version, flags, reserved) != (
        b"OVQ1",
        VERSION,
        0,
        0,
    ) or length > MAX_METADATA:
        raise ValueError("Unsupported binary header")
    start = HEADER.size + length
    payload_start = start + (-length) % 8
    if payload_start > len(data) or any(data[start:payload_start]):
        raise ValueError("Truncated or nonzero alignment padding")
    metadata = json.loads(data[HEADER.size : start])
    if (
        not isinstance(metadata, dict)
        or metadata.get("version") != VERSION
        or not isinstance(metadata.get("items"), list)
    ):
        raise ValueError("Invalid binary metadata")
    offset = 0
    for item in metadata["items"]:
        if not isinstance(item, dict):
            raise ValueError("Invalid binary item")
        count = item.get("count")
        if (
            type(count) is not int
            or count < 0
            or count > max_codes
            or type(item.get("offset")) is not int
            or item["offset"] != offset
        ):
            raise ValueError("Invalid code range")
        end = payload_start + offset + 2 * count
        if end > len(data):
            raise ValueError("Truncated code payload")
        item["codes"] = _codes(
            np.frombuffer(data, dtype="<u2", count=count, offset=payload_start + offset)
        ).tolist()
        offset += 2 * count
        max_codes -= count
    if payload_start + offset != len(data):
        raise ValueError("Trailing binary payload")
    return metadata


def encode_frame(frame: dict) -> bytes:
    entry = dict(frame)
    kind = entry.get("frame")
    if kind not in FRAME_TYPES:
        raise ValueError("Unknown stream frame type")
    codes = _codes(entry.pop("codes", []))
    entry["version"] = VERSION
    if kind != "data" and len(codes):
        raise ValueError("Only data frames carry codes")
    entry["code_count"] = len(codes) if kind == "data" else entry.get("code_count", 0)
    body = _json(entry)
    return (
        HEADER.pack(b"OVS1", VERSION, FRAME_TYPES[kind], len(body), len(codes))
        + body
        + bytes((-len(body)) % 8)
        + codes.tobytes()
    )


class FrameDecoder:
    """Incremental decoder; finish() rejects incomplete streams and frames."""

    def __init__(self, max_frame_codes=1_000_000):
        self.buffer = bytearray()
        self.max_frame_codes = max_frame_codes
        self.done = False
        self.active_items = {}
        self.finished_items = set()

    def feed(self, chunk: bytes) -> list[dict]:
        if self.done and chunk:
            raise ValueError("Data after done frame")
        self.buffer.extend(chunk)
        frames = []
        while len(self.buffer) >= HEADER.size:
            magic, version, kind, length, count = HEADER.unpack_from(self.buffer)
            if (
                magic != b"OVS1"
                or version != VERSION
                or kind not in FRAME_TYPES.values()
                or length > MAX_METADATA
                or count > self.max_frame_codes
            ):
                raise ValueError("Invalid stream header")
            padding = (-length) % 8
            total = HEADER.size + length + padding + count * 2
            if len(self.buffer) < total:
                break
            start = HEADER.size + length
            if any(self.buffer[start : start + padding]):
                raise ValueError("Nonzero stream alignment padding")
            frame = json.loads(self.buffer[HEADER.size : start])
            if (
                not isinstance(frame, dict)
                or frame.get("version") != VERSION
                or FRAME_TYPES.get(frame.get("frame")) != kind
            ):
                raise ValueError("Stream frame type mismatch")
            if kind == FRAME_TYPES["data"]:
                if frame.get("code_count") != count:
                    raise ValueError("Stream code count mismatch")
                frame["codes"] = _codes(
                    np.frombuffer(
                        bytes(self.buffer[start + padding : total]), dtype="<u2"
                    )
                ).tolist()
            elif count:
                raise ValueError("Unexpected code payload")
            item_id = frame.get("item_id")
            if kind != FRAME_TYPES["done"] and not isinstance(item_id, str):
                raise ValueError("Item frames require a string item_id")
            if kind == FRAME_TYPES["start"]:
                if item_id in self.active_items or item_id in self.finished_items:
                    raise ValueError("Duplicate stream item")
                self.active_items[item_id] = 0
            elif kind == FRAME_TYPES["data"]:
                if item_id not in self.active_items:
                    raise ValueError("Data frame before item start")
                self.active_items[item_id] += count
            elif kind in (FRAME_TYPES["end"], FRAME_TYPES["error"]):
                if item_id in self.finished_items:
                    raise ValueError("Duplicate item termination")
                seen_codes = self.active_items.pop(item_id, None)
                if seen_codes is None:
                    raise ValueError("Item termination before start")
                terminal_count = frame.get(
                    "code_count" if kind == FRAME_TYPES["end"] else "committed_codes"
                )
                if type(terminal_count) is not int or seen_codes != terminal_count:
                    raise ValueError("Item completion count mismatch")
                self.finished_items.add(item_id)
            else:
                if self.active_items:
                    raise ValueError("Done frame before all items terminated")
                if type(frame.get("item_count")) is not int or frame[
                    "item_count"
                ] != len(self.finished_items):
                    raise ValueError("Stream item count mismatch")
            del self.buffer[:total]
            frames.append(frame)
            if kind == FRAME_TYPES["done"]:
                self.done = True
                if self.buffer:
                    raise ValueError("Data after done frame")
                break
        return frames

    def finish(self):
        if self.buffer or not self.done or self.active_items:
            raise ValueError("Incomplete token stream")
