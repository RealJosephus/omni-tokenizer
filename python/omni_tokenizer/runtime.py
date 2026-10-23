"""Shared model ownership and a single bounded compute lane."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager

from .native import Library, Model, NativeError


class TokenizationError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class Cancellation:
    def __init__(self):
        self.event = threading.Event()
        self._lock = threading.Lock()
        self._context = None

    def cancel(self):
        self.event.set()
        with self._lock:
            if self._context:
                self._context.cancel()

    def check(self):
        if self.event.is_set():
            raise TokenizationError("cancelled", "Tokenization cancelled")

    @contextmanager
    def bind(self, context):
        with self._lock:
            self.check()
            self._context = context
        try:
            yield
        finally:
            with self._lock:
                self._context = None


class Tokenizer:
    """Pull-based audio API; only the current segment resides in the compute lane.

    Model handles are shared. One compute call runs at a time; request admission
    limits belong to the server. Use separate instances for separate GPUs.
    """

    def __init__(
        self,
        *,
        audio_model=None,
        image_model=None,
        device="cpu",
        library=None,
        max_codes=2_000_000,
    ):
        self.paths = {"audio": audio_model, "image": image_model}
        self.device = device
        self.library_path = library
        self.library = None
        self.models = {}
        self.contexts = {}
        self._identities = {}
        self._lane = threading.Lock()
        self.max_codes = max_codes
        self.last_workspace_bytes = 0
        self._closed = False

    @contextmanager
    def _compute(self, cancel):
        while not self._lane.acquire(timeout=0.1):
            cancel.check()
        try:
            cancel.check()
            if self._closed:
                raise TokenizationError("closed", "Tokenizer is closed")
            yield
        finally:
            self._lane.release()

    def _model(self, kind):
        if kind not in self.models:
            path = self.paths[kind]
            if not path:
                raise TokenizationError(
                    "model_unavailable", f"No {kind} model configured"
                )
            if self.library is None:
                self.library = Library(self.library_path)
            model = Model(self.library, path, self.device)
            if model.info.kind != {"audio": 1, "image": 2}[kind]:
                model.close()
                raise TokenizationError(
                    "invalid_model", f"Configured {kind} model has the wrong kind"
                )
            manifest = model.path.parent / "manifest.json"
            try:
                data = (
                    json.loads(manifest.read_text(encoding="utf-8"))
                    if manifest.is_file()
                    else {}
                )
                if not isinstance(data, dict):
                    raise ValueError("Model manifest must be a JSON object")
            except BaseException:
                model.close()
                raise
            try:
                context = model.context()
            except BaseException:
                model.close()
                raise
            self.models[kind] = model
            self.contexts[kind] = context
            self._identities[kind] = {
                "model": "vqwhisper" if kind == "audio" else "vqsiglip",
                "model_sha256": data.get("model_sha256", data.get("sha256")),
                "profile": "audio-fp32-v1" if kind == "audio" else "image-bf16-v1",
                "backend": self.device,
                "preprocessing": "audio-canonical-v1"
                if kind == "audio"
                else "image-pillow-v1",
            }
        return self.models[kind]

    def encode_image(self, source, *, cancel=None):
        from .image import prepare_image

        cancel = cancel or Cancellation()
        cancel.check()
        prepared = prepare_image(source)
        with self._compute(cancel):
            self._model("image")
            context = self.contexts["image"]
            context.library.lib.omni_context_reset_cancel(context.handle)
            with cancel.bind(context):
                codes = context.encode_image(prepared.pixels)
                self.last_workspace_bytes = context.peak_workspace_bytes
        return {
            "type": "image",
            "status": "ok",
            "shape": list(prepared.grid),
            "original_size": list(prepared.original_size),
            "processed_size": list(prepared.processed_size),
            "codes": codes.tolist(),
            **self._identities["image"],
        }

    def iter_audio(self, source, *, cancel=None):
        from .audio import iter_feature_segments

        cancel = cancel or Cancellation()
        code_offset = 0
        iterator = iter_feature_segments(source, cancel=cancel.check)
        try:
            for segment, features in iterator:
                cancel.check()
                with self._compute(cancel):
                    self._model("audio")
                    context = self.contexts["audio"]
                    context.library.lib.omni_context_reset_cancel(context.handle)
                    with cancel.bind(context):
                        codes = context.encode_audio(
                            features.mel, features.valid_frames
                        )
                        self.last_workspace_bytes = context.peak_workspace_bytes
                yield {
                    "type": "audio",
                    "status": "ok",
                    "segment_index": segment.index,
                    "start_sample": segment.start_sample,
                    "valid_samples": segment.valid_samples,
                    "padded_samples": features.padded_samples,
                    "sample_rate": 16000,
                    "code_offset": code_offset,
                    "code_count": len(codes),
                    "codes": codes.tolist(),
                    **self._identities["audio"],
                }
                code_offset += len(codes)
        finally:
            iterator.close()

    def encode_audio(self, source, *, cancel=None):
        result = {
            "type": "audio",
            "status": "ok",
            "sample_rate": 16000,
            "valid_samples": 0,
            "codes": [],
            "segments": [],
        }
        for part in self.iter_audio(source, cancel=cancel):
            if len(result["codes"]) + len(part["codes"]) > self.max_codes:
                raise TokenizationError(
                    "output_limit",
                    "Aggregate code limit exceeded; use streaming output",
                )
            result["codes"].extend(part["codes"])
            result["valid_samples"] += part["valid_samples"]
            result["segments"].append(
                {
                    key: part[key]
                    for key in (
                        "segment_index",
                        "start_sample",
                        "valid_samples",
                        "padded_samples",
                        "code_offset",
                        "code_count",
                    )
                }
            )
        result.update(self._identities["audio"])
        return result

    def iter_batch(self, items, *, cancel=None):
        cancel = cancel or Cancellation()
        for index, item in enumerate(items):
            cancel.check()
            item_id = item.get("item_id", str(index))
            kind = item.get("type")
            try:
                if kind not in {"audio", "image"}:
                    raise ValueError("Batch type must be audio or image")
                encode = self.encode_audio if kind == "audio" else self.encode_image
                yield {"item_id": item_id, **encode(item["source"], cancel=cancel)}
            except Exception as exc:
                yield {
                    "item_id": item_id,
                    "type": kind,
                    "status": "error",
                    "error": error_record(exc),
                }

    def encode_batch(self, items, *, cancel=None, max_items=16):
        bounded = []
        for item in items:
            if len(bounded) >= max_items:
                raise TokenizationError(
                    "input_limit", "Aggregate batch item limit exceeded; use iter_batch"
                )
            bounded.append(item)
        results, count = [], 0
        for result in self.iter_batch(bounded, cancel=cancel):
            count += len(result.get("codes", []))
            if count > self.max_codes:
                raise TokenizationError(
                    "output_limit", "Aggregate code limit exceeded; use iter_batch"
                )
            results.append(result)
        return results

    def close(self):
        with self._lane:
            self._closed = True
            for context in self.contexts.values():
                context.close()
            self.contexts.clear()
            for model in self.models.values():
                model.close()
            self.models.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def error_record(exc):
    if isinstance(exc, TokenizationError):
        code = exc.code
    elif isinstance(exc, NativeError):
        code = {
            1: "invalid_input",
            2: "io_error",
            3: "invalid_model",
            4: "unsupported",
            5: "out_of_memory",
            6: "cancelled",
        }.get(exc.status, "inference_error")
    elif isinstance(exc, (ValueError, OSError)):
        code = getattr(exc, "code", "invalid_media")
    else:
        code = "internal_error"
    return {
        "code": code,
        "message": str(exc)
        if code != "internal_error"
        else "Internal tokenization error",
    }
