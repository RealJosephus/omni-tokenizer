"""Small ctypes binding. No tensor framework is imported at runtime."""

from __future__ import annotations

import ctypes as C
import os
import sys
import threading
from pathlib import Path

import numpy as np


class NativeError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class ModelInfo(C.Structure):
    _fields_ = [
        (name, C.c_uint32)
        for name in (
            "struct_size",
            "kind",
            "layers",
            "width",
            "heads",
            "code_dim",
            "vocabulary_size",
            "backend",
        )
    ] + [("weight_bytes", C.c_uint64)]


class Library:
    def __init__(self, path=None):
        filename = (
            "omni.dll"
            if sys.platform == "win32"
            else "libomni.dylib"
            if sys.platform == "darwin"
            else "libomni.so"
        )
        root = Path(__file__).resolve().parents[2]
        explicit = path or os.environ.get("OMNI_LIBRARY")
        candidates = (
            [Path(explicit)]
            if explicit
            else [Path(__file__).parent / "_native" / filename]
            + [
                root / d / sub / filename
                for d in ("build", "build-cuda", "build-cpu")
                for sub in ("", "Release", "lib", "bin")
            ]
        )
        found = next((p for p in candidates if p.is_file()), None)
        if found is None:
            raise NativeError(
                4,
                "Native library not found. Build with CMake and set OMNI_LIBRARY to the library path.",
            )
        self._dll_dirs = []
        if os.name == "nt":
            for directory in (
                found.resolve().parent,
                Path(os.environ.get("CUDA_PATH", "")) / "bin",
            ):
                if directory.is_dir():
                    self._dll_dirs.append(
                        os.add_dll_directory(str(directory.resolve()))
                    )
        self.path = found.resolve()
        try:
            self.lib = C.CDLL(str(self.path))
        except OSError as exc:
            raise NativeError(
                4,
                "Cannot load the native library or its dependencies. For CUDA builds, configure CUDA_PATH or the shared-library search path.",
            ) from exc
        pointer = C.c_void_p
        signature = {
            "omni_last_error": (C.c_char_p, []),
            "omni_version": (C.c_char_p, []),
            "omni_has_cuda": (C.c_int, []),
            "omni_model_load": (
                C.c_int,
                [C.c_char_p, C.c_int, C.c_int, C.POINTER(pointer)],
            ),
            "omni_model_free": (None, [pointer]),
            "omni_model_get_info": (C.c_int, [pointer, C.POINTER(ModelInfo)]),
            "omni_context_create": (C.c_int, [pointer, C.POINTER(pointer)]),
            "omni_context_free": (None, [pointer]),
            "omni_context_cancel": (None, [pointer]),
            "omni_context_reset_cancel": (None, [pointer]),
            "omni_encode_audio": (
                C.c_int,
                [
                    pointer,
                    C.POINTER(C.c_float),
                    C.c_uint32,
                    C.c_uint32,
                    C.POINTER(C.c_uint16),
                    C.c_size_t,
                    C.POINTER(C.c_size_t),
                ],
            ),
            "omni_encode_image": (
                C.c_int,
                [
                    pointer,
                    C.POINTER(C.c_float),
                    C.c_uint32,
                    C.c_uint32,
                    C.POINTER(C.c_uint16),
                    C.c_size_t,
                    C.POINTER(C.c_size_t),
                ],
            ),
            "omni_context_copy_features": (
                C.c_int,
                [pointer, C.POINTER(C.c_float), C.c_size_t, C.POINTER(C.c_size_t)],
            ),
            "omni_context_peak_workspace": (C.c_uint64, [pointer]),
        }
        for name, (result, arguments) in signature.items():
            fn = getattr(self.lib, name)
            fn.restype, fn.argtypes = result, arguments

    def check(self, status):
        if status:
            message = self.lib.omni_last_error()
            raise NativeError(
                status,
                message.decode("utf-8", errors="replace")
                if message
                else f"Native error {status}",
            )


class Model:
    def __init__(self, library: Library, path, device="cpu"):
        self.library = library
        self.handle = C.c_void_p()
        if device == "cpu":
            backend, index = 0, 0
        elif device == "cuda" or device.startswith("cuda:"):
            backend, index = 1, int(device.split(":")[1]) if ":" in device else 0
            if index < 0:
                raise ValueError("CUDA device index cannot be negative")
        else:
            raise ValueError("Device must be cpu, cuda, or cuda:N")
        model_path = Path(path)
        if model_path.is_dir():
            model_path /= "model.otm"
        self.path = model_path.resolve()
        library.check(
            library.lib.omni_model_load(
                str(self.path).encode("utf-8"), backend, index, C.byref(self.handle)
            )
        )
        self.info = ModelInfo(struct_size=C.sizeof(ModelInfo))
        try:
            library.check(
                library.lib.omni_model_get_info(self.handle, C.byref(self.info))
            )
        except BaseException:
            self.close()
            raise

    def context(self):
        if not self.handle:
            raise RuntimeError("Model is closed")
        return Context(self)

    def close(self):
        if self.handle:
            self.library.lib.omni_model_free(self.handle)
            self.handle = C.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Context:
    def __init__(self, model):
        self.model = model
        self.library = model.library
        self.handle = C.c_void_p()
        self._guard = threading.Lock()
        self.library.check(
            self.library.lib.omni_context_create(model.handle, C.byref(self.handle))
        )

    def cancel(self):
        with self._guard:
            if self.handle:
                self.library.lib.omni_context_cancel(self.handle)

    def encode_audio(self, mel, valid_frames):
        mel = np.ascontiguousarray(mel, dtype=np.float32)
        if (
            mel.ndim != 2
            or mel.shape[0] != 128
            or not 0 < valid_frames <= mel.shape[1] <= 3000
        ):
            raise ValueError("Expected mel [128, frames <= 3000] and valid_frames > 0")
        return self._encode(
            self.library.lib.omni_encode_audio,
            mel,
            mel.shape[1],
            int(valid_frames),
            (valid_frames + 7) // 8,
        )

    def encode_image(self, pixels):
        pixels = np.ascontiguousarray(pixels, dtype=np.float32)
        if (
            pixels.ndim != 3
            or pixels.shape[0] != 3
            or not all(0 < x <= 0xFFFFFFFF for x in pixels.shape[1:])
        ):
            raise ValueError("Expected RGB CHW pixels with positive uint32 dimensions")
        h, w = pixels.shape[1:]
        return self._encode(self.library.lib.omni_encode_image, pixels, h, w, h * w)

    def _encode(self, fn, data, dim1, dim2, capacity):
        if not self.handle:
            raise RuntimeError("Context is closed")
        if not np.isfinite(data).all():
            raise ValueError("Input contains NaN or infinity")
        output = np.empty(capacity, dtype=np.uint16)
        count = C.c_size_t()
        self.library.check(
            fn(
                self.handle,
                data.ctypes.data_as(C.POINTER(C.c_float)),
                dim1,
                dim2,
                output.ctypes.data_as(C.POINTER(C.c_uint16)),
                capacity,
                C.byref(count),
            )
        )
        return output[: count.value].copy()

    def features(self, count):
        output = np.empty((count, self.model.info.code_dim), dtype=np.float32)
        size = C.c_size_t()
        self.library.check(
            self.library.lib.omni_context_copy_features(
                self.handle,
                output.ctypes.data_as(C.POINTER(C.c_float)),
                output.size,
                C.byref(size),
            )
        )
        return output.ravel()[: size.value].reshape(-1, self.model.info.code_dim)

    @property
    def peak_workspace_bytes(self):
        return int(self.library.lib.omni_context_peak_workspace(self.handle))

    def close(self):
        with self._guard:
            if self.handle:
                self.library.lib.omni_context_free(self.handle)
                self.handle = C.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
