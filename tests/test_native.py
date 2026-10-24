import ctypes as C
import json
import os
import shutil
import struct

import numpy as np
import pytest
from omni_tokenizer.native import Library, Model, NativeError

from tools.convert import FIELDS, HEADER
from tools.make_fixture import generate, write_model


@pytest.fixture(scope="module")
def library():
    try:
        return Library()
    except NativeError:
        if os.environ.get("OMNI_LIBRARY") or os.environ.get("OMNI_REQUIRE_NATIVE"):
            raise
        pytest.skip("Build the native library or set OMNI_LIBRARY to run native tests")


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    directory = tmp_path_factory.mktemp("native-fixtures")
    generate(directory)
    return directory


def test_complete_audio_graph_and_valid_key_mask(library, models):
    with np.load(models / "audio" / "expected.npz") as expected:
        with Model(library, models / "audio") as model, model.context() as context:
            assert model.info.layers == 2 and model.info.width == 4
            codes = context.encode_audio(expected["mel"], int(expected["valid_frames"]))
            np.testing.assert_array_equal(codes, expected["codes"])
            np.testing.assert_allclose(
                context.features(len(codes)), expected["features"], rtol=1e-5, atol=1e-6
            )
            full = context.encode_audio(expected["mel"], expected["mel"].shape[1])
            np.testing.assert_array_equal(full, expected["full_codes"])
            np.testing.assert_allclose(
                context.features(len(full)),
                expected["full_features"],
                rtol=1e-5,
                atol=1e-6,
            )
            assert context.peak_workspace_bytes > 0


def test_complete_image_graph_and_bfloat16_distance(library, models):
    with np.load(models / "image" / "expected.npz") as expected:
        with Model(library, models / "image") as model, model.context() as context:
            codes = context.encode_image(expected["pixels"])
            np.testing.assert_array_equal(codes, expected["codes"])
            np.testing.assert_allclose(
                context.features(len(codes)), expected["features"], rtol=0, atol=1 / 128
            )
            for _ in range(3):
                np.testing.assert_array_equal(
                    context.encode_image(expected["pixels"]), codes
                )


@pytest.mark.parametrize("patch_size", [8, 16])
def test_image_output_allocation_uses_model_patch_size(
    library, models, tmp_path, monkeypatch, patch_size
):
    source = models / "image"
    manifest = json.loads((source / "manifest.json").read_text())
    payload = (source / "model.otm").read_bytes()
    weights = {}
    for tensor in manifest["tensors"]:
        bits = np.frombuffer(
            payload, dtype="<u2", count=tensor["nbytes"] // 2, offset=tensor["offset"]
        )
        weights[tensor["name"]] = (
            (bits.astype(np.uint32) << 16).view(np.float32).reshape(tensor["shape"])
        )
    weights["patch_embed.proj.weight"] = weights["patch_embed.proj.weight"][
        :, :, :patch_size, :patch_size
    ]
    config = dict(manifest["config"], patch_size=patch_size)
    write_model(tmp_path, "image", tuple(config[field] for field in FIELDS), weights)
    with Model(library, tmp_path) as model, model.context() as context:
        pixels = np.zeros((3, patch_size * 2, patch_size * 3), np.float32)
        allocations = []
        original = np.empty

        def allocate(shape, dtype=float, *args, **kwargs):
            if np.dtype(dtype) == np.dtype(np.uint16):
                allocations.append(shape)
            return original(shape, dtype, *args, **kwargs)

        monkeypatch.setattr(np, "empty", allocate)
        codes = context.encode_image(pixels)
        assert codes.size == 6
        assert allocations == [6]


def test_image_capacity_query_does_not_read_input_or_execute(library, models):
    with Model(library, models / "image") as model, model.context() as context:
        pixel, count = C.c_float(), C.c_size_t()
        status = library.lib.omni_encode_image(
            context.handle, C.byref(pixel), 32, 48, None, 0, C.byref(count)
        )
        assert status == 1 and count.value == 6
        assert context.peak_workspace_bytes == 0
        with pytest.raises(NativeError, match="No completed feature"):
            context.features(6)
        context.cancel()
        with pytest.raises(NativeError) as failure:
            context.encode_image(np.zeros((3, 32, 48), np.float32))
        assert failure.value.status == 6
        library.lib.omni_context_reset_cancel(context.handle)
        assert context.encode_image(np.zeros((3, 32, 48), np.float32)).size == 6


def test_unicode_model_path(library, models, tmp_path):
    destination = tmp_path / "model-\u6a21\u578b with spaces.otm"
    shutil.copyfile(models / "audio" / "model.otm", destination)
    with Model(library, destination) as model:
        assert model.info.kind == 1


@pytest.mark.parametrize("size", [(32, 1536), (1536, 32)])
def test_native_image_dimensions_above_default_limit(library, models, size):
    with Model(library, models / "image") as model, model.context() as context:
        codes = context.encode_image(np.zeros((3, *size), np.float32))
        assert len(codes) == size[0] * size[1] // 256
        assert np.isfinite(context.features(len(codes))).all()


def test_native_image_index_bounds_are_checked_before_input_access(library, models):
    with Model(library, models / "image") as model, model.context() as context:
        pixel, code, count = C.c_float(), C.c_uint16(), C.c_size_t()
        for height, width in ((0xFFFFFFF0, 16), (0xFFFFFFF0, 0xFFFFFFF0)):
            status = library.lib.omni_encode_image(
                context.handle,
                C.byref(pixel),
                height,
                width,
                C.byref(code),
                1,
                C.byref(count),
            )
            assert status == 1
            assert b"indexing capacity" in library.lib.omni_last_error()
        status = library.lib.omni_encode_image(
            context.handle, C.byref(pixel), 16, 2048, C.byref(code), 0, C.byref(count)
        )
        assert status == 1 and count.value == 128
        assert b"capacity is too small" in library.lib.omni_last_error()


def test_contexts_retain_weights_after_model_handle_is_freed(library, models):
    with np.load(models / "audio" / "expected.npz") as expected:
        model = Model(library, models / "audio")
        first, second = model.context(), model.context()
        model.close()
        try:
            for context in (first, second, first):
                np.testing.assert_array_equal(
                    context.encode_audio(expected["mel"], 9), expected["codes"]
                )
            with pytest.raises(RuntimeError, match="closed"):
                model.context()
        finally:
            first.close()
            second.close()


def test_cancel_is_sticky_until_reset(library, models):
    with np.load(models / "audio" / "expected.npz") as expected:
        with Model(library, models / "audio") as model, model.context() as context:
            context.cancel()
            for _ in range(2):
                with pytest.raises(NativeError) as failure:
                    context.encode_audio(expected["mel"], 9)
                assert failure.value.status == 6
            library.lib.omni_context_reset_cancel(context.handle)
            np.testing.assert_array_equal(
                context.encode_audio(expected["mel"], 9), expected["codes"]
            )


def test_invalid_audio_frame_count_and_nonfinite_values(library, models):
    with Model(library, models / "audio") as model, model.context() as context:
        with pytest.raises(NativeError) as failure:
            context.encode_audio(np.zeros((128, 9), np.float32), 9)
        assert failure.value.status == 1
        invalid = np.zeros((128, 8), np.float32)
        invalid[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            context.encode_audio(invalid, 8)


def test_output_capacity_is_reported(library, models):
    with np.load(models / "audio" / "expected.npz") as expected:
        with Model(library, models / "audio") as model, model.context() as context:
            mel = expected["mel"]
            output = np.empty(1, np.uint16)
            count = C.c_size_t()
            status = library.lib.omni_encode_audio(
                context.handle,
                mel.ctypes.data_as(C.POINTER(C.c_float)),
                16,
                9,
                output.ctypes.data_as(C.POINTER(C.c_uint16)),
                1,
                C.byref(count),
            )
            assert status == 1
            assert count.value == 2


@pytest.mark.parametrize(
    "mutation", ["truncated", "config", "dtype", "overlap", "name", "shape"]
)
def test_invalid_models_fail_before_execution(library, models, tmp_path, mutation):
    data = bytearray((models / "audio" / "model.otm").read_bytes())
    if mutation == "truncated":
        data = data[:-1]
    elif mutation == "config":
        struct.pack_into("<I", data, 20, 0)
    elif mutation == "dtype":
        struct.pack_into("<I", data, HEADER.size + 4, 2)
    elif mutation == "overlap":
        struct.pack_into("<Q", data, HEADER.size + 16, 0)
    elif mutation == "name":
        rank = struct.unpack_from("<I", data, HEADER.size + 8)[0]
        data[HEADER.size + 32 + 8 * rank] = ord("X")
    elif mutation == "shape":
        struct.pack_into("<Q", data, HEADER.size + 32, 2)
    path = tmp_path / (mutation + ".otm")
    path.write_bytes(data)
    with pytest.raises(NativeError) as failure:
        Model(library, path)
    assert failure.value.status == 3


def test_missing_model_has_io_error(library, tmp_path):
    with pytest.raises(NativeError) as failure:
        Model(library, tmp_path / "missing.otm")
    assert failure.value.status == 2


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_image_input_rejected_before_inference(
    library, models, invalid_value
):
    with (
        np.load(models / "image" / "expected.npz") as expected,
        Model(library, models / "image") as model,
        model.context() as context,
    ):
        pixels = expected["pixels"].copy()
        pixels.flat[-1] = invalid_value
        codes = np.empty(expected["codes"].size, dtype=np.uint16)
        count = C.c_size_t()
        status = library.lib.omni_encode_image(
            context.handle,
            pixels.ctypes.data_as(C.POINTER(C.c_float)),
            pixels.shape[1],
            pixels.shape[2],
            codes.ctypes.data_as(C.POINTER(C.c_uint16)),
            codes.size,
            C.byref(count),
        )
        assert status == 1
        assert b"non-finite" in library.lib.omni_last_error()
        np.testing.assert_array_equal(
            context.encode_image(expected["pixels"]), expected["codes"]
        )
