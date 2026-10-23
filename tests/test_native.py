import ctypes as C
import os
from pathlib import Path
import shutil
import struct

import numpy as np
import pytest

from omni_tokenizer.native import Library, Model, NativeError
from tools.convert import HEADER
from tools.make_fixture import generate


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
            np.testing.assert_allclose(context.features(len(codes)), expected["features"], rtol=1e-5, atol=1e-6)
            full = context.encode_audio(expected["mel"], expected["mel"].shape[1])
            np.testing.assert_array_equal(full, expected["full_codes"])
            np.testing.assert_allclose(context.features(len(full)), expected["full_features"], rtol=1e-5, atol=1e-6)
            assert context.peak_workspace_bytes > 0


def test_complete_image_graph_and_bfloat16_distance(library, models):
    with np.load(models / "image" / "expected.npz") as expected:
        with Model(library, models / "image") as model, model.context() as context:
            codes = context.encode_image(expected["pixels"])
            np.testing.assert_array_equal(codes, expected["codes"])
            np.testing.assert_allclose(context.features(len(codes)), expected["features"], rtol=0, atol=1 / 128)
            for _ in range(3):
                np.testing.assert_array_equal(context.encode_image(expected["pixels"]), codes)


def test_unicode_model_path(library, models, tmp_path):
    destination = tmp_path / "model-\u6a21\u578b with spaces.otm"
    shutil.copyfile(models / "audio" / "model.otm", destination)
    with Model(library, destination) as model:
        assert model.info.kind == 1


def test_contexts_retain_weights_after_model_handle_is_freed(library, models):
    with np.load(models / "audio" / "expected.npz") as expected:
        model = Model(library, models / "audio")
        first, second = model.context(), model.context()
        model.close()
        try:
            for context in (first, second, first):
                np.testing.assert_array_equal(context.encode_audio(expected["mel"], 9), expected["codes"])
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
            np.testing.assert_array_equal(context.encode_audio(expected["mel"], 9), expected["codes"])


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
                context.handle, mel.ctypes.data_as(C.POINTER(C.c_float)), 16, 9,
                output.ctypes.data_as(C.POINTER(C.c_uint16)), 1, C.byref(count),
            )
            assert status == 1
            assert count.value == 2


@pytest.mark.parametrize("mutation", ["truncated", "config", "dtype", "overlap", "name", "shape"])
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
