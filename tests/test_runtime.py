import json
import os
import shutil
import wave

import numpy as np
import pytest
from omni_tokenizer import runtime
from omni_tokenizer.native import Library, NativeError
from omni_tokenizer.runtime import Cancellation, TokenizationError, Tokenizer
from PIL import Image

from tools.make_fixture import generate


@pytest.fixture(scope="module")
def runtime_library():
    try:
        return Library()
    except NativeError:
        if os.environ.get("OMNI_LIBRARY") or os.environ.get("OMNI_REQUIRE_NATIVE"):
            raise
        pytest.skip("Build the native library to run runtime integration tests")


@pytest.fixture(scope="module")
def runtime_inputs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("runtime-fixtures")
    generate(directory / "models")
    for name, length in (
        ("short.wav", 159),
        ("tail.wav", 1281),
        ("three.wav", 2561),
        ("empty.wav", 0),
    ):
        samples = ((np.arange(length, dtype=np.int32) * 137) % 20000 - 10000).astype(
            "<i2"
        )
        with wave.open(str(directory / name), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(samples.tobytes())
    Image.new("RGB", (32, 48), (17, 113, 231)).save(directory / "image.png")
    return directory


@pytest.fixture
def tokenizer(runtime_library, runtime_inputs):
    with Tokenizer(
        audio_model=runtime_inputs / "models" / "audio",
        image_model=runtime_inputs / "models" / "image",
        library=runtime_library.path,
    ) as engine:
        yield engine


def test_audio_iterator_aggregate_and_context_reuse(tokenizer, runtime_inputs):
    source = runtime_inputs / "tail.wav"
    parts = list(tokenizer.iter_audio(source))
    assert len(parts) == 1
    assert parts[0]["segment_index"] == 0
    assert parts[0]["start_sample"] == 0
    assert parts[0]["valid_samples"] == 1281
    assert parts[0]["padded_samples"] == 2560
    assert parts[0]["code_offset"] == 0
    assert parts[0]["code_count"] == 2
    context = tokenizer.contexts["audio"]
    model = tokenizer.models["audio"]
    aggregate = tokenizer.encode_audio(source)
    assert aggregate["codes"] == parts[0]["codes"]
    assert aggregate["valid_samples"] == 1281
    assert aggregate["segments"][0]["code_count"] == 2
    assert tokenizer.contexts["audio"] is context
    assert tokenizer.models["audio"] is model
    assert tokenizer.last_workspace_bytes > 0
    assert len(tokenizer.encode_audio(runtime_inputs / "short.wav")["codes"]) == 1
    assert tokenizer.contexts["audio"] is context


def test_mixed_batch_order_errors_and_context_reuse(tokenizer, runtime_inputs):
    items = [
        {
            "item_id": "audio-one",
            "type": "audio",
            "source": runtime_inputs / "short.wav",
        },
        {"item_id": "invalid-type", "type": "video", "source": "unused"},
        {
            "item_id": "image-one",
            "type": "image",
            "source": runtime_inputs / "image.png",
        },
        {
            "item_id": "empty-audio",
            "type": "audio",
            "source": runtime_inputs / "empty.wav",
        },
        {
            "item_id": "audio-tail",
            "type": "audio",
            "source": runtime_inputs / "tail.wav",
        },
    ]
    aggregated = tokenizer.encode_batch(items)
    contexts = dict(tokenizer.contexts)
    streamed = list(tokenizer.iter_batch(iter(items)))
    assert [item["item_id"] for item in aggregated] == [
        item["item_id"] for item in items
    ]
    assert [item["status"] for item in aggregated] == [
        "ok",
        "error",
        "ok",
        "error",
        "ok",
    ]
    assert aggregated == streamed
    assert len(aggregated[0]["codes"]) == 1
    assert aggregated[2]["shape"] == [36, 24]
    assert len(aggregated[2]["codes"]) == 864
    assert len(aggregated[4]["codes"]) == 2
    assert all(tokenizer.contexts[key] is value for key, value in contexts.items())


def test_batch_iterator_applies_backpressure(tokenizer, runtime_inputs):
    consumed = []

    def inputs():
        for index in range(20):
            consumed.append(index)
            yield {
                "item_id": str(index),
                "type": "audio",
                "source": runtime_inputs / "short.wav",
            }

    iterator = tokenizer.iter_batch(inputs())
    assert next(iterator)["item_id"] == "0"
    assert consumed == [0]
    iterator.close()
    assert consumed == [0]


def test_cancel_during_compute_then_new_request_resets_native_state(
    tokenizer, runtime_inputs, monkeypatch
):
    source = runtime_inputs / "tail.wav"
    baseline = tokenizer.encode_audio(source)["codes"]
    context = tokenizer.contexts["audio"]
    encode = context.encode_audio
    cancellation = Cancellation()

    def cancel_at_compute(mel, valid_frames):
        cancellation.cancel()
        return encode(mel, valid_frames)

    with monkeypatch.context() as patch:
        patch.setattr(context, "encode_audio", cancel_at_compute)
        with pytest.raises(NativeError) as failure:
            tokenizer.encode_audio(source, cancel=cancellation)
        assert failure.value.status == 6
    with pytest.raises(TokenizationError) as failure:
        tokenizer.encode_audio(source, cancel=cancellation)
    assert failure.value.code == "cancelled"
    assert tokenizer.encode_audio(source)["codes"] == baseline
    assert tokenizer.contexts["audio"] is context


def test_output_limit_does_not_poison_reused_context(tokenizer, runtime_inputs):
    tokenizer.max_codes = 1
    with pytest.raises(TokenizationError) as failure:
        tokenizer.encode_audio(runtime_inputs / "tail.wav")
    assert failure.value.code == "output_limit"
    context = tokenizer.contexts["audio"]
    tokenizer.max_codes = 10
    assert len(tokenizer.encode_audio(runtime_inputs / "three.wav")["codes"]) == 3
    assert tokenizer.contexts["audio"] is context


def test_aggregate_batch_limits_items_and_total_codes(tokenizer, runtime_inputs):
    item = {"item_id": "audio", "type": "audio", "source": runtime_inputs / "short.wav"}
    with pytest.raises(TokenizationError) as failure:
        tokenizer.encode_batch([item, item], max_items=1)
    assert failure.value.code == "input_limit"
    assert tokenizer.models == {}
    tokenizer.max_codes = 1
    with pytest.raises(TokenizationError) as failure:
        tokenizer.encode_batch([item, item])
    assert failure.value.code == "output_limit"


def test_closed_tokenizer_rejects_new_work_and_releases_handles(
    tokenizer, runtime_inputs
):
    tokenizer.encode_audio(runtime_inputs / "short.wav")
    context = tokenizer.contexts["audio"]
    model = tokenizer.models["audio"]
    tokenizer.close()
    assert not context.handle and not model.handle
    assert tokenizer.models == {} and tokenizer.contexts == {}
    with pytest.raises(TokenizationError) as failure:
        tokenizer.encode_audio(runtime_inputs / "short.wav")
    assert failure.value.code == "closed"
    with pytest.raises(TokenizationError) as failure:
        tokenizer.encode_image(runtime_inputs / "image.png")
    assert failure.value.code == "closed"
    tokenizer.close()


@pytest.mark.parametrize("manifest_text", ["{", "[]"])
def test_malformed_manifest_closes_loaded_model_before_retry(
    runtime_library, runtime_inputs, tmp_path, monkeypatch, manifest_text
):
    model_directory = tmp_path / "broken-audio-model"
    shutil.copytree(runtime_inputs / "models" / "audio", model_directory)
    (model_directory / "manifest.json").write_text(manifest_text, encoding="utf-8")
    original_model = runtime.Model
    observed = []

    def tracked_model(*args, **kwargs):
        model = original_model(*args, **kwargs)
        observed.append(model)
        return model

    monkeypatch.setattr(runtime, "Model", tracked_model)
    with Tokenizer(audio_model=model_directory, library=runtime_library.path) as engine:
        with pytest.raises(ValueError):
            engine.encode_audio(runtime_inputs / "short.wav")
        assert len(observed) == 1 and not observed[0].handle
        assert engine.models == {} and engine.contexts == {}
        (model_directory / "manifest.json").write_text(
            json.dumps({"model_sha256": "fixture"}), encoding="utf-8"
        )
        assert len(engine.encode_audio(runtime_inputs / "short.wav")["codes"]) == 1
        assert len(observed) == 2 and observed[1].handle
    assert not observed[1].handle
