import base64
import io
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


def test_explicit_image_size_reaches_native_encoder_without_auto_resize(
    tokenizer, runtime_inputs
):
    source = runtime_inputs / "image.png"
    result = tokenizer.encode_image(source.read_bytes(), resize=False)
    assert result["processed_size"] == [48, 32]
    assert result["shape"] == [3, 2]
    assert len(result["codes"]) == 6
    assert result["preprocessing"] == "image-rgb-native-size-v1"
    standard = tokenizer.encode_image(source)
    assert standard["shape"] == [36, 24]
    assert standard["preprocessing"] == "image-pillow-v1"


def test_image_instance_defaults_overrides_and_budget(runtime_library, runtime_inputs):
    with Tokenizer(
        image_model=runtime_inputs / "models/image",
        library=runtime_library.path,
        image_min_short_edge=16,
        image_max_long_edge=1536,
    ) as engine:
        image = Image.new("RGB", (2048, 32), (17, 113, 231))
        first = engine.encode_image(image)
        assert first["processed_size"] == [32, 1536]
        override = engine.encode_image(image, max_long_edge=2048, resize=False)
        assert override["shape"] == [2, 128]
        assert override["image_options"]["max_long_edge"] == 2048
        assert engine.encode_image(image)["codes"] == first["codes"]
        assert engine.image_options.max_long_edge == 1536
        batch = engine.encode_batch(
            [
                {
                    "item_id": "custom",
                    "type": "image",
                    "source": image,
                    "image_options": {"resize": False, "max_long_edge": 2048},
                },
                {"item_id": "default", "type": "image", "source": image},
            ]
        )
        assert [item["codes"] for item in batch] == [
            override["codes"],
            first["codes"],
        ]
        engine.max_image_pixels = 256
        with pytest.raises(ValueError, match="Processed image"):
            engine.encode_image(Image.new("RGB", (16, 16)), min_short_edge=512)


def test_http_image_options_reach_native_in_all_formats(tokenizer):
    from omni_tokenizer.protocol import FrameDecoder
    from omni_tokenizer.server import create_app
    from starlette.testclient import TestClient

    image = io.BytesIO()
    Image.new("RGB", (1536, 32), (17, 113, 231)).save(image, format="PNG")
    body = image.getvalue()
    data = base64.b64encode(body).decode("ascii")
    options = {"resize": False, "max_long_edge": 2048}
    with TestClient(create_app(tokenizer)) as client:
        item = {
            "item_id": "image",
            "type": "image",
            "data": data,
            "image_options": options,
        }
        request = {"version": 1, "items": [item]}
        response = client.post("/v1/tokenize", json=request)
        result = response.json()["items"][0]
        assert result["processed_size"] == [32, 1536]
        assert len(result["codes"]) == 192
        multipart_item = {key: value for key, value in item.items() if key != "data"}
        response = client.post(
            "/v1/tokenize",
            data={
                "manifest": json.dumps(
                    {"version": 1, "items": [{**multipart_item, "part": "upload"}]}
                )
            },
            files={"upload": ("image.png", body, "image/png")},
        )
        assert response.json()["items"][0] == result
        decoder = FrameDecoder()
        frames = decoder.feed(
            client.post("/v1/tokenize", json={**request, "stream": True}).content
        )
        decoder.finish()
        assert (
            next(frame for frame in frames if frame["frame"] == "data")["codes"]
            == result["codes"]
        )
        chat = {
            "image_options": options,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + data},
                        }
                    ],
                }
            ],
        }
        response = client.post("/v1/chat/completions", json=chat)
        assert response.status_code == 200
        assert (
            response.json()["choices"][0]["message"]["content"].count("<|image_") == 192
        )
        stream = client.post("/v1/chat/completions", json={**chat, "stream": True})
        assert stream.text.count("<|image_") == 192
        assert stream.text.endswith("data: [DONE]\n\n")
        headers = {"Content-Type": "image/png"}
        assert (
            client.post("/ui/encode", content=body, headers=headers).status_code == 422
        )
        response = client.post(
            "/ui/encode?max_long_edge=2048", content=body, headers=headers
        )
        assert response.status_code == 200
        assert response.json()["shape"] == [2, 96]
        assert response.json()["image_options"]["resize"] is False


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
