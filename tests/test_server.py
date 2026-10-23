import asyncio
import base64
import json
import threading
import time

import pytest
from omni_tokenizer.image import ImageOptions
from omni_tokenizer.protocol import BINARY_MEDIA_TYPE, FrameDecoder, decode_binary
from omni_tokenizer.runtime import TokenizationError
from omni_tokenizer.server import Limits, create_app
from starlette.testclient import TestClient


class Engine:
    def encode_image(self, source, cancel=None, resize=True):
        if (source if isinstance(source, bytes) else source.read_bytes()) == b"bad":
            raise ValueError("Invalid image")
        return {
            "type": "image",
            "status": "ok",
            "shape": [2, 2],
            "codes": [7, 8, 9, 10],
            "profile": "fixture",
        }

    def iter_audio(self, source, cancel=None):
        for index in range(2):
            cancel.check()
            yield {
                "type": "audio",
                "status": "ok",
                "segment_index": index,
                "start_sample": index * 480000,
                "valid_samples": 480000,
                "padded_samples": 480000,
                "code_offset": index * 2,
                "code_count": 2,
                "codes": [17 + index, 230],
                "profile": "fixture",
            }

    def encode_audio(self, source, cancel=None):
        return {
            "type": "audio",
            "status": "ok",
            "codes": [
                c
                for segment in self.iter_audio(source, cancel)
                for c in segment["codes"]
            ],
        }


@pytest.fixture
def client():
    with TestClient(create_app(Engine())) as value:
        yield value


def item(kind="image", item_id="a", value=b"media"):
    return {"item_id": item_id, "type": kind, "data": base64.b64encode(value).decode()}


def test_mixed_batch_binary_errors_and_order(client):
    value = {
        "version": 1,
        "items": [item(), item("audio", "b"), item("image", "c", b"bad")],
    }
    response = client.post("/v1/tokenize", json=value)
    assert response.status_code == 200
    entries = response.json()["items"]
    assert [entry["item_id"] for entry in entries] == ["a", "b", "c"]
    assert entries[2]["status"] == "error"
    binary = client.post(
        "/v1/tokenize", json=value, headers={"Accept": BINARY_MEDIA_TYPE}
    )
    decoded = decode_binary(binary.content)
    assert decoded["items"][0]["codes"] == entries[0]["codes"]
    assert decoded["items"][1]["codes"] == entries[1]["codes"]


def test_multipart_stream_and_terminal_error(client):
    manifest = {
        "version": 1,
        "stream": True,
        "items": [
            {"item_id": "audio", "type": "audio", "part": "clip"},
            {"item_id": "missing", "type": "image", "part": "absent"},
        ],
    }
    response = client.post(
        "/v1/tokenize",
        data={"manifest": json.dumps(manifest)},
        files={"clip": ("clip.wav", b"media")},
    )
    assert response.status_code == 200
    decoder = FrameDecoder()
    frames = decoder.feed(response.content)
    decoder.finish()
    assert [frame["frame"] for frame in frames] == [
        "start",
        "data",
        "data",
        "end",
        "start",
        "error",
        "done",
    ]
    assert frames[2]["code_offset"] == 2
    assert frames[5]["committed_codes"] == 0


def test_chat_strict_image_and_audio_sse(client):
    image = {
        "model": "omni-tokenizer",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,bWVkaWE="},
                    }
                ],
            }
        ],
    }
    response = client.post("/v1/chat/completions", json=image)
    assert response.status_code == 200
    assert (
        response.json()["choices"][0]["message"]["content"]
        == "<|image_7|><|image_8|>\n<|image_9|><|image_10|>"
    )
    audio = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "bWVkaWE=", "format": "wav"},
                    }
                ],
            }
        ],
        "stream": True,
    }
    response = client.post("/v1/chat/completions", json=audio)
    chunks = [
        line[6:] for line in response.text.splitlines() if line.startswith("data: ")
    ]
    assert chunks[-1] == "[DONE]"
    text = "".join(
        json.loads(chunk)["choices"][0]["delta"].get("content", "")
        for chunk in chunks[:-1]
    )
    assert text == "<|audio_17|><|audio_230|><|audio_18|><|audio_230|>"
    for extra in ({"temperature": 0}, {"n": 2}, {"stream": "yes"}):
        assert (
            client.post("/v1/chat/completions", json={**image, **extra}).status_code
            == 400
        )


def test_limits_ids_and_auth(client):
    assert (
        client.post(
            "/v1/tokenize", json={"version": 1, "items": [item(), item()]}
        ).status_code
        == 400
    )
    with TestClient(
        create_app(
            Engine(),
            limits=Limits(max_json_bytes=16, max_upload_bytes=20),
            api_key="test",
        )
    ) as protected:
        assert protected.post("/v1/tokenize", json={}).status_code == 401
        headers = {"Authorization": "Bearer test"}
        assert (
            protected.post(
                "/v1/tokenize", json={"long": "x" * 100}, headers=headers
            ).status_code
            == 413
        )
        assert (
            protected.post(
                "/v1/tokenize",
                data={"manifest": "{}"},
                files={"clip": ("clip.wav", b"data")},
                headers=headers,
            ).status_code
            == 413
        )


def test_partial_failure_reports_committed_range():
    class Broken(Engine):
        def iter_audio(self, source, cancel=None):
            yield next(super().iter_audio(source, cancel))
            raise TokenizationError("decoder_error", "Source ended unexpectedly")

    with TestClient(create_app(Broken())) as client:
        response = client.post(
            "/v1/tokenize",
            json={"version": 1, "stream": True, "items": [item("audio")]},
        )
    frames = FrameDecoder().feed(response.content)
    error = frames[-2]
    assert error["frame"] == "error"
    assert error["committed_codes"] == 2 and error["committed_samples"] == 480000


def test_bounded_admission():
    entered, release = threading.Event(), threading.Event()

    class Slow(Engine):
        def encode_image(self, source, cancel=None):
            entered.set()
            assert release.wait(5)
            return super().encode_image(source, cancel)

    with TestClient(create_app(Slow(), limits=Limits(max_requests=1))) as client:
        first = []
        thread = threading.Thread(
            target=lambda: first.append(
                client.post("/v1/tokenize", json={"version": 1, "items": [item()]})
            )
        )
        thread.start()
        try:
            assert entered.wait(5)
            response = client.post(
                "/v1/tokenize", json={"version": 1, "items": [item()]}
            )
            assert response.status_code == 429
        finally:
            release.set()
            thread.join(5)
        assert first[0].status_code == 200


def test_stream_backpressure_and_disconnect_cleanup():
    calls = []
    closed = threading.Event()

    class Observed(Engine):
        def iter_audio(self, source, cancel=None):
            try:
                for part in super().iter_audio(source, cancel):
                    calls.append(part["segment_index"])
                    yield part
            finally:
                closed.set()

    async def scenario():
        app = create_app(Observed())
        inbox = asyncio.Queue()
        body = json.dumps(
            {"version": 1, "stream": True, "items": [item("audio")]}
        ).encode()
        await inbox.put({"type": "http.request", "body": body, "more_body": False})
        sent_start = asyncio.Event()
        allow_start = asyncio.Event()
        sent_data = asyncio.Event()
        hold_data = asyncio.Event()
        decoder = FrameDecoder()

        async def receive():
            return await inbox.get()

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                for frame in decoder.feed(message["body"]):
                    if frame["frame"] == "start":
                        sent_start.set()
                        await allow_start.wait()
                    if frame["frame"] == "data":
                        sent_data.set()
                        await hold_data.wait()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/tokenize",
            "raw_path": b"/v1/tokenize",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "server": ("test", 80),
            "client": ("test", 123),
        }
        task = asyncio.create_task(app(scope, receive, send))
        await asyncio.wait_for(sent_start.wait(), 5)
        assert calls == []
        allow_start.set()
        await asyncio.wait_for(sent_data.wait(), 5)
        assert calls == [0]
        await inbox.put({"type": "http.disconnect"})
        await asyncio.wait_for(task, 5)
        assert closed.is_set()
        with TestClient(app) as client:
            assert client.get("/health").json()["active_requests"] == 0

    asyncio.run(scenario())


def test_webui_assets_and_no_store(client):
    page = client.get("/")
    assert page.status_code == 200
    assert 'data-upload-limit="16777216"' in page.text
    assert 'id="crop-canvas"' in page.text
    for path, content_type in (
        ("/", "text/html"),
        ("/ui/app.js", "text/javascript"),
        ("/ui/style.css", "text/css"),
    ):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith(content_type)
        assert response.headers["cache-control"] == "no-store"
        assert "connect-src 'self'" in response.headers["content-security-policy"]
    assert client.get("/ui/server.py").status_code == 404


def test_webui_receives_server_image_defaults_and_budget():
    engine = Engine()
    engine.image_options = ImageOptions(min_short_edge=512, max_long_edge=2048)
    engine.max_image_pixels = 8_000_000
    with TestClient(create_app(engine)) as client:
        html = client.get("/").text
        assert 'data-min-short-edge="512"' in html
        assert 'data-max-long-edge="2048"' in html
        assert 'data-max-image-pixels="8000000"' in html
        assert "{{" not in html


@pytest.mark.parametrize(
    "options",
    [
        None,
        [],
        {"max_long_edge": True},
        {"max_long_edge": "2048"},
        {"min_short_edge": 17},
        {"resize": 1},
        {"max_image_pixels": 1},
    ],
)
def test_http_rejects_invalid_image_options(client, options):
    value = item()
    value["image_options"] = options
    assert (
        client.post("/v1/tokenize", json={"version": 1, "items": [value]}).status_code
        == 400
    )


@pytest.mark.parametrize(
    "query",
    [
        "max_long_edge=x",
        "max_long_edge=17",
        "max_long_edge=2048&max_long_edge=4096",
        "max_image_pixels=999999999",
    ],
)
def test_webui_rejects_invalid_policy_queries(client, query):
    response = client.post(
        "/ui/encode?" + query, content=b"media", headers={"Content-Type": "image/png"}
    )
    assert response.status_code == 400
    assert client.get("/health").json()["active_requests"] == 0


def test_webui_uses_memory_and_reports_encoding_time(monkeypatch):
    def no_directory(*args, **kwargs):
        raise AssertionError("WebUI must not create a media directory")

    class MemoryEngine(Engine):
        def encode_image(self, source, cancel=None, resize=True):
            assert isinstance(source, bytes)
            assert resize is False
            time.sleep(0.01)
            return super().encode_image(source, cancel)

        def encode_audio(self, source, cancel=None):
            assert isinstance(source, bytes)
            return super().encode_audio(source, cancel)

    monkeypatch.setattr(
        "omni_tokenizer.server.tempfile.TemporaryDirectory", no_directory
    )
    with TestClient(create_app(MemoryEngine())) as client:
        for media_type, expected in (
            ("image/png", "<|image_7|><|image_8|>\n<|image_9|><|image_10|>"),
            ("audio/wav", "<|audio_17|><|audio_230|><|audio_18|><|audio_230|>"),
        ):
            response = client.post(
                "/ui/encode", content=b"media", headers={"Content-Type": media_type}
            )
            assert response.status_code == 200
            result = response.json()
            assert result["text"] == expected and result["code_count"] == 4
            assert result["inference_ms"] >= 0
            if media_type == "image/png":
                assert result["inference_ms"] >= 10
            assert response.headers["server-timing"].startswith("inference;dur=")
            assert response.headers["cache-control"] == "no-store"
            assert client.get("/health").json()["active_requests"] == 0
        bad = client.post(
            "/ui/encode", content=b"bad", headers={"Content-Type": "image/png"}
        )
        assert bad.status_code == 422 and "text" not in bad.json()


def test_webui_auth_limits_and_failed_requests_release_admission():
    with TestClient(
        create_app(Engine(), limits=Limits(max_json_bytes=4), api_key="test")
    ) as client:
        assert client.get("/").status_code == 200
        assert client.post("/ui/encode", content=b"x").status_code == 401
        headers = {"Authorization": "Bearer test", "Content-Type": "image/png"}
        for body, status in (
            (b"12345", 413),
            (b"", 400),
            (b"bad", 422),
            (b"good", 200),
        ):
            response = client.post("/ui/encode", content=body, headers=headers)
            assert response.status_code == status
            assert response.headers["cache-control"] == "no-store"
            assert client.get("/health").json()["active_requests"] == 0
        headers["Content-Type"] = "text/plain"
        assert (
            client.post("/ui/encode", content=b"text", headers=headers).status_code
            == 415
        )


def test_webui_disconnect_cancels_work_and_releases_memory(monkeypatch):
    entered, cancelled = threading.Event(), threading.Event()

    class Waiting(Engine):
        def encode_image(self, source, cancel=None, resize=True):
            assert isinstance(source, bytes)
            assert resize is False
            entered.set()
            assert cancel.event.wait(5)
            cancelled.set()
            cancel.check()

    async def scenario():
        app = create_app(Waiting())
        inbox = asyncio.Queue()
        await inbox.put({"type": "http.request", "body": b"media", "more_body": False})

        async def receive():
            return await inbox.get()

        async def send(message):
            pass

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/ui/encode",
            "raw_path": b"/ui/encode",
            "query_string": b"",
            "headers": [(b"content-type", b"image/png")],
            "server": ("test", 80),
            "client": ("test", 123),
        }
        task = asyncio.create_task(app(scope, receive, send))
        assert await asyncio.to_thread(entered.wait, 5)
        await inbox.put({"type": "http.disconnect"})
        await asyncio.wait_for(task, 5)
        assert cancelled.is_set()
        with TestClient(app) as client:
            assert client.get("/health").json()["active_requests"] == 0

    asyncio.run(scenario())
