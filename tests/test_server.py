import base64
import asyncio
import json
import threading
import time

import pytest
from starlette.testclient import TestClient

from omni_tokenizer.protocol import BINARY_MEDIA_TYPE, FrameDecoder, decode_binary
from omni_tokenizer.runtime import TokenizationError
from omni_tokenizer.server import Limits, create_app


class Engine:
    def encode_image(self, source, cancel=None):
        if source.read_bytes() == b"bad":
            raise ValueError("Invalid image")
        return {"type": "image", "status": "ok", "shape": [2, 2], "codes": [7, 8, 9, 10], "profile": "fixture"}

    def iter_audio(self, source, cancel=None):
        for index in range(2):
            cancel.check()
            yield {"type": "audio", "status": "ok", "segment_index": index, "start_sample": index*480000, "valid_samples": 480000, "padded_samples": 480000, "code_offset": index*2, "code_count": 2, "codes": [17+index, 230], "profile": "fixture"}

    def encode_audio(self, source, cancel=None):
        return {"type": "audio", "status": "ok", "codes": [c for segment in self.iter_audio(source, cancel) for c in segment["codes"]]}


@pytest.fixture
def client():
    with TestClient(create_app(Engine())) as value:
        yield value


def item(kind="image", item_id="a", value=b"media"):
    return {"item_id": item_id, "type": kind, "data": base64.b64encode(value).decode()}


def test_mixed_batch_binary_errors_and_order(client):
    value = {"version": 1, "items": [item(), item("audio", "b"), item("image", "c", b"bad")]}
    response = client.post("/v1/tokenize", json=value)
    assert response.status_code == 200
    entries = response.json()["items"]
    assert [entry["item_id"] for entry in entries] == ["a", "b", "c"]
    assert entries[2]["status"] == "error"
    binary = client.post("/v1/tokenize", json=value, headers={"Accept": BINARY_MEDIA_TYPE})
    decoded = decode_binary(binary.content)
    assert decoded["items"][0]["codes"] == entries[0]["codes"]
    assert decoded["items"][1]["codes"] == entries[1]["codes"]


def test_multipart_stream_and_terminal_error(client):
    manifest = {"version": 1, "stream": True, "items": [{"item_id": "audio", "type": "audio", "part": "clip"}, {"item_id": "missing", "type": "image", "part": "absent"}]}
    response = client.post("/v1/tokenize", data={"manifest": json.dumps(manifest)}, files={"clip": ("clip.wav", b"media")})
    assert response.status_code == 200
    decoder = FrameDecoder()
    frames = decoder.feed(response.content)
    decoder.finish()
    assert [frame["frame"] for frame in frames] == ["start", "data", "data", "end", "start", "error", "done"]
    assert frames[2]["code_offset"] == 2
    assert frames[5]["committed_codes"] == 0


def test_chat_strict_image_and_audio_sse(client):
    image = {"model": "omni-tokenizer", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,bWVkaWE="}}]}]}
    response = client.post("/v1/chat/completions", json=image)
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "<|image_7|><|image_8|>\n<|image_9|><|image_10|>"
    audio = {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "bWVkaWE=", "format": "wav"}}]}], "stream": True}
    response = client.post("/v1/chat/completions", json=audio)
    chunks = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    assert chunks[-1] == "[DONE]"
    text = "".join(json.loads(chunk)["choices"][0]["delta"].get("content", "") for chunk in chunks[:-1])
    assert text == "<|audio_17|><|audio_230|><|audio_18|><|audio_230|>"
    for extra in ({"temperature": 0}, {"n": 2}, {"stream": "yes"}):
        assert client.post("/v1/chat/completions", json={**image, **extra}).status_code == 400


def test_limits_ids_and_auth(client):
    assert client.post("/v1/tokenize", json={"version": 1, "items": [item(), item()]}).status_code == 400
    with TestClient(create_app(Engine(), limits=Limits(max_json_bytes=16, max_upload_bytes=20), api_key="test")) as protected:
        assert protected.post("/v1/tokenize", json={}).status_code == 401
        headers = {"Authorization": "Bearer test"}
        assert protected.post("/v1/tokenize", json={"long": "x"*100}, headers=headers).status_code == 413
        assert protected.post("/v1/tokenize", data={"manifest": "{}"}, files={"clip": ("clip.wav", b"data")}, headers=headers).status_code == 413


def test_partial_failure_reports_committed_range():
    class Broken(Engine):
        def iter_audio(self, source, cancel=None):
            yield next(super().iter_audio(source, cancel))
            raise TokenizationError("decoder_error", "Source ended unexpectedly")
    with TestClient(create_app(Broken())) as client:
        response = client.post("/v1/tokenize", json={"version": 1, "stream": True, "items": [item("audio")]})
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
        thread = threading.Thread(target=lambda: first.append(client.post("/v1/tokenize", json={"version": 1, "items": [item()]})))
        thread.start()
        try:
            assert entered.wait(5)
            response = client.post("/v1/tokenize", json={"version": 1, "items": [item()]})
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
        body = json.dumps({"version": 1, "stream": True, "items": [item("audio")]}).encode()
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
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1", "method": "POST", "scheme": "http", "path": "/v1/tokenize", "raw_path": b"/v1/tokenize", "query_string": b"", "headers": [(b"content-type", b"application/json")], "server": ("test", 80), "client": ("test", 123)}
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
