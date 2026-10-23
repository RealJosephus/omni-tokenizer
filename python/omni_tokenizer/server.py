"""Optional ASGI HTTP adapter with bounded admission and disk-spooled uploads."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from .image import DEFAULT_MAX_PIXELS, ImageOptions, validate_image_options
from .protocol import (
    BINARY_MEDIA_TYPE,
    STREAM_MEDIA_TYPE,
    encode_binary,
    encode_frame,
    token_text,
)
from .runtime import Cancellation, error_record


@dataclass
class Limits:
    max_json_bytes: int = 16 * 1024 * 1024
    max_upload_bytes: int = 4 * 1024 * 1024 * 1024
    max_items: int = 16
    max_requests: int = 4
    max_codes: int = 2_000_000
    url_hosts: tuple[str, ...] = ()


class RequestError(ValueError):
    def __init__(self, message, status=400, code="invalid_request"):
        super().__init__(message)
        self.status, self.code = status, code


def _image_options(value):
    try:
        return validate_image_options(value)
    except ValueError as exc:
        raise RequestError(str(exc)) from exc


class _ResponseHeaders:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        async def with_headers(message):
            if message["type"] == "http.response.start":
                headers = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() != b"cache-control"
                ]
                headers.extend(
                    [
                        (b"cache-control", b"no-store"),
                        (b"x-content-type-options", b"nosniff"),
                        (b"referrer-policy", b"no-referrer"),
                        (
                            b"content-security-policy",
                            (
                                b"default-src 'none'; script-src 'self'; style-src 'self'; "
                                b"img-src 'self' blob: data:; media-src blob:; connect-src 'self'; "
                                b"base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
                            ),
                        ),
                    ]
                )
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, with_headers)


def _b64(value, limit):
    if not isinstance(value, str) or len(value) > ((limit + 2) // 3) * 4:
        raise RequestError("Invalid or oversized base64 media")
    try:
        result = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise RequestError("Invalid base64 media") from exc
    if not result or len(result) > limit:
        raise RequestError("Empty or oversized media")
    return result


def validate_manifest(value, limits):
    if (
        not isinstance(value, dict)
        or set(value) - {"version", "items", "stream"}
        or value.get("version") != 1
        or type(value.get("version")) is not int
    ):
        raise RequestError("Expected version 1 manifest with items and optional stream")
    if type(value.get("stream", False)) is not bool:
        raise RequestError("stream must be a boolean")
    items = value.get("items")
    if not isinstance(items, list) or not 0 < len(items) <= limits.max_items:
        raise RequestError(f"items must contain 1 to {limits.max_items} entries")
    seen = set()
    for item in items:
        if not isinstance(item, dict) or set(item) - {
            "item_id",
            "type",
            "part",
            "data",
            "format",
            "image_options",
        }:
            raise RequestError("Unknown item fields")
        item_id = item.get("item_id")
        if (
            not isinstance(item_id, str)
            or not item_id
            or len(item_id.encode("utf-8")) > 256
            or item_id in seen
        ):
            raise RequestError(
                "item_id must be a unique nonempty UTF-8 string of at most 256 bytes"
            )
        seen.add(item_id)
        if item.get("type") not in {"audio", "image"}:
            raise RequestError("Item type must be audio or image")
        if "image_options" in item:
            if item["type"] != "image":
                raise RequestError("image_options requires an image item")
            _image_options(item["image_options"])
        if ("part" in item) == ("data" in item):
            raise RequestError(
                "Each item must name one multipart part or supply base64 data"
            )
        if "part" in item and (
            not isinstance(item["part"], str)
            or not item["part"]
            or len(item["part"]) > 256
        ):
            raise RequestError("Invalid part name")
        if "format" in item and item["format"] not in {
            "wav",
            "mp3",
            "flac",
            "ogg",
            "m4a",
            "png",
            "jpeg",
            "jpg",
            "webp",
        }:
            raise RequestError("Unsupported format")
    return value


def parse_chat(value):
    allowed = {"model", "messages", "stream", "image_options"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise RequestError(
            "Supported chat fields are model, messages, stream, and image_options"
        )
    if type(value.get("stream", False)) is not bool or (
        "model" in value and value["model"] != "omni-tokenizer"
    ):
        raise RequestError("model must be omni-tokenizer and stream must be boolean")
    messages = value.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise RequestError("Exactly one user message with one media item is supported")
    message = messages[0]
    if (
        not isinstance(message, dict)
        or set(message) != {"role", "content"}
        or message.get("role") != "user"
    ):
        raise RequestError("Expected a user message")
    content = message["content"]
    if (
        not isinstance(content, list)
        or len(content) != 1
        or not isinstance(content[0], dict)
    ):
        raise RequestError("Exactly one image_url or input_audio item is supported")
    media = content[0]
    kind = media.get("type")
    if kind == "image_url":
        image = media.get("image_url")
        if (
            set(media) != {"type", "image_url"}
            or not isinstance(image, dict)
            or set(image) - {"url", "detail"}
            or not isinstance(image.get("url"), str)
        ):
            raise RequestError("Invalid image_url")
        if image.get("detail", "auto") != "auto":
            raise RequestError("Only image_url.detail=auto is supported")
        options = _image_options(value.get("image_options", {}))
        return {
            "item_id": "0",
            "type": "image",
            "url": image["url"],
            "image_options": options,
        }, value.get("stream", False)
    if kind == "input_audio":
        if "image_options" in value:
            raise RequestError("image_options requires an image item")
        audio = media.get("input_audio")
        if (
            set(media) != {"type", "input_audio"}
            or not isinstance(audio, dict)
            or set(audio) != {"data", "format"}
            or audio["format"] not in {"wav", "mp3"}
        ):
            raise RequestError("input_audio requires base64 data and format wav or mp3")
        return {"item_id": "0", "type": "audio", **audio}, value.get("stream", False)
    raise RequestError("Only image_url and input_audio are supported")


def _download(url, target, limits):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *_):
            raise RequestError("Image URL redirects are not supported")

    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in limits.url_hosts
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        raise RequestError(
            "Remote image URLs require HTTPS and an explicitly allowed host"
        )
    opener = urllib.request.build_opener(NoRedirect())
    with opener.open(url, timeout=15) as response, target.open("wb") as output:
        total = 0
        while chunk := response.read(65536):
            total += len(chunk)
            if total > limits.max_json_bytes:
                raise RequestError(
                    "Remote image exceeds media limit", 413, "input_limit"
                )
            output.write(chunk)


async def _json_request(request, limit):
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > limit:
            raise RequestError("JSON request exceeds limit", 413, "input_limit")
        body.extend(chunk)
    try:
        return json.loads(body)
    except (ValueError, UnicodeError) as exc:
        raise RequestError("Invalid JSON") from exc


async def _webui_input(request, limits):
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    kinds = {
        "image/png": "image",
        "image/jpeg": "image",
        "image/webp": "image",
        "audio/wav": "audio",
        "audio/x-wav": "audio",
        "audio/mpeg": "audio",
    }
    if content_type not in kinds:
        raise RequestError("Use PNG, JPEG, WebP, PCM WAV, or MP3", 415)
    params = request.query_params
    if set(params) - {"min_short_edge", "max_long_edge"} or any(
        len(params.getlist(key)) != 1 for key in params
    ):
        raise RequestError(
            "Expected unique min_short_edge and max_long_edge query parameters"
        )
    if params and kinds[content_type] != "image":
        raise RequestError("Image resize parameters require an image")
    try:
        values = {key: int(value) for key, value in params.items()}
    except ValueError as exc:
        raise RequestError("Image resize parameters must be integers") from exc
    options = _image_options(values)
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > limits.max_json_bytes:
            raise RequestError(
                "Media exceeds the WebUI upload limit", 413, "input_limit"
            )
        body.extend(chunk)
    if not body:
        raise RequestError("Media is empty")
    return [
        {
            "item_id": "0",
            "type": kinds[content_type],
            "source": bytes(body),
            "image_options": options,
        }
    ], False


async def _inputs(request, directory, limits, *, chat=False):
    from starlette.datastructures import UploadFile
    from starlette.requests import Request

    parts = {}
    content_type = (
        request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    )
    if content_type == "application/json":
        value = await _json_request(request, limits.max_json_bytes)
        if chat:
            item, stream = parse_chat(value)
            value = {"version": 1, "items": [item], "stream": stream}
        else:
            validate_manifest(value, limits)
    elif content_type == "multipart/form-data" and not chat:
        total = 0
        receive = request.receive

        async def limited_receive():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > limits.max_upload_bytes:
                raise RequestError("Upload exceeds limit", 413, "input_limit")
            return message

        bounded_request = Request(request.scope, limited_receive)
        async with bounded_request.form(
            max_files=limits.max_items, max_fields=1, max_part_size=1024 * 1024
        ) as form:
            entries = list(form.multi_items())
            if len({name for name, _ in entries}) != len(entries):
                raise RequestError("Duplicate multipart part names")
            manifest = form.get("manifest")
            if not isinstance(manifest, str):
                raise RequestError(
                    "Multipart requests require a JSON manifest text field"
                )
            try:
                value = validate_manifest(json.loads(manifest), limits)
            except (ValueError, UnicodeError) as exc:
                raise RequestError(str(exc)) from exc
            for index, (name, upload) in enumerate(entries):
                if name == "manifest":
                    continue
                if not isinstance(upload, UploadFile):
                    raise RequestError("Media parts must be file uploads")
                target = directory / f"part-{index}.media"
                with target.open("wb") as output:
                    while chunk := await upload.read(65536):
                        output.write(chunk)
                parts[name] = target
    else:
        raise RequestError(
            "Expected application/json"
            + (" or multipart/form-data" if not chat else ""),
            415,
            "unsupported_media_type",
        )
    inputs = []
    for index, item in enumerate(value["items"]):
        prepared = {"item_id": item["item_id"], "type": item["type"]}
        if "image_options" in item:
            prepared["image_options"] = item["image_options"]
        try:
            if "part" in item:
                if item["part"] not in parts:
                    raise RequestError(f"Missing media part: {item['part']}")
                target = parts[item["part"]]
            else:
                target = directory / f"item-{index}.media"
                if "url" in item:
                    url = item["url"]
                    if url.startswith("data:"):
                        prefix, separator, data = url.partition(",")
                        if not separator or prefix not in {
                            "data:image/png;base64",
                            "data:image/jpeg;base64",
                            "data:image/webp;base64",
                        }:
                            raise RequestError(
                                "Expected a base64 PNG, JPEG, or WebP data URL"
                            )
                        target.write_bytes(_b64(data, limits.max_json_bytes))
                    else:
                        await asyncio.to_thread(_download, url, target, limits)
                else:
                    target.write_bytes(_b64(item.get("data"), limits.max_json_bytes))
            prepared["source"] = target
        except Exception as exc:
            prepared["error"] = {"code": "invalid_media", "message": str(exc)}
        inputs.append(prepared)
    return inputs, value.get("stream", False)


def _aggregate(engine, inputs, cancel, limit, *, resize_images=True):
    results = []
    total_codes = 0
    for item in inputs:
        cancel.check()
        result = {"item_id": item["item_id"], "type": item["type"]}
        try:
            if "error" in item:
                result.update(status="error", error=item["error"])
            else:
                fn = (
                    engine.encode_image
                    if item["type"] == "image"
                    else engine.encode_audio
                )
                options = dict(item.get("image_options", {}))
                if item["type"] == "image" and not resize_images:
                    options["resize"] = False
                output = fn(item["source"], cancel=cancel, **options)
                total_codes += len(output["codes"])
                if total_codes > limit:
                    raise RequestError(
                        "Aggregate code limit exceeded; use stream=true",
                        413,
                        "output_limit",
                    )
                result.update(output)
        except Exception as exc:
            result.update(status="error", error=error_record(exc))
        results.append(result)
    return results


def _frames(engine, inputs, cancel):
    for item in inputs:
        identity = {"item_id": item["item_id"], "type": item["type"]}
        yield {
            "frame": "start",
            **identity,
            "sample_rate": 16000 if item["type"] == "audio" else None,
        }
        total_codes = total_samples = 0
        iterator = None
        try:
            if "error" in item:
                yield {
                    "frame": "error",
                    **identity,
                    "error": item["error"],
                    "committed_codes": 0,
                    "committed_samples": 0,
                }
                continue
            iterator = (
                engine.iter_audio(item["source"], cancel=cancel)
                if item["type"] == "audio"
                else iter(
                    [
                        engine.encode_image(
                            item["source"],
                            cancel=cancel,
                            **item.get("image_options", {}),
                        )
                    ]
                )
            )
            for output in iterator:
                cancel.check()
                total_codes += len(output["codes"])
                total_samples += output.get("valid_samples", 0)
                yield {"frame": "data", **identity, **output}
            yield {
                "frame": "end",
                **identity,
                "code_count": total_codes,
                "valid_samples": total_samples,
                "status": "ok",
            }
        except Exception as exc:
            yield {
                "frame": "error",
                **identity,
                "error": error_record(exc),
                "committed_codes": total_codes,
                "committed_samples": total_samples,
            }
        finally:
            if iterator is not None and hasattr(iterator, "close"):
                iterator.close()
    yield {"frame": "done", "item_count": len(inputs)}


def create_app(engine, *, limits=None, api_key=None):
    """Create an ASGI app. Keep one worker process per device/model set."""
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.requests import ClientDisconnect
    from starlette.responses import JSONResponse, Response, StreamingResponse
    from starlette.routing import Route

    limits = limits or Limits()
    image_defaults = getattr(engine, "image_options", ImageOptions())
    max_image_pixels = getattr(engine, "max_image_pixels", DEFAULT_MAX_PIXELS)
    active = 0

    async def health(_):
        return JSONResponse({"status": "ok", "version": 1, "active_requests": active})

    async def webui(request):
        name = request.path_params.get("asset", "index.html")
        types = {
            "index.html": "text/html",
            "app.js": "text/javascript",
            "style.css": "text/css",
        }
        if name not in types:
            return Response(status_code=404)
        content = files("omni_tokenizer").joinpath("web", name).read_bytes()
        if name == "index.html":
            settings = {
                "upload_limit": limits.max_json_bytes,
                "min_short_edge": image_defaults.min_short_edge,
                "max_long_edge": image_defaults.max_long_edge,
                "max_image_pixels": max_image_pixels,
            }
            for key, value in settings.items():
                content = content.replace(
                    ("{{" + key + "}}").encode("ascii"), str(value).encode("ascii")
                )
        return Response(content, media_type=types[name])

    async def endpoint(request):
        nonlocal active
        if api_key and request.headers.get("authorization") != "Bearer " + api_key:
            return JSONResponse(
                {"error": {"code": "unauthorized", "message": "Invalid API key"}}, 401
            )
        if active >= limits.max_requests:
            return JSONResponse(
                {"error": {"code": "busy", "message": "Request queue is full"}},
                429,
                headers={"Retry-After": "1"},
            )
        active += 1
        ui = request.url.path == "/ui/encode"
        directory = None
        inputs = []
        cancel = Cancellation()
        released = False

        def release():
            nonlocal active, released
            if not released:
                if directory is not None:
                    directory.cleanup()
                inputs.clear()
                active -= 1
                released = True

        chat = request.url.path.endswith("/chat/completions")
        handed_off = False
        job = None
        try:
            if ui:
                inputs, stream = await _webui_input(request, limits)
            else:
                directory = tempfile.TemporaryDirectory(prefix="omni-tokenizer-")
                inputs, stream = await _inputs(
                    request, Path(directory.name), limits, chat=chat
                )
            if stream:
                request_id = "chatcmpl-" + uuid.uuid4().hex
                created = int(time.time())

                def event(delta, finish=None):
                    return (
                        "data: "
                        + json.dumps(
                            {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": "omni-tokenizer",
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": delta,
                                        "finish_reason": finish,
                                    }
                                ],
                            },
                            separators=(",", ":"),
                        )
                        + "\n\n"
                    )

                iterator = _frames(engine, inputs, cancel)
                stream_started = False

                def pull():
                    return next(iterator, None)

                async def output():
                    nonlocal stream_started
                    stream_started = True
                    task = None
                    try:
                        if chat:
                            yield event({"role": "assistant"})
                        while True:
                            task = asyncio.create_task(asyncio.to_thread(pull))
                            frame = await asyncio.shield(task)
                            task = None
                            if frame is None:
                                break
                            if chat:
                                if frame["frame"] == "data":
                                    yield event(
                                        {
                                            "content": token_text(
                                                frame["codes"],
                                                frame["type"],
                                                frame.get("shape"),
                                            )
                                        }
                                    )
                                elif frame["frame"] == "error":
                                    yield (
                                        "data: "
                                        + json.dumps({"error": frame["error"]})
                                        + "\n\n"
                                    )
                                    return
                                elif frame["frame"] == "done":
                                    yield event({}, "stop")
                                    yield "data: [DONE]\n\n"
                            else:
                                yield encode_frame(frame)
                    finally:
                        cancel.cancel()

                        def finish(_=None):
                            try:
                                iterator.close()
                            finally:
                                release()

                        if task is not None and not task.done():
                            task.add_done_callback(finish)
                        else:
                            finish()

                class ManagedStream(StreamingResponse):
                    async def __call__(self, scope, receive, send):
                        try:
                            await super().__call__(scope, receive, send)
                        finally:
                            cancel.cancel()
                            await self.body_iterator.aclose()
                            if not stream_started:
                                iterator.close()
                                release()

                handed_off = True
                return ManagedStream(
                    output(),
                    media_type="text/event-stream" if chat else STREAM_MEDIA_TYPE,
                    headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
                )

            def encode():
                started = time.perf_counter()
                results = _aggregate(
                    engine, inputs, cancel, limits.max_codes, resize_images=not ui
                )
                return results, (time.perf_counter() - started) * 1000

            job = asyncio.create_task(asyncio.to_thread(encode))
            while not job.done():
                await asyncio.wait({job}, timeout=0.1)
                if await request.is_disconnected():
                    cancel.cancel()
            results, inference_ms = await job
            if ui:
                result = results[0]
                if result["status"] != "ok":
                    return JSONResponse({"error": result["error"]}, 422)
                return JSONResponse(
                    {
                        "text": token_text(
                            result["codes"], result["type"], result.get("shape")
                        ),
                        "type": result["type"],
                        "shape": result.get("shape"),
                        "code_count": len(result["codes"]),
                        "inference_ms": inference_ms,
                        "image_options": result.get("image_options"),
                    },
                    headers={"Server-Timing": f"inference;dur={inference_ms:.3f}"},
                )
            if chat:
                result = results[0]
                if result["status"] != "ok":
                    return JSONResponse({"error": result["error"]}, 422)
                return JSONResponse(
                    {
                        "id": "chatcmpl-" + uuid.uuid4().hex,
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": "omni-tokenizer",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": token_text(
                                        result["codes"],
                                        result["type"],
                                        result.get("shape"),
                                    ),
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )
            if BINARY_MEDIA_TYPE in request.headers.get("accept", ""):
                return Response(encode_binary(results), media_type=BINARY_MEDIA_TYPE)
            return JSONResponse({"version": 1, "items": results})
        except RequestError as exc:
            return JSONResponse(
                {"error": {"code": exc.code, "message": str(exc)}}, exc.status
            )
        except ClientDisconnect:
            cancel.cancel()
            return Response(status_code=499)
        except Exception as exc:
            return JSONResponse({"error": error_record(exc)}, 400)
        finally:
            if not handed_off:
                cancel.cancel()
                if job is not None and not job.done():
                    job.add_done_callback(lambda _: release())
                else:
                    release()

    return Starlette(
        middleware=[Middleware(_ResponseHeaders)],
        routes=[
            Route("/", webui),
            Route("/ui/encode", endpoint, methods=["POST"]),
            Route("/ui/{asset}", webui),
            Route("/health", health),
            Route("/v1/tokenize", endpoint, methods=["POST"]),
            Route("/v1/chat/completions", endpoint, methods=["POST"]),
        ],
    )
