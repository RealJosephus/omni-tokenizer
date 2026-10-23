"""Command-line entry points; token text is always written with literal LF."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

from .protocol import encode_binary, encode_frame, token_text
from .runtime import Cancellation, Tokenizer, error_record


def _dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _parser():
    parser = argparse.ArgumentParser(prog="omni-tokenizer")
    parser.add_argument("--audio-model", default=os.environ.get("OMNI_AUDIO_MODEL"))
    parser.add_argument("--image-model", default=os.environ.get("OMNI_IMAGE_MODEL"))
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N; precision does not change automatically")
    parser.add_argument("--library", default=os.environ.get("OMNI_LIBRARY"))
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Inspect a converted model manifest")
    inspect.add_argument("model")
    inspect.add_argument("--verify", action="store_true", help="Read and verify the entire model SHA-256")
    encode = commands.add_parser("encode", help="Encode one image or audio file")
    encode.add_argument("type", choices=["audio", "image"])
    encode.add_argument("input")
    encode.add_argument("--format", choices=["json", "tokens", "binary", "stream", "jsonl"], default="json")
    encode.add_argument("-o", "--output")
    batch = commands.add_parser("batch", help="Encode JSONL items with item_id, type and path")
    batch.add_argument("input")
    batch.add_argument("--format", choices=["jsonl", "stream"], default="jsonl")
    batch.add_argument("-o", "--output")
    bench = commands.add_parser("bench", help="Measure cold and warm end-to-end latency")
    bench.add_argument("type", choices=["audio", "image"])
    bench.add_argument("input")
    bench.add_argument("--repeat", type=int, default=3)
    serve = commands.add_parser("serve", help="Run the optional HTTP server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--api-key", default=os.environ.get("OMNI_API_KEY"))
    serve.add_argument("--max-requests", type=int, default=4)
    serve.add_argument("--max-upload-bytes", type=int, default=4*1024**3)
    serve.add_argument("--allow-url-host", action="append", default=[])
    return parser


def _inspect(path, verify):
    root = Path(path)
    if root.is_file():
        root = root.parent
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    summary = {k: v for k, v in manifest.items() if k != "tensors"}
    summary["tensor_count"] = len(manifest.get("tensors", []))
    if verify:
        digest = hashlib.sha256()
        with (root / "model.otm").open("rb") as source:
            for chunk in iter(lambda: source.read(8*1024**2), b""):
                digest.update(chunk)
        if digest.hexdigest() != manifest.get("model_sha256"):
            raise ValueError("Model SHA-256 mismatch")
        summary["verified"] = True
    return summary


def _encode(engine, kind, source, format, output):
    cancel = Cancellation()
    if format in {"stream", "jsonl"}:
        from .server import _frames
        for frame in _frames(engine, [{"item_id": "0", "type": kind, "source": source}], cancel):
            output.write(encode_frame(frame) if format == "stream" else _dump(frame)+b"\n")
            output.flush()
        return
    if kind == "audio" and format == "tokens":
        for segment in engine.iter_audio(source, cancel=cancel):
            output.write(token_text(segment["codes"], "audio").encode("ascii"))
            output.flush()
        return
    result = (engine.encode_image if kind == "image" else engine.encode_audio)(source, cancel=cancel)
    result["item_id"] = "0"
    if format == "tokens":
        output.write(token_text(result["codes"], kind, result.get("shape")).encode("ascii"))
    elif format == "binary":
        output.write(encode_binary([result]))
    else:
        output.write(_dump({"version": 1, "items": [result]})+b"\n")


def _batch(engine, args, output):
    from .server import _frames
    seen = set()
    any_error = False
    with Path(args.input).open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if not line.strip():
                continue
            item_id = str(index)
            try:
                item = json.loads(line)
                if not isinstance(item, dict) or set(item) != {"item_id", "type", "path"}:
                    raise ValueError("Expected item_id, type and path")
                item_id = item["item_id"]
                if not isinstance(item_id, str) or not item_id or item_id in seen:
                    raise ValueError("item_id must be a unique nonempty string")
                seen.add(item_id)
                if item["type"] not in {"audio", "image"} or not isinstance(item["path"], str):
                    raise ValueError("Invalid media type or path")
                path = Path(item["path"])
                if not path.is_absolute():
                    path = Path(args.input).resolve().parent / path
                if args.format == "stream":
                    for frame in _frames(engine, [{"item_id": item_id, "type": item["type"], "source": path}], Cancellation()):
                        if frame["frame"] != "done":
                            output.write(encode_frame(frame))
                        any_error |= frame["frame"] == "error"
                else:
                    result = (engine.encode_image if item["type"] == "image" else engine.encode_audio)(path)
                    output.write(_dump({"item_id": item_id, **result})+b"\n")
            except Exception as exc:
                any_error = True
                result = {"item_id": item_id, "status": "error", "error": error_record(exc)}
                output.write(encode_frame({"frame": "error", **result}) if args.format == "stream" else _dump(result)+b"\n")
            output.flush()
    if args.format == "stream":
        output.write(encode_frame({"frame": "done", "item_count": len(seen)}))
    return int(any_error)


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect":
            print(json.dumps(_inspect(args.model, args.verify), indent=2))
            return 0
        with Tokenizer(audio_model=args.audio_model, image_model=args.image_model, device=args.device, library=args.library) as engine:
            if args.command == "serve":
                import uvicorn
                from .server import Limits, create_app
                if args.max_requests <= 0 or args.max_upload_bytes <= 0:
                    raise ValueError("Limits must be positive")
                uvicorn.run(create_app(engine, limits=Limits(max_requests=args.max_requests, max_upload_bytes=args.max_upload_bytes, url_hosts=tuple(args.allow_url_host)), api_key=args.api_key), host=args.host, port=args.port, workers=1)
            elif args.command == "bench":
                if not 1 <= args.repeat <= 100:
                    raise ValueError("repeat must be between 1 and 100")
                durations = []
                baseline = None
                flips = []
                for _ in range(args.repeat+1):
                    start = time.perf_counter()
                    result = (engine.encode_image if args.type == "image" else engine.encode_audio)(args.input)
                    durations.append(time.perf_counter()-start)
                    if baseline is None:
                        baseline = result["codes"]
                    flips.append(sum(a != b for a, b in zip(baseline, result["codes"])))
                report = {"device": args.device, "profile": result["profile"], "codes": len(baseline), "cold_seconds": durations[0], "warm_seconds": durations[1:], "warm_median_seconds": statistics.median(durations[1:]), "repeat_code_flips": flips, "context_peak_workspace_bytes": engine.last_workspace_bytes, "memory_note": "Context workspace excludes model weights, driver and library allocations."}
                print(json.dumps(report, indent=2))
            else:
                output_context = Path(args.output).open("wb") if args.output else nullcontext(sys.stdout.buffer)
                with output_context as output:
                    if args.command == "batch":
                        return _batch(engine, args, output)
                    _encode(engine, args.type, args.input, args.format, output)
        return 0
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        return 141
    except Exception as exc:
        print(json.dumps({"error": error_record(exc)}), file=sys.stderr)
        return 1
