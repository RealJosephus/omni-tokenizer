"""Convert trusted local checkpoints to the encoding-only OMNIVQ01 format.

PyTorch is used only by conversion. Inspect/check work with the standard library.
No checkpoint is downloaded and source weights are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

MAGIC = b"OMNIVQ01"
HEADER = struct.Struct("<8s15I")
RECORD = struct.Struct("<4I2Q")
FIELDS = (
    "layers",
    "width",
    "heads",
    "ffn",
    "input_channels",
    "patch_size",
    "pool_size",
    "block_size",
    "pos_height",
    "pos_width",
    "code_dim",
    "vocab_size",
)
CONFIGS = {
    "audio": (16, 1280, 20, 5120, 128, 0, 4, 200, 1500, 1, 1280, 16384),
    "image": (40, 1536, 16, 6144, 3, 16, 0, 0, 128, 128, 2048, 16384),
}
DTYPES = {1: ("float32", 4), 2: ("bfloat16", 2)}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def aligned(value: int) -> int:
    return (value + 63) // 64 * 64


def expected_shapes(kind: str) -> dict[str, tuple[int, ...]]:
    if kind == "image":
        shapes = {
            "pos_embed": (1, 16384, 1536),
            "patch_embed.proj.weight": (1536, 3, 16, 16),
            "patch_embed.proj.bias": (1536,),
            "quant_conv.weight": (2048, 1536, 1, 1),
            "quant_conv.bias": (2048,),
            "quantize.embedding.weight": (16384, 2048),
        }
        for index in range(40):
            prefix = f"blocks.{index}."
            for name in ("norm1", "norm2"):
                shapes[prefix + name + ".weight"] = (1536,)
                shapes[prefix + name + ".bias"] = (1536,)
            for name, output, input_ in (
                ("attn.qkv", 4608, 1536),
                ("attn.proj", 1536, 1536),
                ("mlp.fc1", 6144, 1536),
                ("mlp.fc2", 1536, 6144),
            ):
                shapes[prefix + name + ".weight"] = (output, input_)
                shapes[prefix + name + ".bias"] = (output,)
        return shapes
    shapes = {
        "conv1.weight": (1280, 128, 3),
        "conv1.bias": (1280,),
        "conv2.weight": (1280, 1280, 3),
        "conv2.bias": (1280,),
        "embed_positions.weight": (1500, 1280),
        "codebook.weight": (16384, 1280),
    }
    for index in range(16):
        prefix = f"layers.{index}."
        for name in ("self_attn_layer_norm", "final_layer_norm"):
            shapes[prefix + name + ".weight"] = (1280,)
            shapes[prefix + name + ".bias"] = (1280,)
        for name, output, input_ in (
            ("self_attn.q_proj", 1280, 1280),
            ("self_attn.k_proj", 1280, 1280),
            ("self_attn.v_proj", 1280, 1280),
            ("self_attn.out_proj", 1280, 1280),
            ("fc1", 5120, 1280),
            ("fc2", 1280, 5120),
        ):
            shapes[prefix + name + ".weight"] = (output, input_)
            if name != "self_attn.k_proj":
                shapes[prefix + name + ".bias"] = (output,)
    return shapes


def inspect_model(path: Path) -> dict:
    """Validate index bounds, types, shapes, alignment, and non-overlap."""
    path = path / "model.otm" if path.is_dir() else path
    file_size = path.stat().st_size
    entries = []
    names = set()
    with path.open("rb") as stream:
        raw = stream.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise ValueError("Truncated model header")
        magic, version, kind_id, *remaining = HEADER.unpack(raw)
        if magic != MAGIC or version != 1 or kind_id not in (1, 2):
            raise ValueError("Unsupported model magic, version, or kind")
        config, tensor_count = remaining[:12], remaining[12]
        if tensor_count < 1 or tensor_count > 4096:
            raise ValueError("Invalid tensor count")
        (
            layers,
            width,
            heads,
            ffn,
            channels,
            patch,
            pool,
            block,
            pos_h,
            pos_w,
            code_dim,
            vocabulary,
        ) = config
        if not (
            1 <= layers <= 80
            and 1 <= width <= 8192
            and 1 <= heads <= width
            and width % heads == 0
            and 1 <= ffn <= 32768
            and 1 <= channels <= 512
            and 1 <= pos_h <= 16384
            and 0 <= pos_w <= 16384
            and 1 <= code_dim <= 8192
            and 1 <= vocabulary <= 65536
        ):
            raise ValueError("Invalid model configuration")
        if kind_id == 1 and (
            pool != 4 or patch != 0 or code_dim != width or block == 0
        ):
            raise ValueError("Invalid audio configuration")
        if kind_id == 2 and (channels != 3 or not 1 <= patch <= 64 or pos_h != pos_w):
            raise ValueError("Invalid image configuration")
        for _ in range(tensor_count):
            raw = stream.read(RECORD.size)
            if len(raw) != RECORD.size:
                raise ValueError("Truncated tensor index")
            name_length, dtype, rank, reserved, offset, nbytes = RECORD.unpack(raw)
            if (
                not 1 <= name_length <= 1024
                or not 1 <= rank <= 4
                or dtype != kind_id
                or reserved
            ):
                raise ValueError("Invalid tensor record")
            raw_shape = stream.read(rank * 8)
            if len(raw_shape) != rank * 8:
                raise ValueError("Truncated tensor dimensions")
            shape = struct.unpack("<" + "Q" * rank, raw_shape)
            raw_name = stream.read(name_length)
            if len(raw_name) != name_length:
                raise ValueError("Truncated tensor name")
            name = raw_name.decode("utf-8")
            if name in names or "\x00" in name:
                raise ValueError("Duplicate or invalid tensor name")
            names.add(name)
            if any(not 1 <= dimension <= 1 << 24 for dimension in shape):
                raise ValueError(f"Invalid tensor dimension: {name}")
            if math.prod(shape) * DTYPES[dtype][1] != nbytes:
                raise ValueError(f"Tensor shape does not match byte length: {name}")
            if offset % 64 or offset + nbytes > file_size:
                raise ValueError(f"Tensor is unaligned or outside the file: {name}")
            entries.append(
                dict(
                    name=name,
                    dtype=DTYPES[dtype][0],
                    shape=list(shape),
                    offset=offset,
                    nbytes=nbytes,
                )
            )
        index_end = stream.tell()
    previous_end = aligned(index_end)
    for entry in entries:
        if entry["offset"] < previous_end:
            raise ValueError("Tensor payloads overlap the index or one another")
        previous_end = entry["offset"] + entry["nbytes"]
    if previous_end != file_size:
        raise ValueError("Unexpected trailing model data")
    return {
        "format": "OMNIVQ01",
        "version": version,
        "kind": "audio" if kind_id == 1 else "image",
        "config": dict(zip(FIELDS, config)),
        "tensor_count": tensor_count,
        "file_bytes": file_size,
        "tensor_bytes": sum(entry["nbytes"] for entry in entries),
        "tensors": entries,
    }


def check_package(path: Path) -> dict:
    directory = path if path.is_dir() else path.parent
    model = path / "model.otm" if path.is_dir() else path
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    inspected = inspect_model(model)
    if sha256(model) != manifest["model_sha256"]:
        raise ValueError("Model SHA256 does not match manifest")
    if (
        inspected["kind"] != manifest["kind"]
        or inspected["config"] != manifest["config"]
    ):
        raise ValueError("Model header does not match manifest")
    if inspected["tensors"] != [
        {key: entry[key] for key in ("name", "dtype", "shape", "offset", "nbytes")}
        for entry in manifest["tensors"]
    ]:
        raise ValueError("Tensor index does not match manifest")
    with model.open("rb") as stream:
        for entry in manifest["tensors"]:
            stream.seek(entry["offset"])
            digest = hashlib.sha256()
            remaining = entry["nbytes"]
            while remaining:
                chunk = stream.read(min(remaining, 8 * 1024 * 1024))
                if not chunk:
                    raise ValueError("Truncated tensor payload")
                digest.update(chunk)
                remaining -= len(chunk)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"Tensor SHA256 mismatch: {entry['name']}")
    return dict(
        kind=inspected["kind"],
        tensor_count=inspected["tensor_count"],
        model_sha256=manifest["model_sha256"],
        status="ok",
    )


def convert(args) -> dict:
    import torch

    if sys.byteorder != "little":
        raise RuntimeError("Conversion currently requires a little-endian host")
    source = args.checkpoint
    tensors = {}
    sources = []
    if args.kind == "image":
        for filename in ("vit_g.pth", "siglip_vq.pt"):
            path = source / filename
            tensors.update(
                torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            )
            sources.append(path)
    else:
        from safetensors import safe_open

        path = source / "model.safetensors"
        with safe_open(path, framework="pt", device="cpu") as weights:
            for key in weights.keys():
                name = key.removeprefix("model.encoder.").removeprefix("encoder.")
                tensors[name] = weights.get_tensor(key)
        sources.extend(
            (path, source / "config.json", source / "preprocessor_config.json")
        )
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        checks = {
            "d_model": 1280,
            "num_mel_bins": 128,
            "encoder_attention_heads": 20,
            "encoder_ffn_dim": 5120,
            "pooling_kernel_size": 4,
            "pooling_position": 16,
            "quantize_position": 16,
            "quantize_causal_block_size": 200,
            "quantize_vocab_size": 16384,
            "encoder_causal_convolution": True,
            "encoder_causal_attention": False,
            "quantize_causal_encoder": False,
            "quantize_encoder_only": True,
            "pooling_type": "avg",
            "activation_function": "gelu",
            "max_source_positions": 1500,
        }
        for name, value in checks.items():
            if config.get(name) != value:
                raise ValueError(
                    f"Unsupported audio config {name}: {config.get(name)!r}"
                )
    required = expected_shapes(args.kind)
    missing = sorted(set(required) - set(tensors))
    if missing:
        raise ValueError(f"Missing encoding tensors: {missing}")
    expected_dtype = torch.bfloat16 if args.kind == "image" else torch.float32
    for name, shape in required.items():
        tensor = tensors[name]
        if tuple(tensor.shape) != shape or tensor.dtype != expected_dtype:
            raise ValueError(
                f"Unexpected shape or dtype: {name} {tuple(tensor.shape)} {tensor.dtype}"
            )
    excluded = sorted(set(tensors) - set(required))
    tensors = {name: tensors[name] for name in sorted(required)}
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "model.otm"
    manifest_path = args.output / "manifest.json"
    if not args.force and (model_path.exists() or manifest_path.exists()):
        raise FileExistsError(
            "Output package already exists; use --force to replace it"
        )
    entries = []
    index_bytes = HEADER.size
    for name, tensor in tensors.items():
        index_bytes += RECORD.size + 8 * tensor.ndim + len(name.encode("utf-8"))
    offset = aligned(index_bytes)
    for name, tensor in tensors.items():
        nbytes = tensor.numel() * tensor.element_size()
        entries.append(
            dict(
                name=name,
                dtype=str(tensor.dtype).removeprefix("torch."),
                shape=list(tensor.shape),
                offset=offset,
                nbytes=nbytes,
            )
        )
        offset = aligned(offset + nbytes)
    partial = args.output / "model.otm.partial"
    try:
        with partial.open("wb") as stream:
            stream.write(
                HEADER.pack(
                    MAGIC,
                    1,
                    1 if args.kind == "audio" else 2,
                    *CONFIGS[args.kind],
                    len(entries),
                )
            )
            for entry in entries:
                name_bytes = entry["name"].encode("utf-8")
                stream.write(
                    RECORD.pack(
                        len(name_bytes),
                        1 if entry["dtype"] == "float32" else 2,
                        len(entry["shape"]),
                        0,
                        entry["offset"],
                        entry["nbytes"],
                    )
                )
                stream.write(
                    struct.pack("<" + "Q" * len(entry["shape"]), *entry["shape"])
                )
                stream.write(name_bytes)
            for entry in entries:
                stream.write(b"\x00" * (entry["offset"] - stream.tell()))
                tensor = tensors[entry["name"]].detach().contiguous()
                raw = tensor.view(torch.uint8).numpy()
                payload = memoryview(raw).cast("B")
                digest = hashlib.sha256()
                for start in range(0, entry["nbytes"], 8 * 1024 * 1024):
                    chunk = payload[start : start + 8 * 1024 * 1024]
                    stream.write(chunk)
                    digest.update(chunk)
                entry["sha256"] = digest.hexdigest()
            stream.flush()
            os.fsync(stream.fileno())
        inspect_model(partial)
        manifest = {
            "format": "OMNIVQ01",
            "version": 1,
            "kind": args.kind,
            "config": dict(zip(FIELDS, CONFIGS[args.kind])),
            "model_file": "model.otm",
            "model_sha256": sha256(partial),
            "precision": "source-bfloat16"
            if args.kind == "image"
            else "source-float32",
            "converter_torch_version": torch.__version__,
            "source_files": [
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in sources
            ],
            "excluded_tensors": excluded,
            "tensors": entries,
            "preprocessing": "xomni-rgb-pil-bicubic-v1"
            if args.kind == "image"
            else "whisper-mel128-segment1280-v1",
        }
        if args.reference_source:
            repo = args.reference_source
            names = (
                ("modeling_vit.py", "modeling_siglip_tokenizer.py", "config.json")
                if args.kind == "image"
                else (
                    "speech_tokenizer/modeling_whisper.py",
                    "speech_tokenizer/utils.py",
                )
            )
            manifest["reference_files"] = [
                {"name": name, "sha256": sha256(repo / name)}
                for name in names
                if (repo / name).is_file()
            ]
            result = subprocess.run(
                [
                    "git",
                    "-c",
                    f"safe.directory={repo.resolve().as_posix()}",
                    "-C",
                    str(repo),
                    "rev-parse",
                    "HEAD",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                manifest["reference_revision"] = result.stdout.strip()
        temporary_manifest = args.output / "manifest.json.partial"
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(partial, model_path)
        os.replace(temporary_manifest, manifest_path)
    finally:
        if partial.exists():
            partial.unlink()
    return check_package(args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for kind in ("image", "audio"):
        subparser = subparsers.add_parser(kind)
        subparser.set_defaults(kind=kind)
        subparser.add_argument("--checkpoint", type=Path, required=True)
        subparser.add_argument("--output", type=Path, required=True)
        subparser.add_argument("--reference-source", type=Path)
        subparser.add_argument("--force", action="store_true")
    for command in ("inspect", "check"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            result = inspect_model(args.path)
        elif args.command == "check":
            result = check_package(args.path)
        else:
            result = convert(args)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, RuntimeError, KeyError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
