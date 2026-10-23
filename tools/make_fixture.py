"""Generate small deterministic models and independent NumPy reference outputs.

All weights are synthetic. The generated fixtures are safe for public CPU CI
and exercise both complete encoding graphs without downloading checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

try:
    from .convert import FIELDS, HEADER, MAGIC, RECORD, aligned, inspect_model, sha256
except ImportError:
    from convert import FIELDS, HEADER, MAGIC, RECORD, aligned, inspect_model, sha256


def bf16(value):
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    return (
        (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    ).view(np.float32)


def write_model(path, kind, config, weights):
    path.mkdir(parents=True, exist_ok=True)
    entries = []
    offset = aligned(
        HEADER.size
        + sum(
            RECORD.size + 8 * value.ndim + len(name.encode())
            for name, value in weights.items()
        )
    )
    for name, value in weights.items():
        payload = (
            (bf16(value).view(np.uint32) >> 16).astype("<u2").tobytes()
            if kind == "image"
            else value.astype("<f4").tobytes()
        )
        entries.append(
            dict(
                name=name,
                dtype="bfloat16" if kind == "image" else "float32",
                shape=list(value.shape),
                offset=offset,
                nbytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
        offset = aligned(offset + len(payload))
    with (path / "model.otm").open("wb") as stream:
        stream.write(
            HEADER.pack(MAGIC, 1, 2 if kind == "image" else 1, *config, len(entries))
        )
        for entry in entries:
            name = entry["name"].encode()
            stream.write(
                RECORD.pack(
                    len(name),
                    2 if kind == "image" else 1,
                    len(entry["shape"]),
                    0,
                    entry["offset"],
                    entry["nbytes"],
                )
            )
            stream.write(struct.pack("<" + "Q" * len(entry["shape"]), *entry["shape"]))
            stream.write(name)
        for entry, value in zip(entries, weights.values()):
            stream.write(b"\x00" * (entry["offset"] - stream.tell()))
            payload = (
                (bf16(value).view(np.uint32) >> 16).astype("<u2").tobytes()
                if kind == "image"
                else value.astype("<f4").tobytes()
            )
            stream.write(payload)
    inspect_model(path)
    manifest = dict(
        format="OMNIVQ01",
        version=1,
        kind=kind,
        config=dict(zip(FIELDS, config)),
        model_sha256=sha256(path / "model.otm"),
        tensors=entries,
        synthetic=True,
    )
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


class Reference:
    def __init__(self, weights, image=False):
        self.weights = weights
        self.image = image

    def rounded(self, value):
        return bf16(value) if self.image else np.asarray(value, np.float32)

    def linear(self, x, name):
        weight = self.weights[name + ".weight"].reshape(
            self.weights[name + ".weight"].shape[0], -1
        )
        product = x @ weight.T
        if self.image and name in ("patch_embed.proj", "quant_conv"):
            product = bf16(product)
        return self.rounded(product + self.weights.get(name + ".bias", np.float32(0)))

    def norm(self, x, name):
        mean = x.mean(-1, keepdims=True)
        variance = ((x - mean) ** 2).mean(-1, keepdims=True)
        eps = np.float32(1e-6 if self.image else 1e-5)
        value = (x - mean) / np.sqrt(variance + eps)
        return self.rounded(
            value * self.weights[name + ".weight"] + self.weights[name + ".bias"]
        )

    def gelu(self, x):
        erf = np.array(
            [math.erf(float(value) / math.sqrt(2)) for value in x.ravel()], np.float32
        ).reshape(x.shape)
        return self.rounded(np.float32(0.5) * x * (np.float32(1) + erf))

    def attention(self, q, k, v, valid, block=0):
        length, width = q.shape
        heads, dimension = 2, width // 2
        output = np.empty_like(q)
        for query in range(length):
            limit = min(
                valid, length, (query // block + 1) * block if block else length
            )
            for head in range(heads):
                sl = slice(head * dimension, (head + 1) * dimension)
                logits = (q[query, sl] @ k[:limit, sl].T) / np.float32(
                    math.sqrt(dimension)
                )
                weights = np.exp(logits - logits.max())
                weights /= weights.sum()
                output[query, sl] = weights @ v[:limit, sl]
        return self.rounded(output)

    def quantize(self, x, codebook):
        if self.image:

            def normalize(value):
                norm = bf16(np.sqrt((value**2).sum(-1, keepdims=True)))
                return bf16(value / np.maximum(norm, bf16(np.float32(1e-12))))

            x, codebook = normalize(x), normalize(codebook)
            xn = bf16(bf16(x * x).sum(-1, keepdims=True))
            cn = bf16(bf16(codebook * codebook).sum(-1))
            distance = bf16(bf16(xn + cn) - bf16(2 * bf16(x @ codebook.T)))
        else:
            distance = (
                (x * x).sum(-1, keepdims=True)
                + (codebook * codebook).sum(-1)
                - 2 * (x @ codebook.T)
            )
        return distance.argmin(-1).astype(np.uint16), x

    def audio(self, mel, valid):
        x = mel.T
        for name, stride in (("conv1", 1), ("conv2", 2)):
            padded = np.pad(x, ((2, 0), (0, 0)))
            gathered = np.stack(
                [padded[t : t + 3].T.ravel() for t in range(0, x.shape[0], stride)]
            )
            x = self.gelu(self.linear(gathered, name))
        x += self.weights["embed_positions.weight"][: x.shape[0]]
        for index in range(2):
            pre = f"layers.{index}."
            norm = self.norm(x, pre + "self_attn_layer_norm")
            q, k, v = [
                self.linear(norm, pre + "self_attn." + part + "_proj")
                for part in ("q", "k", "v")
            ]
            x += self.linear(
                self.attention(q, k, v, (valid + 1) // 2, block=4),
                pre + "self_attn.out_proj",
            )
            x += self.linear(
                self.gelu(
                    self.linear(self.norm(x, pre + "final_layer_norm"), pre + "fc1")
                ),
                pre + "fc2",
            )
        pooled = x.reshape(-1, 4, x.shape[-1]).mean(1)[: (valid + 7) // 8]
        return self.quantize(pooled, self.weights["codebook.weight"])

    def image_codes(self, pixels):
        _, height, width = pixels.shape
        gh, gw = height // 16, width // 16
        patches = (
            pixels.reshape(3, gh, 16, gw, 16)
            .transpose(1, 3, 0, 2, 4)
            .reshape(gh * gw, -1)
        )
        x = self.linear(bf16(patches), "patch_embed.proj")
        pos = self.weights["pos_embed"].reshape(4, 4, 4)
        interpolated = np.empty((gh, gw, 4), np.float32)
        for y in range(gh):
            sy = max(0, (y + 0.5) * 4 / gh - 0.5)
            iy, fy = int(sy), np.float32(sy - int(sy))
            for xx in range(gw):
                sx = max(0, (xx + 0.5) * 4 / gw - 0.5)
                ix, fx = int(sx), np.float32(sx - int(sx))
                top = (1 - fx) * pos[iy, ix] + fx * pos[iy, min(ix + 1, 3)]
                bottom = (1 - fx) * pos[min(iy + 1, 3), ix] + fx * pos[
                    min(iy + 1, 3), min(ix + 1, 3)
                ]
                interpolated[y, xx] = (1 - fy) * top + fy * bottom
        x = bf16(x + bf16(interpolated.reshape(-1, 4)))
        for index in range(2):
            pre = f"blocks.{index}."
            q, k, v = np.split(
                self.linear(self.norm(x, pre + "norm1"), pre + "attn.qkv"), 3, axis=-1
            )
            x = bf16(
                x + self.linear(self.attention(q, k, v, len(x)), pre + "attn.proj")
            )
            x = bf16(
                x
                + self.linear(
                    self.gelu(
                        self.linear(self.norm(x, pre + "norm2"), pre + "mlp.fc1")
                    ),
                    pre + "mlp.fc2",
                )
            )
        return self.quantize(
            self.linear(x, "quant_conv"), self.weights["quantize.embedding.weight"]
        )


def generate(output: Path) -> dict:
    generator = np.random.default_rng(617)
    report = {}
    for kind in ("audio", "image"):
        weights = {}

        def random(shape, scale=0.25):
            return (generator.standard_normal(shape) * scale).astype(np.float32)

        def linear(name, output_width, input_width, bias=True, shape=None):
            weights[name + ".weight"] = random(
                shape or (output_width, input_width), 0.6 / math.sqrt(input_width)
            )
            if bias:
                weights[name + ".bias"] = random((output_width,), 0.05)

        def norm(name):
            weights[name + ".weight"] = np.ones(4, np.float32) + random((4,), 0.05)
            weights[name + ".bias"] = random((4,), 0.05)

        if kind == "audio":
            config = (2, 4, 2, 8, 128, 0, 4, 4, 16, 1, 4, 16)
            linear("conv1", 4, 384, shape=(4, 128, 3))
            linear("conv2", 4, 12, shape=(4, 4, 3))
            weights["embed_positions.weight"] = random((16, 4), 0.8)
            for index in range(2):
                pre = f"layers.{index}."
                norm(pre + "self_attn_layer_norm")
                norm(pre + "final_layer_norm")
                for part in ("q", "k", "v", "out"):
                    linear(pre + "self_attn." + part + "_proj", 4, 4, bias=part != "k")
                linear(pre + "fc1", 8, 4)
                linear(pre + "fc2", 4, 8)
            weights["codebook.weight"] = random((16, 4), 0.8)
            inputs = random((128, 16), 0.8)
            reference = Reference(weights)
            codes, features = reference.audio(inputs, 9)
            full_codes, full_features = reference.audio(inputs, 16)
            expected = dict(
                mel=inputs,
                valid_frames=np.int64(9),
                codes=codes,
                features=features,
                full_codes=full_codes,
                full_features=full_features,
            )
        else:
            config = (2, 4, 2, 8, 3, 16, 0, 0, 4, 4, 6, 16)
            linear("patch_embed.proj", 4, 768, shape=(4, 3, 16, 16))
            weights["patch_embed.proj.weight"] *= np.float32(3)
            weights["pos_embed"] = random((1, 16, 4), 0.3)
            for index in range(2):
                pre = f"blocks.{index}."
                norm(pre + "norm1")
                norm(pre + "norm2")
                linear(pre + "attn.qkv", 12, 4)
                linear(pre + "attn.proj", 4, 4)
                linear(pre + "mlp.fc1", 8, 4)
                linear(pre + "mlp.fc2", 4, 8)
            linear("quant_conv", 6, 4, shape=(6, 4, 1, 1))
            weights["quantize.embedding.weight"] = random((16, 6), 0.8)
            weights = {name: bf16(value) for name, value in weights.items()}
            inputs = random((3, 32, 48), 0.5)
            codes, features = Reference(weights, image=True).image_codes(inputs)
            expected = dict(pixels=inputs, codes=codes, features=features)
        destination = output / kind
        write_model(destination, kind, config, weights)
        np.savez(destination / "expected.npz", **expected)
        report[kind] = dict(
            codes=codes.tolist(),
            input_shape=list(inputs.shape),
            model_bytes=(destination / "model.otm").stat().st_size,
        )
    (output / "expected.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/fixtures"))
    args = parser.parse_args()
    print(json.dumps(generate(args.output), indent=2))


if __name__ == "__main__":
    main()
