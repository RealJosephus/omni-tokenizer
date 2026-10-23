"""Generate image baselines from X-Omni equations.

This is an offline validation tool. PyTorch SDPA replaces the external
flash_attn dependency; results are labelled with that substitution and are not
claimed bit-exact to a particular release of FlashAttention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from omni_tokenizer.image import prepare_image


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


class ImageReference:
    """Encoding-only functional translation of X-Omni's VQSigLIP graph."""

    def __init__(
        self,
        checkpoint: Path,
        device: str = "cuda",
        dtype: str = "bf16",
        trace_directory: Path | None = None,
        attention: str = "math",
    ):
        self.device = torch.device(device)
        self.dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        if attention not in ("math", "flash"):
            raise ValueError("attention must be 'math' or 'flash'")
        if attention == "flash" and (
            self.device.type != "cuda"
            or self.dtype != torch.bfloat16
            or not torch.backends.cuda.is_flash_attention_available()
        ):
            raise ValueError(
                "FlashAttention requires a supported CUDA PyTorch build and BF16 inputs"
            )
        self.attention = "torch-sdpa-" + attention
        self.weights = {}
        for filename in ("vit_g.pth", "siglip_vq.pt"):
            tensors = torch.load(
                checkpoint / filename, map_location="cpu", weights_only=True, mmap=True
            )
            for name, value in tensors.items():
                if name == "pos_embed" or name.startswith(
                    ("patch_embed.", "blocks.", "quant_conv.", "quantize.embedding.")
                ):
                    self.weights[name] = value.to(device=self.device, dtype=self.dtype)
        self.last_features = None
        self.last_projected = None
        self.last_normalized = None
        self.trace_directory = trace_directory
        self.convolution_probe = {}
        if trace_directory:
            trace_directory.mkdir(parents=True, exist_ok=True)

    def trace(self, name, value):
        if self.trace_directory:
            np.save(self.trace_directory / (name + ".npy"), value.float().cpu().numpy())

    def probe_convolution(self, name, inputs, convolution, weight, bias):
        if self.trace_directory:
            matrix = inputs.reshape(-1, inputs.shape[-1]).contiguous()
            fused = F.linear(matrix, weight.flatten(1), bias).reshape(convolution.shape)
            split = (F.linear(matrix, weight.flatten(1)) + bias).reshape(
                convolution.shape
            )
            self.convolution_probe[name] = {
                "fused_differences": int((convolution != fused).sum()),
                "fused_max_abs": float((convolution - fused).abs().max()),
                "split_differences": int((convolution != split).sum()),
                "split_max_abs": float((convolution - split).abs().max()),
                "elements": convolution.numel(),
            }

    def linear(self, x, prefix):
        return F.linear(
            x, self.weights[prefix + ".weight"], self.weights[prefix + ".bias"]
        )

    def norm(self, x, prefix):
        return F.layer_norm(
            x,
            (1536,),
            self.weights[prefix + ".weight"],
            self.weights[prefix + ".bias"],
            eps=1e-6,
        )

    @torch.inference_mode()
    def encode(self, pixels: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(pixels).unsqueeze(0).to(self.device, self.dtype)
        h, w = x.shape[-2] // 16, x.shape[-1] // 16
        source = x
        x = (
            F.conv2d(
                x,
                self.weights["patch_embed.proj.weight"],
                self.weights["patch_embed.proj.bias"],
                stride=16,
            )
            .flatten(2)
            .transpose(1, 2)
        )
        if self.trace_directory:
            patches = F.unfold(source, kernel_size=16, stride=16).transpose(1, 2)
            self.probe_convolution(
                "patch",
                patches,
                x,
                self.weights["patch_embed.proj.weight"],
                self.weights["patch_embed.proj.bias"],
            )
        self.trace("patch", x)
        pos = self.weights["pos_embed"]
        pos = pos[0].T.contiguous().view(1, -1, 128, 128)
        pos = F.interpolate(pos, (h, w), mode="bilinear", align_corners=False)
        pos = pos.view(-1, h * w).T.contiguous().unsqueeze(0)
        self.trace("position", pos)
        x = x + pos
        self.trace("input", x)
        backend = (
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
            if self.attention == "torch-sdpa-flash"
            else torch.nn.attention.SDPBackend.MATH
        )
        with torch.nn.attention.sdpa_kernel(backend):
            for index in range(40):
                prefix = f"blocks.{index}"
                normalized = self.norm(x, prefix + ".norm1")
                if index == 0:
                    self.trace("layer0_norm1", normalized)
                qkv = self.linear(normalized, prefix + ".attn.qkv")
                if index == 0:
                    self.trace("layer0_qkv", qkv)
                q, k, v = qkv.view(1, h * w, 3, 16, 96).permute(2, 0, 3, 1, 4).unbind(0)
                attended = (
                    F.scaled_dot_product_attention(
                        q, k, v, dropout_p=0.0, is_causal=False, scale=96**-0.5
                    )
                    .transpose(1, 2)
                    .reshape(1, h * w, 1536)
                )
                if index == 0:
                    self.trace("layer0_attention", attended)
                x = x + self.linear(attended, prefix + ".attn.proj")
                if index == 0:
                    self.trace("layer0_residual1", x)
                hidden = self.linear(
                    self.norm(x, prefix + ".norm2"), prefix + ".mlp.fc1"
                )
                if index == 0:
                    self.trace("layer0_fc1", hidden)
                hidden = F.gelu(hidden, approximate="none")
                if index == 0:
                    self.trace("layer0_gelu", hidden)
                x = x + self.linear(hidden, prefix + ".mlp.fc2")
                self.trace(f"layer{index}", x)
        self.last_features = x
        projected = (
            F.conv2d(
                x.reshape(1, h, w, 1536).permute(0, 3, 1, 2),
                self.weights["quant_conv.weight"],
                self.weights["quant_conv.bias"],
            )
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(-1, 2048)
        )
        self.probe_convolution(
            "projection",
            x.view(-1, 1536),
            projected,
            self.weights["quant_conv.weight"],
            self.weights["quant_conv.bias"],
        )
        self.trace("projected", projected)
        self.last_projected = projected
        z = F.normalize(projected, p=2, dim=-1)
        self.last_normalized = z
        self.trace("normalized", z)
        embedding = F.normalize(self.weights["quantize.embedding.weight"], p=2, dim=-1)
        distance = torch.sum(z**2, dim=1, keepdim=True) + torch.sum(embedding**2, dim=1)
        distance = distance - 2 * torch.einsum("bd,dn->bn", z, embedding.T)
        return distance.argmin(dim=1).reshape(h, w).to("cpu").numpy().astype(np.uint16)


def synthetic_image(width: int, height: int) -> Image.Image:
    """Create a deterministic spatial/color pattern without external media."""
    y, x = np.indices((height, width), dtype=np.uint32)
    values = np.stack(
        (
            (x * 7 + y * 3) % 256,
            (x // 11 * 37 + y // 17 * 23) % 256,
            ((x ^ y) * 5 + x * y % 127) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("build/validation/image"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--attention",
        choices=("math", "flash"),
        default="math",
        help="Select the attention reference explicitly; math is the native comparison profile",
    )
    parser.add_argument(
        "--sizes", nargs="+", default=["384x384", "576x384", "1152x1152"]
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--trace",
        type=Path,
        help="Save intermediate tensors and compare convolution/GEMM bias rounding",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    started = time.perf_counter()
    reference = ImageReference(
        args.checkpoint, args.device, args.dtype, args.trace, args.attention
    )
    if reference.device.type == "cuda":
        torch.cuda.synchronize()
    report = {
        "profile": f"reference-{args.dtype}-{reference.attention}",
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": str(reference.device),
        "device_name": torch.cuda.get_device_name()
        if reference.device.type == "cuda"
        else platform.processor(),
        "load_seconds": time.perf_counter() - started,
        "attention_substitution": "PyTorch SDPA replaces external flash_attn_func; cross-library bit equality is unverified",
        "tracing_enabled": bool(args.trace),
        "sources": {
            name: sha256(args.checkpoint / name)
            for name in ("vit_g.pth", "siglip_vq.pt")
        },
        "cases": [],
    }
    for size in args.sizes:
        width, height = map(int, size.split("x"))
        source = args.output / f"pattern-{size}.png"
        synthetic_image(width, height).save(source)
        prepared = prepare_image(source)
        if reference.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        codes = reference.encode(prepared.pixels)
        if reference.device.type == "cuda":
            torch.cuda.synchronize()
        first_seconds = time.perf_counter() - started
        timings = []
        repeat_flips = []
        for _ in range(args.repeat):
            started = time.perf_counter()
            repeated = reference.encode(prepared.pixels)
            if reference.device.type == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
            repeat_flips.append(int(np.count_nonzero(codes != repeated)))
        np.savez(
            args.output / f"pattern-{size}-{args.dtype}.npz",
            pixels=prepared.pixels,
            codes=codes,
            features=reference.last_features.float().cpu().numpy(),
            projected=reference.last_projected.float().cpu().numpy(),
            normalized=reference.last_normalized.float().cpu().numpy(),
        )
        case = {
            "input": source.name,
            "input_sha256": sha256(source),
            "original_size": prepared.original_size,
            "processed_size": prepared.processed_size,
            "grid": prepared.grid,
            "count": int(codes.size),
            "codes_sha256": hashlib.sha256(codes.astype("<u2").tobytes()).hexdigest(),
            "first_seconds": first_seconds,
            "steady_seconds": timings,
            "median_seconds": statistics.median(timings),
            "repeat_code_flips": repeat_flips,
        }
        if reference.device.type == "cuda":
            case.update(
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
        report["cases"].append(case)
        if args.trace:
            report["convolution_probe"] = reference.convolution_probe
        (args.output / f"report-{args.dtype}.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(case), flush=True)


if __name__ == "__main__":
    main()
