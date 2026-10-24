"""Validate native image encoding against unmodified X-Omni on supplied images.

The reference imports X-Omni's tokenizer and its external FlashAttention path.
PyTorch, timm, einops, torchvision and flash-attn are offline validation
dependencies; they are not runtime dependencies of omni-tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import statistics
import sys
import time
import types
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from omni_tokenizer.image import prepare_image
from omni_tokenizer.native import Library, Model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ImageReference:
    """Load the upstream implementation without changing its operators or flags."""

    def __init__(self, checkpoint: Path, x_omni: Path, device: str = "cuda"):
        source = x_omni.resolve()
        for name in ("modeling_siglip_tokenizer.py", "modeling_vit.py"):
            if not (source / name).is_file():
                raise ValueError(f"X-Omni source file is missing: {source / name}")
        # Give the two upstream modules a package for their relative import,
        # without executing unrelated language-model package initialization.
        package_name = (
            "_omni_reference_" + hashlib.sha256(str(source).encode()).hexdigest()[:12]
        )
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [str(source)]
            sys.modules[package_name] = package
        upstream = importlib.import_module(package_name + ".modeling_siglip_tokenizer")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("The original BF16 FlashAttention reference requires CUDA")
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            self.model = upstream.SiglipTokenizer(
                "siglip2_giant_patch16_384",
                str(checkpoint / "vit_g.pth"),
                str(checkpoint / "siglip_vq.pt"),
            )
        finally:
            torch.set_default_dtype(previous_dtype)
        self.model = self.model.eval().to(self.device, torch.bfloat16)
        self.preprocess = upstream.create_anyres_preprocess()
        self.last_projected = None
        self.model.vqproj.quantize.register_forward_pre_hook(self._capture_projection)
        self.sources = {
            name: sha256(source / name)
            for name in ("modeling_siglip_tokenizer.py", "modeling_vit.py")
        }

    def _capture_projection(self, module, args):
        self.last_projected = args[0]

    @torch.inference_mode()
    def encode(self, pixels: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(pixels).unsqueeze(0).to(self.device, torch.bfloat16)
        return self.model.encode(tensor).cpu().numpy().astype(np.uint16).ravel()

    @torch.inference_mode()
    def normalized_features(self) -> np.ndarray:
        if self.last_projected is None:
            raise RuntimeError("Encode an image before requesting features")
        z = self.last_projected.permute(0, 2, 3, 1).contiguous()
        z = z.reshape(-1, z.shape[-1])
        return F.normalize(z, p=2, dim=-1).float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--x-omni", type=Path, required=True, help="Unmodified X-Omni source checkout"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--images", type=Path, nargs="+", help="Real image files to validate"
    )
    inputs.add_argument(
        "--manifest",
        type=Path,
        help="JSON with images entries containing relative file paths",
    )
    parser.add_argument(
        "--limit", type=int, help="Validate only the first N supplied images"
    )
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--native-model", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--output", type=Path, default=Path("build/validation/image-fa2/report.json")
    )
    args = parser.parse_args()
    if args.repeat < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--repeat and --limit must be positive")
    if bool(args.native_library) != bool(args.native_model):
        parser.error("--native-library and --native-model must be specified together")
    if args.manifest:
        entries = json.loads(args.manifest.read_text(encoding="utf-8"))["images"]
        paths = [args.manifest.parent / item["file"] for item in entries]
    else:
        paths = args.images
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        parser.error("The image set is empty")
    reference = ImageReference(args.checkpoint, args.x_omni, args.device)
    report = {
        "reference": "Unmodified X-Omni BF16 with external FlashAttention",
        "source_sha256": reference.sources,
        "timing": "Prepared CPU FP32 pixels to CPU uint16 codes; excludes preprocessing, loading and feature comparison.",
        "images": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        context = None
        if args.native_library:
            native = stack.enter_context(
                Model(Library(args.native_library), args.native_model, args.device)
            )
            context = stack.enter_context(native.context())
        for index, path in enumerate(paths):
            with Image.open(path) as image:
                pixels = reference.preprocess(image)
            if pixels is None:
                raise ValueError(
                    f"Image is rejected by X-Omni's preprocessing filter: {path}"
                )
            prepared = prepare_image(path)
            np.testing.assert_array_equal(pixels.numpy(), prepared.pixels)
            expected = reference.encode(prepared.pixels)
            target = reference.normalized_features()
            record = {
                "input": path.name,
                "input_sha256": sha256(path),
                "shape": list(prepared.pixels.shape[-2:]),
                "codes": int(expected.size),
                "reference_codes_sha256": hashlib.sha256(
                    expected.astype("<u2").tobytes()
                ).hexdigest(),
            }
            if context:
                actual = context.encode_image(prepared.pixels)
                features = context.features(len(actual))
                if not np.isfinite(features).all():
                    raise ValueError(f"Native features are non-finite: {path}")
                delta = features.astype(np.float64) - target
                record.update(
                    changed_codes=int(np.count_nonzero(actual != expected)),
                    changed_features=int(np.count_nonzero(delta)),
                    max_abs=float(np.abs(delta).max()),
                    rms=float(np.sqrt(np.mean(delta * delta))),
                )
            # Correctness runs above warm the shape before paired timing.
            times = {"reference": [], **({"native": []} if context else {})}
            calls = {"reference": reference.encode}
            if context:
                calls["native"] = context.encode_image
            for repeat in range(args.repeat):
                order = list(calls)
                if repeat % 2:
                    order.reverse()
                for name in order:
                    started = time.perf_counter()
                    repeated = calls[name](prepared.pixels)
                    times[name].append(time.perf_counter() - started)
                    first = expected if name == "reference" else actual
                    if not np.array_equal(repeated, first):
                        raise RuntimeError(f"{name} output is not repeatable: {path}")
            record["median_seconds"] = {
                name: statistics.median(values) for name, values in times.items()
            }
            report["images"].append(record)
            if (index + 1) % 50 == 0 or index + 1 == len(paths):
                report["total_codes"] = sum(item["codes"] for item in report["images"])
                if context:
                    report["changed_codes"] = sum(
                        item["changed_codes"] for item in report["images"]
                    )
                    report["changed_feature_images"] = sum(
                        item["changed_features"] > 0 for item in report["images"]
                    )
                report["complete"] = index + 1 == len(paths)
                args.output.write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
                print(
                    json.dumps(
                        {key: value for key, value in report.items() if key != "images"}
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
