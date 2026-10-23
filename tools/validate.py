"""Validate native outputs or evaluate an offline audio INT8 candidate.

The native command requires NumPy and the built library only. The audio-int8
command additionally requires the offline reference dependencies. Quantization
is simulated by replacing selected FP32 weights with quantized/dequantized
values; it neither writes compressed weights nor measures an INT8 kernel.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1048576):
            digest.update(data)
    return digest.hexdigest()


def process_memory() -> dict:
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak_working", "working", "peak_paged", "paged", "peak_nonpaged", "nonpaged",
                    "pagefile", "peak_pagefile", "private",
                )
            ]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return {"resident_bytes": counters.working, "process_peak_resident_bytes": counters.peak_working, "private_bytes": counters.private}
        return {}
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = {"process_peak_resident_bytes": int(peak if sys.platform == "darwin" else peak * 1024)}
        if sys.platform.startswith("linux"):
            result["resident_bytes"] = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        return result
    except (ImportError, OSError, ValueError, IndexError):
        return {}


class MemorySampler:
    def __init__(self, sample_gpu=False, gpu_index=0):
        self.sample_gpu = sample_gpu and shutil.which("nvidia-smi") is not None
        self.gpu_index = gpu_index
        self.stop = threading.Event()
        self.peak = {}
        self.gpu_samples = []
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        next_gpu = 0.0
        while not self.stop.is_set():
            for key, value in process_memory().items():
                self.peak[key] = max(self.peak.get(key, 0), value)
            now = time.monotonic()
            if self.sample_gpu and now >= next_gpu:
                try:
                    result = subprocess.run(
                        [shutil.which("nvidia-smi"), f"--id={self.gpu_index}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=3, check=False,
                    )
                    self.gpu_samples.append(int(result.stdout.strip()) * 1048576)
                except (ValueError, OSError, subprocess.TimeoutExpired):
                    pass
                next_gpu = now + 0.25
            self.stop.wait(0.02)

    def __enter__(self):
        self.initial = process_memory()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=4)

    def report(self):
        result = {"process_before": self.initial, "process_sampled_peak": self.peak, "process_after": process_memory()}
        if self.gpu_samples:
            result.update(
                gpu_global_peak_used_bytes=max(self.gpu_samples), gpu_global_first_used_bytes=self.gpu_samples[0],
                gpu_samples=len(self.gpu_samples), gpu_physical_index=self.gpu_index,
                gpu_measurement_note="nvidia-smi samples of the whole physical device, including other processes; short peaks can be missed",
            )
        return result


def feature_error(actual, expected) -> dict:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1, actual.shape[-1])
    expected = np.asarray(expected, dtype=np.float64).reshape(-1, expected.shape[-1])
    if actual.shape != expected.shape:
        return {"shape_mismatch": [list(actual.shape), list(expected.shape)]}
    difference = actual - expected
    norms = np.linalg.norm(actual, axis=1) * np.linalg.norm(expected, axis=1)
    cosines = np.sum(actual * expected, axis=1) / np.maximum(norms, 1e-30)
    return {
        "max_abs_error": float(np.max(np.abs(difference))),
        "rms_error": float(np.sqrt(np.mean(difference ** 2))),
        "relative_l2_error": float(np.linalg.norm(difference) / max(np.linalg.norm(expected), 1e-30)),
        "mean_vector_l2_error": float(np.linalg.norm(difference, axis=1).mean()),
        "mean_cosine_similarity": float(cosines.mean()),
    }


def fixture_paths(patterns) -> list[Path]:
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(pattern)))
        if not matches:
            raise ValueError(f"No fixture matches {pattern}")
        paths.extend(Path(path) for path in matches)
    return list(dict.fromkeys(paths))


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def native(args):
    from omni_tokenizer.native import Library, Model
    fixtures = fixture_paths(args.fixtures)
    report = {"mode": "native", "platform": platform.platform(), "numpy": np.__version__,
              "device": args.device, "repeats": args.repeats, "cases": [],
              "timing_note": "encode calls synchronize in the native ABI; features and fixture I/O are outside timing"}
    memory = MemorySampler(args.sample_gpu, args.gpu_index)
    with memory:
        library = Library(args.library)
        report["library"] = str(library.path)
        report["library_sha256"] = sha256(library.path)
        started = time.perf_counter()
        with Model(library, args.model, args.device) as model:
            report["load_seconds"] = time.perf_counter() - started
            report["model"] = str(model.path)
            report["model_sha256"] = sha256(model.path)
            report["static_weight_bytes"] = model.info.weight_bytes
            report["kind"] = "audio" if model.info.kind == 1 else "image"
            with model.context() as context:
                for path in fixtures:
                    with np.load(path, allow_pickle=False) as data:
                        expected = data["codes"].reshape(-1)
                        if model.info.kind == 1:
                            values = data["mel"]
                            valid_frames = int(data["valid_frames"])
                            encode = lambda: context.encode_audio(values, valid_frames)
                            feature_key = "pooled"
                        else:
                            values = data["pixels"]
                            encode = lambda: context.encode_image(values)
                            feature_key = "normalized"
                        if feature_key not in data and "features" in data and data["features"].shape[-1] == model.info.code_dim:
                            feature_key = "features"
                        started = time.perf_counter()
                        actual = encode().reshape(-1)
                        first_seconds = time.perf_counter() - started
                        if actual.shape != expected.shape:
                            raise ValueError(f"Code shape mismatch for {path}: {actual.shape} != {expected.shape}")
                        errors = feature_error(context.features(actual.size), data[feature_key]) if feature_key in data else None
                        timings, repeat_flips = [], []
                        for _ in range(args.repeats):
                            started = time.perf_counter()
                            repeated = encode().reshape(-1)
                            timings.append(time.perf_counter() - started)
                            repeat_flips.append(int(np.count_nonzero(actual != repeated)))
                        record = {"fixture": str(path), "fixture_sha256": sha256(path),
                                  "code_count": actual.size, "code_flips": int(np.count_nonzero(actual != expected)),
                                  "code_flip_rate": float(np.mean(actual != expected)),
                                  "first_seconds": first_seconds, "steady_seconds": timings,
                                  "p50_seconds": float(np.percentile(timings, 50)), "p95_seconds": float(np.percentile(timings, 95)),
                                  "repeat_code_flips": repeat_flips, "feature_error": errors,
                                  "native_peak_workspace_bytes": context.peak_workspace_bytes,
                                  "workspace_note": "Native allocation tracker; excludes resident model and CUDA library/driver overhead"}
                        report["cases"].append(record)
                        print(json.dumps(record), flush=True)
                        write_report(args.output, report)
    report["memory"] = memory.report()
    report["total_codes"] = sum(case["code_count"] for case in report["cases"])
    report["total_code_flips"] = sum(case["code_flips"] for case in report["cases"])
    write_report(args.output, report)
    if args.max_flip_rate is not None and any(case["code_flip_rate"] > args.max_flip_rate for case in report["cases"]):
        return 1
    return 0


def audio_int8(args):
    import torch
    from reference_audio import AudioReference
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = AudioReference(args.checkpoint, args.reference_root, args.device)
    codebook = reference.model.codebook.weight
    fixtures = fixture_paths(args.fixtures)
    baseline = []
    report = {"mode": "offline-audio-int8-experiment", "platform": platform.platform(), "torch": torch.__version__,
              "device": args.device, "checkpoint_sha256": sha256(args.checkpoint / "model.safetensors"),
              "candidate": "Transformer linear weights only, symmetric per-output-channel INT8 [-127,127] with FP32 scale",
              "runtime": "Weights immediately dequantized to FP32; all inference remains FP32 eager PyTorch",
              "decision": "Experimental only; default unchanged, no acceptance threshold supplied",
              "timing_note": "These are FP32 quantize/dequantize simulation timings, not compressed native INT8 speed",
              "cases": []}

    def measured(mel, valid_frames):
        if args.device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        codes = reference.encode(mel, valid_frames)[0]
        if args.device == "cuda":
            torch.cuda.synchronize()
        return codes, reference.last_pooled[0].copy(), time.perf_counter() - started

    with torch.inference_mode():
        # Warm up only the shortest supplied fixture.
        with np.load(fixtures[0], allow_pickle=False) as data:
            reference.encode(data["mel"], int(data["valid_frames"]))
        for path in fixtures:
            with np.load(path, allow_pickle=False) as data:
                mel = data["mel"]
                valid = int(data["valid_frames"])
            codes, features, seconds = measured(mel, valid)
            count = (valid + 7) // 8
            baseline.append((path, codes[:count], features[:count], seconds))
        total_original = 0
        compressed = 0
        quantized_count = 0
        quantized_elements = 0
        weight_squared_error = 0.0
        for name, parameter in reference.model.named_parameters():
            if name == "embed_positions2.weight":
                continue
            total_original += parameter.numel() * parameter.element_size()
            if name.startswith("layers.") and name.endswith(".weight") and parameter.ndim == 2:
                scale = parameter.abs().amax(dim=1, keepdim=True) / 127.0
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)
                dequantized = (parameter / scale).round().clamp(-127, 127) * scale
                weight_squared_error += float(((parameter - dequantized).double() ** 2).sum())
                compressed += parameter.numel() + parameter.shape[0] * 4
                quantized_elements += parameter.numel()
                quantized_count += 1
                parameter.copy_(dequantized)
            else:
                compressed += parameter.numel() * parameter.element_size()
        report.update(original_static_weight_bytes=total_original, estimated_int8_static_weight_bytes=compressed,
                      quantized_tensor_count=quantized_count, quantized_elements=quantized_elements,
                      quantized_weight_rms_error=(weight_squared_error / quantized_elements) ** 0.5,
                      static_note="Encoding weights only; excludes EMA, postquant positions, allocator/layout/workspace overhead")
        del scale, dequantized
        for path, expected, expected_features, baseline_seconds in baseline:
            with np.load(path, allow_pickle=False) as data:
                mel, valid = data["mel"], int(data["valid_frames"])
            timings = []
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for _ in range(args.repeats):
                actual, actual_features, seconds = measured(mel, valid)
                timings.append(seconds)
            inference_peak = torch.cuda.max_memory_allocated() if args.device == "cuda" else None
            actual = actual[:expected.size]
            actual_features = actual_features[:expected.size]
            features_t = torch.from_numpy(expected_features).to(reference.device)
            expected_embedding = codebook[torch.from_numpy(expected.astype(np.int64)).to(reference.device)]
            actual_embedding = codebook[torch.from_numpy(actual.astype(np.int64)).to(reference.device)]
            expected_distortion = ((features_t - expected_embedding) ** 2).sum(-1)
            changed_distortion = ((features_t - actual_embedding) ** 2).sum(-1)
            distances = torch.addmm((features_t ** 2).sum(-1, keepdim=True) + (codebook ** 2).sum(-1), features_t, codebook.T, alpha=-2)
            closest = distances.topk(2, largest=False, dim=-1).values
            margin = closest[:, 1] - closest[:, 0]
            record = {"fixture": str(path), "code_count": int(actual.size),
                      "code_flips": int(np.count_nonzero(actual != expected)), "code_flip_rate": float(np.mean(actual != expected)),
                      "feature_error": feature_error(actual_features, expected_features),
                      "baseline_fp32_seconds": baseline_seconds, "candidate_qdeq_fp32_seconds": timings,
                      "candidate_p50_seconds": float(np.percentile(timings, 50)),
                      "mean_baseline_quantization_distortion": float(expected_distortion.mean()),
                      "mean_candidate_code_distortion_on_baseline_features": float(changed_distortion.mean()),
                      "mean_nearest_neighbor_margin": float(margin.mean()), "minimum_nearest_neighbor_margin": float(margin.min()),
                      "cuda_peak_allocated_bytes": inference_peak}
            report["cases"].append(record)
            print(json.dumps(record), flush=True)
            write_report(args.output, report)
            del features_t, expected_embedding, actual_embedding, expected_distortion, changed_distortion, distances, closest, margin
    report["total_codes"] = sum(case["code_count"] for case in report["cases"])
    report["total_code_flips"] = sum(case["code_flips"] for case in report["cases"])
    write_report(args.output, report)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    native_parser = commands.add_parser("native", help="Compare native codes and features with reference NPZ fixtures")
    native_parser.add_argument("--library", type=Path)
    native_parser.add_argument("--model", type=Path, required=True)
    native_parser.add_argument("--device", default="cpu")
    native_parser.add_argument("--fixtures", nargs="+", required=True)
    native_parser.add_argument("--output", type=Path, required=True)
    native_parser.add_argument("--repeats", type=int, default=3)
    native_parser.add_argument("--max-flip-rate", type=float)
    native_parser.add_argument("--sample-gpu", action="store_true")
    native_parser.add_argument("--gpu-index", type=int, default=0)
    native_parser.set_defaults(handler=native)
    quant = commands.add_parser("audio-int8", help="Measure offline INT8 quantize/dequantize errors without changing native defaults")
    quant.add_argument("--checkpoint", type=Path, required=True)
    quant.add_argument("--reference-root", type=Path, required=True)
    quant.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    quant.add_argument("--fixtures", nargs="+", required=True)
    quant.add_argument("--output", type=Path, required=True)
    quant.add_argument("--repeats", type=int, default=3)
    quant.add_argument("--threads", type=int, default=4)
    quant.set_defaults(handler=audio_int8)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if getattr(args, "max_flip_rate", None) is not None and not 0 <= args.max_flip_rate <= 1:
        parser.error("--max-flip-rate must be between 0 and 1")
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
