"""Compare real speech with the unmodified GLM-4-Voice encoder.

Uses upstream loading/attention/precision defaults and singleton segment
padding. Reports model differences on identical mel inputs separately from
the complete decode/resample/mel/encoder comparison. No data is downloaded.
A JSON manifest contains an "audio" list of {"file": "relative/path", ...};
optional "sha256" values are verified. Token agreement is not an ASR score.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
import transformers
from transformers import WhisperFeatureExtractor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from omni_tokenizer.audio import SEGMENT_SAMPLES, iter_audio_segments, log_mel
from omni_tokenizer.native import Library, Model
from reference_audio import AudioReference, sha256
from validate import feature_error, write_report


def compare_codes(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(f"Code count mismatch: {actual.shape} != {expected.shape}")
    positions = np.flatnonzero(actual != expected)
    return {
        "count": int(actual.size),
        "different": int(positions.size),
        "positions": positions.tolist(),
        "actual": actual[positions].tolist(),
        "reference": expected[positions].tolist(),
    }


def summary(cases):
    result = {"segments": len(cases)}
    comparisons = ["same_mel", "end_to_end", "reference_preprocessing"]
    if cases and "fp64_same_mel" in cases[0]:
        comparisons.append("fp64_same_mel")
    for key in comparisons:
        count = sum(c[key]["count"] for c in cases)
        different = sum(c[key]["different"] for c in cases)
        result[key] = {
            "count": count,
            "different": different,
            "different_segments": sum(c[key]["different"] > 0 for c in cases),
            "rate": different / count if count else 0.0,
        }
    result["max_feature_relative_l2"] = max(
        (c["pooled_features"]["relative_l2_error"] for c in cases), default=0.0
    )
    result["max_pcm_abs_error"] = max(
        (c["pcm_max_abs_error"] for c in cases), default=0.0
    )
    result["max_mel_abs_error"] = max(
        (c["mel_max_abs_error"] for c in cases), default=0.0
    )
    if cases and "fp64_pooled_features" in cases[0]:
        result["max_fp64_feature_relative_l2"] = max(
            c["fp64_pooled_features"]["relative_l2_error"] for c in cases
        )
    return result


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--native-model", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", type=Path)
    inputs.add_argument("--audio", type=Path, nargs="+")
    parser.add_argument(
        "--output", type=Path, default=Path("build/validation/real-audio")
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--benchmark-cases", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--save-mismatches", action="store_true")
    parser.add_argument(
        "--fp64-reference",
        action="store_true",
        help="Also run the original model in FP64 for numerical diagnosis",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.benchmark_cases < 0 or args.repeats < 1:
        parser.error("--benchmark-cases must be nonnegative and --repeats positive")
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        records = manifest["audio"]
        root = args.manifest.resolve().parent
    else:
        records = [{"file": str(p.resolve())} for p in args.audio]
        root = Path.cwd()
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        parser.error("No audio inputs")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    reference = AudioReference(args.checkpoint, args.reference_root, args.device)
    precise_reference = None
    if args.fp64_reference:
        precise_reference = AudioReference(
            args.checkpoint, args.reference_root, args.device
        )
        precise_reference.model.double()
    extractor = WhisperFeatureExtractor.from_pretrained(
        str(args.checkpoint), local_files_only=True
    )
    report = {
        "scope": "Real speech; numerical compatibility, not downstream quality",
        "padding": "Singleton segments, independently padded to 1280 samples",
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "device": args.device,
        "execution": reference.execution_settings(),
        "fp64_execution": (
            precise_reference.execution_settings() if precise_reference else None
        ),
        "checkpoint_sha256": sha256(args.checkpoint / "model.safetensors"),
        "reference_source_sha256": sha256(
            args.reference_root / "speech_tokenizer/modeling_whisper.py"
        ),
        "native_library_sha256": sha256(args.native_library),
        "manifest_sha256": sha256(args.manifest) if args.manifest else None,
        "files": [],
        "cases": [],
        "benchmark": [],
    }
    benchmark_inputs = []
    begin = time.perf_counter()
    with (
        Model(Library(args.native_library), args.native_model, args.device) as model,
        model.context() as context,
    ):
        for file_index, record in enumerate(records):
            path = root / record["file"]
            digest = sha256(path)
            if record.get("sha256", digest) != digest:
                raise ValueError(f"Input hash mismatch: {path}")
            waveform, rate = torchaudio.load(str(path))
            channels = waveform.shape[0]
            if rate != 16000:
                waveform = (
                    torchaudio.transforms.Resample(rate, 16000)
                    .to(args.device)(waveform.to(args.device))
                    .cpu()
                )
            pcm = waveform[0].numpy()
            file_record = {
                "file": record["file"],
                "dataset": record.get("dataset"),
                "sha256": digest,
                "source_rate": rate,
                "channels": channels,
                "samples_16k": int(pcm.size),
            }
            report["files"].append(file_record)
            consumed = 0
            for segment in iter_audio_segments(path):
                start = segment.start_sample
                expected_pcm = pcm[start : start + SEGMENT_SAMPLES]
                if segment.samples.shape != expected_pcm.shape:
                    raise ValueError(
                        f"PCM length mismatch: {path}, segment {segment.index}"
                    )
                consumed += segment.valid_samples
                features = log_mel(segment.samples)
                upstream = extractor(
                    expected_pcm,
                    sampling_rate=16000,
                    return_attention_mask=True,
                    return_tensors="np",
                    device=args.device,
                    padding="longest",
                    pad_to_multiple_of=1280,
                )
                upstream_mel = upstream.input_features[0]
                if features.mel.shape != upstream_mel.shape:
                    raise ValueError(
                        f"Mel shape mismatch: {path}, segment {segment.index}"
                    )
                if int(upstream.attention_mask.sum()) != features.valid_frames:
                    raise ValueError(
                        f"Valid frame mismatch: {path}, segment {segment.index}"
                    )
                actual = context.encode_audio(features.mel, features.valid_frames)
                pooled = context.features(actual.size)
                same_mel = reference.encode(features.mel, features.valid_frames)[
                    0, : actual.size
                ]
                reference_pooled = reference.last_pooled[0, : actual.size]
                end_to_end = reference.encode(upstream_mel, features.valid_frames)[
                    0, : actual.size
                ]
                case = {
                    "file_index": file_index,
                    "segment": segment.index,
                    "samples": segment.valid_samples,
                    "frames": features.valid_frames,
                    "pcm_max_abs_error": float(
                        np.abs(segment.samples - expected_pcm).max()
                    ),
                    "mel_max_abs_error": float(
                        np.abs(features.mel - upstream_mel).max()
                    ),
                    "same_mel": compare_codes(actual, same_mel),
                    "end_to_end": compare_codes(actual, end_to_end),
                    "reference_preprocessing": compare_codes(same_mel, end_to_end),
                    "pooled_features": feature_error(pooled, reference_pooled),
                    "peak_workspace_bytes": context.peak_workspace_bytes,
                }
                precise_values = {}
                if precise_reference is not None:
                    precise_codes = precise_reference.encode(
                        features.mel, features.valid_frames
                    )[0, : actual.size]
                    precise_pooled = precise_reference.last_pooled[0, : actual.size]
                    case["fp64_same_mel"] = compare_codes(actual, precise_codes)
                    case["fp64_pooled_features"] = feature_error(pooled, precise_pooled)
                    precise_values = {
                        "fp64_codes": precise_codes,
                        "fp64_pooled": precise_pooled,
                    }
                if args.save_mismatches and (
                    case["same_mel"]["different"]
                    or case["end_to_end"]["different"]
                    or case.get("fp64_same_mel", {}).get("different", 0)
                ):
                    filename = f"difference-{file_index}-{segment.index}.npz"
                    np.savez_compressed(
                        args.output / filename,
                        mel=features.mel,
                        upstream_mel=upstream_mel,
                        valid_frames=features.valid_frames,
                        codes=actual,
                        reference_codes=same_mel,
                        upstream_codes=end_to_end,
                        pooled=pooled,
                        reference_pooled=reference_pooled,
                        **precise_values,
                    )
                    case["diagnostic_file"] = filename
                report["cases"].append(case)
                if len(benchmark_inputs) < args.benchmark_cases:
                    benchmark_inputs.append((file_index, features))
            if consumed != pcm.size:
                raise ValueError(f"Segmentation did not consume all PCM: {path}")
            if (file_index + 1) % 25 == 0 or file_index + 1 == len(records):
                report["summary"] = summary(report["cases"])
                report["elapsed_seconds"] = time.perf_counter() - begin
                write_report(args.output / "report.json", report)
                print(
                    json.dumps({"files": file_index + 1, **report["summary"]}),
                    flush=True,
                )
        for file_index, features in benchmark_inputs:
            calls = {
                "native": lambda f=features: context.encode_audio(
                    f.mel, f.valid_frames
                ),
                "upstream": lambda f=features: reference.encode(f.mel, f.valid_frames)[
                    0, : f.code_count
                ],
            }
            expected = {name: call() for name, call in calls.items()}
            for _ in range(3):
                for call in calls.values():
                    call()
            times = {name: [] for name in calls}
            repeatable = {name: True for name in calls}
            for repeat in range(args.repeats):
                for name in list(calls) if repeat % 2 == 0 else list(reversed(calls)):
                    synchronize(args.device)
                    started = time.perf_counter()
                    codes = calls[name]()
                    synchronize(args.device)
                    times[name].append((time.perf_counter() - started) * 1000)
                    repeatable[name] &= bool(np.array_equal(codes, expected[name]))
            report["benchmark"].append(
                {
                    "file_index": file_index,
                    "samples": features.valid_samples,
                    "scope": "CPU mel to CPU codes, model only, excludes feature copies and preprocessing",
                    "repeats": args.repeats,
                    "repeatable": repeatable,
                    "median_ms": {
                        name: float(np.median(values)) for name, values in times.items()
                    },
                    "milliseconds": times,
                }
            )
    report["complete"] = True
    report["elapsed_seconds"] = time.perf_counter() - begin
    write_report(args.output / "report.json", report)


if __name__ == "__main__":
    main()
