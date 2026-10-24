"""Run the original GLM-4-Voice encoder against audio boundary fixtures.

This offline tool requires torch, safetensors, transformers, torchaudio, and
a separate checkout of GLM-4-Voice. It never downloads or publishes weights.
The reference keeps upstream loading and attention defaults. Explicit attention
and TF32 overrides are diagnostic options, recorded in the results. Synthetic
fixtures test boundaries; use validate_audio.py for real-speech comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import WhisperFeatureExtractor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from omni_tokenizer.audio import SincResampler, log_mel, mel_filters


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def signal(length: int) -> np.ndarray:
    time_axis = np.arange(length, dtype=np.float64) / 16000
    return (
        0.22 * np.sin(2 * np.pi * 440 * time_axis)
        + 0.08 * np.sin(2 * np.pi * 997 * time_axis)
    ).astype(np.float32)


class AudioReference:
    def __init__(
        self,
        checkpoint: Path,
        reference_root: Path,
        device: str = "cuda",
        attention: str = "default",
    ):
        sys.path.insert(0, str(reference_root))
        from speech_tokenizer.modeling_whisper import WhisperVQEncoder

        self.device = torch.device(device)
        options = {} if attention == "default" else {"attn_implementation": attention}
        self.model = (
            WhisperVQEncoder.from_pretrained(
                str(checkpoint), local_files_only=True, **options
            )
            .eval()
            .to(self.device)
        )
        self._last_pooled = None
        self.model.pooling_layer.register_forward_hook(self._pool_hook)

    def _pool_hook(self, module, inputs, output):
        self._last_pooled = output.detach().transpose(1, 2)

    @property
    def last_pooled(self):
        """Copy diagnostic features only when requested, outside timed inference."""
        if self._last_pooled is None:
            return None
        return self._last_pooled.cpu().numpy().copy()

    def execution_settings(self):
        return {
            "attention": self.model.config._attn_implementation,
            "attention_class": type(self.model.layers[0].self_attn).__name__,
            "dtype": str(next(self.model.parameters()).dtype),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        }

    @torch.inference_mode()
    def encode(self, mel: np.ndarray, valid_frames: int | list[int]) -> np.ndarray:
        if mel.ndim == 2:
            mel = mel[None]
        counts = [valid_frames] if isinstance(valid_frames, int) else valid_frames
        mask = (
            torch.arange(mel.shape[-1], device=self.device)[None]
            < torch.tensor(counts, device=self.device)[:, None]
        )
        outputs = self.model(
            input_features=torch.from_numpy(mel).to(
                self.device, dtype=self.model.conv1.weight.dtype
            ),
            attention_mask=mask.long(),
        )
        return outputs.quantized_token_ids.cpu().numpy().astype(np.uint16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--attention", default="default", choices=("default", "eager", "sdpa")
    )
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--lengths", nargs="+", type=int)
    parser.add_argument("--skip-padding-probes", action="store_true")
    parser.add_argument("--skip-resampler-probe", action="store_true")
    args = parser.parse_args()
    if args.lengths and any(length < 1 or length > 480000 for length in args.lengths):
        parser.error("--lengths must be between 1 and 480000 samples")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    reference = AudioReference(
        args.checkpoint, args.reference_root, args.device, args.attention
    )
    extractor = WhisperFeatureExtractor.from_pretrained(str(args.checkpoint))
    report = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "device": args.device,
        "execution": reference.execution_settings(),
        "input_kind": "synthetic boundary fixtures; not a speech quality evaluation",
        "checkpoint_sha256": sha256(args.checkpoint / "model.safetensors"),
        "reference_source_sha256": sha256(
            args.reference_root / "speech_tokenizer" / "modeling_whisper.py"
        ),
        "feature_profile": "numpy-sinc-hann6-mel-v1",
        "canonical_padding": "ceil(valid_samples / 1280) * 1280",
        "mel_filter_max_abs_error": float(
            np.abs(mel_filters() - extractor.mel_filters).max()
        ),
        "cases": [],
        "resampler": [],
    }
    lengths = args.lengths or [1, 159, 160, 1279, 1280, 1281, 12800, 64001]
    if not args.lengths and not args.skip_full:
        lengths.append(480000)
    for length in lengths:
        waveform = signal(length)
        features = log_mel(waveform)
        padded = np.pad(waveform, (0, features.padded_samples - length))
        reference_numpy_mel = extractor._np_extract_fbank_features(
            padded[None], device="cpu"
        )[0]
        reference_torch_mel = extractor._torch_extract_fbank_features(
            padded[None], device=args.device
        )[0]
        if args.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        ids = reference.encode(features.mel, features.valid_frames)[
            0, : features.code_count
        ]
        pooled = reference.last_pooled[0, : features.code_count]
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        peak = torch.cuda.max_memory_allocated() if args.device == "cuda" else None
        reference_ids = reference.encode(reference_torch_mel, features.valid_frames)[
            0, : features.code_count
        ]
        record = {
            "valid_samples": length,
            "padded_samples": features.padded_samples,
            "valid_frames": features.valid_frames,
            "code_count": features.code_count,
            "numpy_feature_max_abs_error": float(
                np.abs(features.mel - reference_numpy_mel).max()
            ),
            "torch_feature_max_abs_error": float(
                np.abs(features.mel - reference_torch_mel).max()
            ),
            "torch_feature_code_flips": int(np.count_nonzero(ids != reference_ids)),
            "encode_seconds": elapsed,
            "cuda_peak_allocated_bytes": peak,
            "codes": ids.tolist(),
        }
        np.savez(
            args.output / f"audio-{length}.npz",
            waveform=waveform,
            mel=features.mel,
            reference_torch_mel=reference_torch_mel,
            codes=ids,
            pooled=pooled,
            valid_frames=np.int64(features.valid_frames),
            valid_samples=np.int64(length),
        )
        if not args.skip_padding_probes and length in (1280, 1281, 12800):
            longest = np.pad(waveform, (0, 480000 - length))
            longest_mel = extractor._torch_extract_fbank_features(
                longest[None], device=args.device
            )[0]
            longest_ids = reference.encode(longest_mel, features.valid_frames)[
                0, : features.code_count
            ]
            record["reference_longest_batch_padding_code_flips"] = int(
                np.count_nonzero(reference_ids != longest_ids)
            )
            record["reference_longest_batch_padding_codes"] = longest_ids.tolist()
            record["longest_padding_valid_mel_max_abs_error"] = float(
                np.abs(
                    reference_torch_mel[:, : features.valid_frames]
                    - longest_mel[:, : features.valid_frames]
                ).max()
            )
        if length == 12800:
            # Feature-level collation retains the canonical waveform boundary.
            long_features = log_mel(signal(25600))
            collated = np.zeros((2, 128, long_features.mel.shape[-1]), dtype=np.float32)
            collated[0, :, : features.mel.shape[-1]] = features.mel
            collated[1] = long_features.mel
            batch_ids = reference.encode(
                collated, [features.valid_frames, long_features.valid_frames]
            )[0, : features.code_count]
            record["feature_collation_code_flips"] = int(
                np.count_nonzero(ids != batch_ids)
            )
            with wave.open(str(args.output / "audio-12800.wav"), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(np.round(waveform * 32767).astype("<i2").tobytes())
        report["cases"].append(record)
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(record), flush=True)
    if not args.skip_resampler_probe:
        import torchaudio

        report["torchaudio"] = torchaudio.__version__
    for rate in (
        () if args.skip_resampler_probe else (8000, 16000, 22050, 44100, 48000, 96000)
    ):
        waveform = (
            np.random.default_rng(0).normal(0, 0.2, rate // 4 + 3).astype(np.float32)
        )
        resampler = SincResampler(rate)
        actual = np.concatenate((resampler.process(waveform), resampler.flush()))
        chunks = SincResampler(rate)
        chunked = np.concatenate(
            [
                *(
                    chunks.process(waveform[i : i + 37])
                    for i in range(0, waveform.size, 37)
                ),
                chunks.flush(),
            ]
        )
        expected = torchaudio.transforms.Resample(rate, 16000)(
            torch.from_numpy(waveform)
        ).numpy()
        report["resampler"].append(
            {
                "source_rate": rate,
                "output_length": int(actual.size),
                "chunk_invariant": bool(np.array_equal(actual, chunked)),
                "reference_max_abs_error": float(np.abs(actual - expected).max()),
                "reference_rms_error": float(
                    np.sqrt(np.mean((actual - expected) ** 2))
                ),
            }
        )
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
