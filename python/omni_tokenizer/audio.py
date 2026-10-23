"""Bounded audio input and the canonical VQWhisper preprocessing profile.

The profile uses the first channel, a continuous Hann-windowed sinc resampler
(six zero crossings, rolloff 0.99), and independent 480000-sample windows.
Each window is zero-padded to a multiple of 1280 before its own log-mel
calculation. This intentionally removes the reference extractor's dependence
on the longest waveform in a batch. It is not bit-exact to its CUDA STFT.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
from typing import BinaryIO, Callable, Iterable, Iterator
import wave

import numpy as np


SAMPLE_RATE = 16000
SEGMENT_SAMPLES = 480000
SAMPLE_MULTIPLE = 1280
PREPROCESSING_PROFILE = "audio-canonical-v1"
NUMERICAL_PROFILE = "numpy-sinc-hann6-mel-v1"
CancelCheck = Callable[[], bool] | None


class AudioError(ValueError):
    """An invalid, empty, unsupported, or truncated audio input."""


class AudioCancelled(RuntimeError):
    """Audio processing was cancelled before another block was consumed."""


def _check_cancel(cancel: CancelCheck) -> None:
    if cancel is not None and cancel():
        raise AudioCancelled("Audio processing was cancelled")


@dataclass(frozen=True)
class AudioSegment:
    index: int
    start_sample: int
    valid_samples: int
    samples: np.ndarray


@dataclass(frozen=True)
class AudioFeatures:
    mel: np.ndarray
    valid_samples: int
    padded_samples: int
    valid_frames: int
    code_count: int


class SincResampler:
    """A continuous, bounded rational resampler with explicit EOF flushing.

    Kernel coefficients are calculated in float64 then stored in float32;
    products and reductions use float32. The mathematical kernel matches
    torchaudio's default sinc_interp_hann settings, but reduction and phase
    rounding differ. Output length is ceil(input_length * target / source).
    Feed boundaries do not change values. No endpoint repetition is used.
    """

    def __init__(self, source_rate: int, target_rate: int = SAMPLE_RATE):
        if not isinstance(source_rate, int) or not 1 <= source_rate <= 384000:
            raise AudioError("Sample rate must be an integer between 1 and 384000")
        if target_rate != SAMPLE_RATE:
            raise AudioError("The canonical resampler target must be 16000 Hz")
        self.source_rate = source_rate
        self.target_rate = target_rate
        divisor = math.gcd(source_rate, target_rate)
        self._source = source_rate // divisor
        self._target = target_rate // divisor
        self._total = 0
        self._next = 0
        self._offset = 0
        self._closed = False
        self._buffer = np.empty(0, dtype=np.float32)
        self._identity = source_rate == target_rate
        scale = min(source_rate, target_rate) * 0.99 / source_rate
        self._radius = math.ceil(6 / scale)
        self._taps = np.arange(-self._radius, self._radius + 1, dtype=np.int64)
        self._weights = None
        if not self._identity:
            fractions = np.arange(self._target, dtype=np.float64) / self._target
            positions = (self._taps[None, :] - fractions[:, None]) * scale
            positions = np.clip(positions, -6.0, 6.0)
            self._weights = (
                np.sinc(positions) * np.cos(positions * (np.pi / 12.0)) ** 2 * scale
            ).astype(np.float32)

    @property
    def buffered_samples(self) -> int:
        return int(self._buffer.size)

    def process(self, samples: np.ndarray, *, final: bool = False) -> np.ndarray:
        if self._closed:
            raise AudioError("The resampler has already been flushed")
        data = np.asarray(samples, dtype=np.float32)
        if data.ndim != 1 or not np.isfinite(data).all():
            raise AudioError("PCM must be a finite mono vector")
        self._total += int(data.size)
        if self._identity:
            self._next = self._total
            self._closed = final
            return data.copy()
        if data.size:
            self._buffer = np.concatenate((self._buffer, data))
        available = self._total if final else max(0, self._total - self._radius)
        stop = (available * self._target + self._source - 1) // self._source
        output = np.empty(max(0, stop - self._next), dtype=np.float32)
        written = 0
        while self._next < stop:
            end = min(stop, self._next + 4096)
            numerators = np.arange(self._next, end, dtype=np.int64) * self._source
            centers, phases = np.divmod(numerators, self._target)
            indices = centers[:, None] + self._taps[None, :]
            valid = (indices >= 0) & (indices < self._total)
            relative = np.clip(indices - self._offset, 0, max(0, self._buffer.size - 1))
            values = self._buffer[relative] if self._buffer.size else np.zeros_like(relative, dtype=np.float32)
            values = np.where(valid, values, np.float32(0))
            count = end - self._next
            output[written : written + count] = np.sum(values * self._weights[phases], axis=1, dtype=np.float32)
            self._next = end
            written += count
        keep_from = max(0, self._next * self._source // self._target - self._radius)
        discard = min(self._buffer.size, max(0, keep_from - self._offset))
        self._buffer = self._buffer[discard:].copy()
        self._offset += int(discard)
        self._closed = final
        if final:
            self._buffer = np.empty(0, dtype=np.float32)
        return output

    def flush(self) -> np.ndarray:
        return self.process(np.empty(0, dtype=np.float32), final=True)


def _pcm_first_channel(raw: bytes, sample_width: int, channels: int) -> np.ndarray:
    frame_bytes = sample_width * channels
    if len(raw) % frame_bytes:
        raise AudioError("Truncated PCM frame")
    if sample_width == 1:
        return (np.frombuffer(raw, np.uint8)[::channels].astype(np.float32) - 128) / 128
    if sample_width == 2:
        return np.frombuffer(raw, "<i2")[::channels].astype(np.float32) / 32768
    if sample_width == 3:
        data = np.frombuffer(raw, np.uint8).reshape(-1, channels, 3)[:, 0, :].astype(np.int32)
        signed = data[:, 0] | (data[:, 1] << 8) | (data[:, 2] << 16)
        signed = (signed ^ 0x800000) - 0x800000
        return signed.astype(np.float32) / 8388608
    if sample_width == 4:
        return np.frombuffer(raw, "<i4")[::channels].astype(np.float32) / 2147483648
    raise AudioError("WAV PCM must use 8, 16, 24, or 32 bits per sample")


def _iter_ffmpeg(path: Path, block_frames: int, cancel: CancelCheck) -> Iterator[tuple[np.ndarray, int]]:
    executable, probe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if executable is None or probe is None:
        raise AudioError("This audio format requires ffmpeg and ffprobe on PATH; PCM WAV works without them")
    try:
        result = subprocess.run(
            [probe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate", "-of", "json", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioError("Audio probing exceeded its 30-second limit") from exc
    try:
        sample_rate = int(json.loads(result.stdout)["streams"][0]["sample_rate"])
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise AudioError("Could not identify a readable audio stream") from exc
    if not 1 <= sample_rate <= 384000:
        raise AudioError("Unsupported audio sample rate")
    # Keep the original rate: the same continuous resampler handles every codec.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            [executable, "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0", "-vn",
             "-af", "pan=mono|c0=c0", "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"],
            stdout=subprocess.PIPE, stderr=errors,
        )
        pending = queue.Queue(maxsize=2)
        stopped = threading.Event()

        def read_pipe():
            while not stopped.is_set():
                try:
                    item = process.stdout.read(block_frames * 4)
                except Exception as exc:
                    item = exc
                while not stopped.is_set():
                    try:
                        pending.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        pass
                if not item or isinstance(item, Exception):
                    return

        reader_thread = threading.Thread(target=read_pipe, daemon=True)
        reader_thread.start()
        try:
            while True:
                _check_cancel(cancel)
                try:
                    raw = pending.get(timeout=0.1)
                except queue.Empty:
                    continue
                if isinstance(raw, Exception):
                    raise AudioError("Could not read decoder output") from raw
                if not raw:
                    break
                if len(raw) % 4:
                    raise AudioError("Decoder returned a truncated PCM sample")
                samples = np.frombuffer(raw, "<f4").astype(np.float32, copy=True)
                if not np.isfinite(samples).all():
                    raise AudioError("Decoded audio contains non-finite samples")
                yield samples, sample_rate
            if process.wait(timeout=30) != 0:
                errors.seek(0)
                detail = errors.read(4096).decode("utf-8", "replace").strip()
                raise AudioError(f"Audio decoder failed: {detail}")
        finally:
            stopped.set()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()
            reader_thread.join(timeout=3)


def iter_decoded_chunks(
    source: str | os.PathLike[str] | BinaryIO, *, block_frames: int = 65536, cancel: CancelCheck = None,
) -> Iterator[tuple[np.ndarray, int]]:
    """Read bounded first-channel PCM chunks; optional FFmpeg handles other files.

    Binary streams must contain seekable PCM WAV. Other codecs are accepted as
    local paths, allowing containers that need seeking without buffering them
    in memory. Closing the generator also closes its decoder subprocess.
    """
    if not 1 <= block_frames <= 1048576:
        raise AudioError("block_frames must be between 1 and 1048576")
    _check_cancel(cancel)
    path = Path(source).resolve() if isinstance(source, (str, os.PathLike)) else None
    if path is not None and not path.is_file():
        raise AudioError("Audio file does not exist")
    try:
        reader = wave.open(str(path) if path is not None else source, "rb")
    except (wave.Error, EOFError) as exc:
        if path is None:
            raise AudioError("Binary streams must contain a supported PCM WAV file") from exc
        yield from _iter_ffmpeg(path, block_frames, cancel)
        return
    with reader:
        channels, width, rate = reader.getnchannels(), reader.getsampwidth(), reader.getframerate()
        if not 1 <= channels <= 256 or not 1 <= rate <= 384000:
            raise AudioError("Unsupported WAV channel count or sample rate")
        remaining = int(reader.getnframes())
        while remaining:
            _check_cancel(cancel)
            frames = min(remaining, block_frames, max(1, 1048576 // (channels * width)))
            raw = reader.readframes(frames)
            if len(raw) != frames * channels * width:
                raise AudioError("WAV data is shorter than its declared length")
            remaining -= frames
            yield _pcm_first_channel(raw, width, channels), rate


def segment_pcm(
    chunks: Iterable[np.ndarray], source_rate: int, *, cancel: CancelCheck = None,
) -> Iterator[AudioSegment]:
    """Continuously resample and emit 30-second windows with one bounded buffer.

    No extra segment is emitted for an exact multiple. Sample offsets are
    Python integers and remain exact beyond 32-bit audio lengths.
    """
    resampler = SincResampler(source_rate)
    buffer = np.empty(SEGMENT_SAMPLES, dtype=np.float32)
    filled = 0
    index = 0

    def output_chunks() -> Iterator[np.ndarray]:
        for chunk in chunks:
            _check_cancel(cancel)
            # Bound resampling temporary arrays even for a large caller chunk.
            samples = np.asarray(chunk, dtype=np.float32)
            if samples.ndim != 1:
                raise AudioError("PCM must be a mono vector")
            input_block = max(1, min(65536, source_rate * 65536 // SAMPLE_RATE))
            for begin in range(0, samples.size, input_block):
                _check_cancel(cancel)
                yield resampler.process(samples[begin : begin + input_block])
        yield resampler.flush()

    for converted in output_chunks():
        begin = 0
        while begin < converted.size:
            _check_cancel(cancel)
            take = min(SEGMENT_SAMPLES - filled, converted.size - begin)
            buffer[filled : filled + take] = converted[begin : begin + take]
            filled += take
            begin += take
            if filled == SEGMENT_SAMPLES:
                yield AudioSegment(index, index * SEGMENT_SAMPLES, filled, buffer.copy())
                index += 1
                filled = 0
    if filled:
        yield AudioSegment(index, index * SEGMENT_SAMPLES, filled, buffer[:filled].copy())
    elif index == 0:
        raise AudioError("Audio input is empty")


def iter_audio_segments(
    source: str | os.PathLike[str] | BinaryIO, *, block_frames: int = 65536, cancel: CancelCheck = None,
) -> Iterator[AudioSegment]:
    decoded = iter_decoded_chunks(source, block_frames=block_frames, cancel=cancel)
    try:
        try:
            first, rate = next(decoded)
        except StopIteration as exc:
            raise AudioError("Audio input is empty") from exc

        def chunks() -> Iterator[np.ndarray]:
            yield first
            for samples, chunk_rate in decoded:
                if chunk_rate != rate:
                    raise AudioError("Sample rate changed inside a stream")
                yield samples

        yield from segment_pcm(chunks(), rate, cancel=cancel)
    finally:
        decoded.close()


@lru_cache(maxsize=1)
def mel_filters() -> np.ndarray:
    """Return the 201 x 128 float64 Slaney filter bank for the fixed model."""
    max_mel = 15.0 + np.log(8000.0 / 1000.0) * (27.0 / np.log(6.4))
    mel = np.linspace(0.0, max_mel, 130)
    frequencies = mel * (200.0 / 3)
    logarithmic = mel >= 15.0
    frequencies[logarithmic] = 1000.0 * np.exp((mel[logarithmic] - 15.0) * (np.log(6.4) / 27.0))
    bins = np.linspace(0.0, 8000.0, 201)
    slopes = frequencies[None, :] - bins[:, None]
    differences = np.diff(frequencies)
    filters = np.maximum(0.0, np.minimum(-slopes[:, :-2] / differences[:-1], slopes[:, 2:] / differences[1:]))
    filters *= 2.0 / (frequencies[2:] - frequencies[:-2])
    filters.setflags(write=False)
    return filters


@lru_cache(maxsize=1)
def _hann() -> np.ndarray:
    window = np.hanning(401)[:-1]
    window.setflags(write=False)
    return window


def log_mel(samples: np.ndarray) -> AudioFeatures:
    """Extract a canonical segment's features independently of every other item.

    The float64 FFT is rounded to complex64 before float64 power/mel/log;
    log-mel then rounds to float32 before dynamic-range clipping. This follows
    the Transformers NumPy extractor's arithmetic, without importing it.
    """
    waveform = np.asarray(samples, dtype=np.float32)
    if waveform.ndim != 1 or not 1 <= waveform.size <= SEGMENT_SAMPLES:
        raise AudioError("A segment must contain between 1 and 480000 mono samples")
    if not np.isfinite(waveform).all():
        raise AudioError("Audio contains non-finite samples")
    valid = int(waveform.size)
    padded = (valid + SAMPLE_MULTIPLE - 1) // SAMPLE_MULTIPLE * SAMPLE_MULTIPLE
    waveform = np.pad(waveform, (0, padded - valid))
    waveform = np.pad(waveform, (200, 200), mode="reflect").astype(np.float64)
    # The final centered STFT frame is deliberately excluded.
    frames = padded // 160
    spectrum = np.empty((frames, 201), dtype=np.complex64)
    windows = np.lib.stride_tricks.sliding_window_view(waveform, 400)[::160][:frames]
    for begin in range(0, frames, 256):
        spectrum[begin : begin + 256] = np.fft.rfft(windows[begin : begin + 256] * _hann(), axis=-1)
    power = np.abs(spectrum, dtype=np.float64) ** 2
    features = np.log10(np.maximum(1e-10, mel_filters().T @ power.T)).astype(np.float32)
    np.maximum(features, features.max() - np.float32(8.0), out=features)
    features += np.float32(4.0)
    features *= np.float32(0.25)
    return AudioFeatures(
        np.ascontiguousarray(features), valid, padded, (valid + 159) // 160,
        (valid + SAMPLE_MULTIPLE - 1) // SAMPLE_MULTIPLE,
    )


def iter_feature_segments(
    source: str | os.PathLike[str] | BinaryIO, *, block_frames: int = 65536, cancel: CancelCheck = None,
) -> Iterator[tuple[AudioSegment, AudioFeatures]]:
    segments = iter_audio_segments(source, block_frames=block_frames, cancel=cancel)
    try:
        for segment in segments:
            _check_cancel(cancel)
            yield segment, log_mel(segment.samples)
    finally:
        segments.close()
