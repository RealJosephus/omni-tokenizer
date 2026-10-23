import io
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
from omni_tokenizer.audio import (
    AudioCancelled,
    AudioError,
    SincResampler,
    iter_audio_segments,
    iter_decoded_chunks,
    log_mel,
    mel_filters,
    segment_pcm,
)


def wav_bytes(samples, *, rate=16000, width=2):
    samples = np.asarray(samples)
    channels = samples.shape[1] if samples.ndim == 2 else 1
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(width)
        stream.setframerate(rate)
        stream.writeframes(samples.tobytes())
    output.seek(0)
    return output


class AudioTests(unittest.TestCase):
    def test_first_channel_and_integer_pcm(self):
        stereo = np.array([[-32768, 2000], [0, 3000], [32767, 4000]], dtype="<i2")
        decoded = list(iter_decoded_chunks(wav_bytes(stereo), block_frames=2))
        np.testing.assert_array_equal(
            np.concatenate([x for x, _ in decoded]),
            stereo[:, 0].astype(np.float32) / 32768,
        )
        self.assertEqual([rate for _, rate in decoded], [16000, 16000])
        for width, data, expected in (
            (1, np.array([0, 128, 255], np.uint8), [-1, 0, 127 / 128]),
            (
                3,
                np.array([0, 0, 128, 0, 0, 0, 255, 255, 127], np.uint8),
                [-1, 0, 8388607 / 8388608],
            ),
            (4, np.array([-2147483648, 0, 1073741824], "<i4"), [-1, 0, 0.5]),
        ):
            chunks = list(iter_decoded_chunks(wav_bytes(data, width=width)))
            np.testing.assert_array_equal(chunks[0][0], np.array(expected, np.float32))

    def test_truncated_and_empty_audio(self):
        data = wav_bytes(np.arange(100, dtype="<i2")).getvalue()
        with self.assertRaisesRegex(AudioError, "shorter"):
            list(iter_decoded_chunks(io.BytesIO(data[:-2])))
        with self.assertRaisesRegex(AudioError, "empty"):
            list(iter_audio_segments(wav_bytes(np.empty(0, dtype="<i2"))))
        with self.assertRaisesRegex(AudioError, "empty"):
            list(segment_pcm([], 16000))

    def test_resampler_chunk_invariance_and_flush(self):
        for rate in (8000, 16000, 22050, 44100, 48000, 96000):
            samples = np.random.default_rng(42).normal(0, 0.2, 5003).astype(np.float32)
            complete = SincResampler(rate)
            expected = np.concatenate((complete.process(samples), complete.flush()))
            resampler = SincResampler(rate)
            output = []
            for begin in range(0, samples.size, 37):
                output.append(resampler.process(samples[begin : begin + 37]))
                self.assertLess(resampler.buffered_samples, 160)
            output.append(resampler.flush())
            actual = np.concatenate(output)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.size, (samples.size * 16000 + rate - 1) // rate)
            self.assertEqual(resampler.buffered_samples, 0)
            with self.assertRaisesRegex(AudioError, "flushed"):
                resampler.flush()

    def test_resampler_impulse_and_anti_aliasing(self):
        resampler = SincResampler(48000)
        impulse = np.zeros(480, np.float32)
        impulse[240] = 1
        output = resampler.process(impulse, final=True)
        self.assertEqual(int(np.argmax(output)), 80)
        self.assertAlmostEqual(float(output.sum()), 1 / 3, delta=0.001)
        time = np.arange(48000) / 48000
        low = np.sin(2 * np.pi * 1000 * time).astype(np.float32)
        high = np.sin(2 * np.pi * 16000 * time).astype(np.float32)
        low_output = SincResampler(48000).process(low, final=True)[100:-100]
        high_output = SincResampler(48000).process(high, final=True)[100:-100]
        self.assertGreater(float(np.std(low_output)), 0.69)
        self.assertLess(float(np.std(high_output)), 0.001)

    def test_exact_segment_boundaries_and_offsets(self):
        for length, expected in (
            (1, [1]),
            (479999, [479999]),
            (480000, [480000]),
            (480001, [480000, 1]),
            (960000, [480000, 480000]),
        ):
            samples = np.zeros(length, np.float32)
            chunks = (samples[begin : begin + 1023] for begin in range(0, length, 1023))
            segments = list(segment_pcm(chunks, 16000))
            self.assertEqual([item.valid_samples for item in segments], expected)
            self.assertEqual(
                [item.start_sample for item in segments],
                [i * 480000 for i in range(len(expected))],
            )
            self.assertEqual(
                [item.index for item in segments], list(range(len(expected)))
            )

    def test_pull_backpressure_and_cancellation(self):
        reads = []

        def chunks():
            for index in range(100):
                reads.append(index)
                yield np.zeros(16000, np.float32)

        generator = segment_pcm(chunks(), 16000)
        self.assertEqual(next(generator).valid_samples, 480000)
        self.assertEqual(len(reads), 30)
        generator.close()
        self.assertEqual(len(reads), 30)
        with self.assertRaises(AudioCancelled):
            list(segment_pcm([np.zeros(1, np.float32)], 16000, cancel=lambda: True))

    def test_padding_and_validity_counts(self):
        for length, valid_frames, codes in (
            (1, 1, 1),
            (159, 1, 1),
            (160, 1, 1),
            (1279, 8, 1),
            (1280, 8, 1),
            (1281, 9, 2),
            (480000, 3000, 375),
        ):
            features = log_mel(np.zeros(length, np.float32))
            self.assertEqual(features.valid_frames, valid_frames)
            self.assertEqual(features.code_count, codes)
            self.assertEqual(features.padded_samples, codes * 1280)
            self.assertEqual(features.mel.shape, (128, codes * 8))
            self.assertEqual(features.mel.dtype, np.float32)
            self.assertTrue(features.mel.flags.c_contiguous)
            np.testing.assert_array_equal(features.mel, -1.5)

    def test_mel_filter_geometry_and_segment_local_range(self):
        filters = mel_filters()
        self.assertEqual(filters.shape, (201, 128))
        self.assertEqual(int(np.count_nonzero(filters)), 394)
        self.assertTrue((filters >= 0).all())
        samples = (0.25 * np.sin(np.arange(12800) * 0.2)).astype(np.float32)
        first = log_mel(samples).mel
        log_mel(np.ones(480000, np.float32))
        np.testing.assert_array_equal(first, log_mel(samples).mel)
        with self.assertRaises(AudioError):
            log_mel(np.array([np.nan], np.float32))
        with self.assertRaises(AudioError):
            log_mel(np.zeros(480001, np.float32))

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is optional"
    )
    def test_ffmpeg_preserves_rate_and_first_channel(self):
        values = np.arange(4401, dtype=np.int32)
        stereo = np.column_stack(
            ((values % 1000 - 500) * 30, np.full(4401, 30000))
        ).astype("<i2")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.wav"
            compressed = Path(directory) / "compressed.flac"
            source.write_bytes(wav_bytes(stereo, rate=44100).getvalue())
            subprocess.run(
                [
                    shutil.which("ffmpeg"),
                    "-v",
                    "error",
                    "-nostdin",
                    "-i",
                    str(source),
                    str(compressed),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            chunks = list(iter_decoded_chunks(compressed, block_frames=317))
            self.assertTrue(all(rate == 44100 for _, rate in chunks))
            np.testing.assert_array_equal(
                np.concatenate([chunk for chunk, _ in chunks]),
                stereo[:, 0].astype(np.float32) / 32768,
            )
            short_blocks = list(iter_audio_segments(compressed, block_frames=37))
            long_blocks = list(iter_audio_segments(compressed, block_frames=4096))
            np.testing.assert_array_equal(
                short_blocks[0].samples, long_blocks[0].samples
            )
            self.assertEqual(
                short_blocks[0].valid_samples, (4401 * 16000 + 44099) // 44100
            )


if __name__ == "__main__":
    unittest.main()
