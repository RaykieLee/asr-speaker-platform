"""Small, dependency-light audio primitives for ASR ingestion.

Samples are represented as ``array('f')`` in interleaved channel order. Timestamps
are always seconds from the beginning of the source recording; resampling changes
the sample grid but never the time origin or reported duration.
"""
from __future__ import annotations

import array
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1


class AudioFormatError(ValueError):
    """The input is not a supported or decodable audio file."""


class PCMFormatError(AudioFormatError):
    """Raw PCM does not match the required signed 16-bit little-endian layout."""


class FFmpegUnavailableError(RuntimeError):
    """FFmpeg is required to decode or resample this file but is unavailable."""


@dataclass(frozen=True)
class AudioBuffer:
    samples: array.array
    sample_rate: int
    channels: int
    source_sample_rate: int | None = None
    source_channels: int | None = None
    source_duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.samples.typecode != "f":
            raise TypeError("samples must be array('f') float32 data")
        if self.sample_rate <= 0 or self.channels <= 0:
            raise ValueError("sample_rate and channels must be positive")
        if len(self.samples) % self.channels:
            raise ValueError("interleaved sample count must be divisible by channels")

    @property
    def frame_count(self) -> int:
        return len(self.samples) // self.channels

    @property
    def duration(self) -> float:
        return self.source_duration_seconds if self.source_duration_seconds is not None else self.frame_count / self.sample_rate

    def timestamp_for_frame(self, frame: int) -> float:
        """Return a frame's source-relative time, including the exact EOF boundary.

        Resampling rounds the output frame count to an integer. Consequently the
        regular output grid can end a fraction of a sample before/after source EOF.
        Treat ``frame_count`` as the exclusive end boundary and anchor it to the
        original duration so segment endpoints do not drift on export.
        """
        if frame < 0 or frame > self.frame_count:
            raise IndexError("frame outside audio buffer")
        if frame == self.frame_count:
            return self.duration
        return frame / self.sample_rate


def validate_pcm_s16le(data: bytes | bytearray | memoryview, *, channels: int = 1,
                       sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    """Validate raw PCM framing for the required 16 kHz mono path by default."""
    if sample_rate != TARGET_SAMPLE_RATE or channels != TARGET_CHANNELS:
        raise PCMFormatError("raw PCM must be 16 kHz, mono, signed s16le")
    if len(data) % 2:
        raise PCMFormatError("s16le byte length must be even (no partial sample)")


def pcm_s16le_to_float32(data: bytes | bytearray | memoryview, *, channels: int = 1,
                          sample_rate: int = TARGET_SAMPLE_RATE) -> array.array:
    """Validate required PCM layout and scale signed little-endian samples to [-1, 1]."""
    validate_pcm_s16le(data, channels=channels, sample_rate=sample_rate)
    raw = array.array("h")
    raw.frombytes(bytes(data))
    if __import__("sys").byteorder != "little":
        raw.byteswap()
    return array.array("f", (max(-1.0, value / 32768.0) for value in raw))


def _pcm_bytes_to_float32(data: bytes, width: int) -> array.array:
    if width == 2:
        return pcm_s16le_to_float32(data)
    if width not in (1, 3, 4):
        raise AudioFormatError(f"unsupported PCM sample width: {width} bytes")
    out = array.array("f")
    if width == 1:  # WAV 8-bit PCM is unsigned.
        out.extend((b - 128) / 128.0 for b in data)
    elif width == 3:
        for i in range(0, len(data), 3):
            value = int.from_bytes(data[i:i + 3], "little", signed=True)
            out.append(value / 8388608.0)
    else:
        raw = array.array("i")
        raw.frombytes(data)
        if __import__("sys").byteorder != "little":
            raw.byteswap()
        out.extend(value / 2147483648.0 for value in raw)
    return out


def decode_wav(source: str | Path | bytes | bytearray | memoryview) -> AudioBuffer:
    """Read uncompressed integer PCM WAV (8/16/24/32 bit), preserving channel layout."""
    import io
    fileobj = io.BytesIO(bytes(source)) if isinstance(source, (bytes, bytearray, memoryview)) else str(source)
    try:
        with wave.open(fileobj, "rb") as wav:
            channels, rate, width, frames = wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getnframes()
            if channels < 1 or rate < 1:
                raise AudioFormatError("WAV must have positive channel count and sample rate")
            if wav.getcomptype() != "NONE":
                raise AudioFormatError(f"compressed WAV codec is unsupported: {wav.getcomptype()}")
            data = wav.readframes(frames)
            expected = frames * channels * width
            if len(data) != expected:
                raise AudioFormatError("truncated WAV frame data")
            samples = _pcm_bytes_to_float32(data, width)
            return AudioBuffer(samples, rate, channels, rate, channels)
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioFormatError(f"invalid WAV audio: {exc}") from exc


def mixdown(audio: AudioBuffer) -> AudioBuffer:
    """Average interleaved channels into mono; supports any positive channel count."""
    if audio.channels == 1:
        return audio
    mono = array.array("f")
    scale = 1.0 / audio.channels
    for frame in range(audio.frame_count):
        start = frame * audio.channels
        mono.append(sum(audio.samples[start:start + audio.channels]) * scale)
    return AudioBuffer(mono, audio.sample_rate, 1, audio.source_sample_rate, audio.source_channels, audio.duration)


def separate_stereo(audio: AudioBuffer) -> tuple[AudioBuffer, AudioBuffer]:
    """Split stereo interleaved audio into left and right mono buffers."""
    if audio.channels != 2:
        raise AudioFormatError("stereo separation requires exactly two channels")
    left, right = array.array("f"), array.array("f")
    left.extend(audio.samples[0::2]); right.extend(audio.samples[1::2])
    return (AudioBuffer(left, audio.sample_rate, 1, audio.source_sample_rate, 1, audio.duration),
            AudioBuffer(right, audio.sample_rate, 1, audio.source_sample_rate, 1, audio.duration))


def _resample(audio: AudioBuffer, target_rate: int) -> AudioBuffer:
    if target_rate <= 0:
        raise ValueError("target sample rate must be positive")
    if audio.sample_rate == target_rate:
        return audio
    if audio.frame_count == 0:
        return AudioBuffer(array.array("f"), target_rate, audio.channels,
                           audio.source_sample_rate or audio.sample_rate, audio.source_channels or audio.channels, audio.duration)
    # Linear interpolation keeps endpoints and computes output length from duration.
    count = max(1, round(audio.frame_count * target_rate / audio.sample_rate))
    out = array.array("f")
    for n in range(count):
        position = n * audio.sample_rate / target_rate
        lo = min(int(position), audio.frame_count - 1)
        hi = min(lo + 1, audio.frame_count - 1)
        frac = position - lo
        for ch in range(audio.channels):
            a, b = audio.samples[lo * audio.channels + ch], audio.samples[hi * audio.channels + ch]
            out.append(a + (b - a) * frac)
    return AudioBuffer(out, target_rate, audio.channels,
                       audio.source_sample_rate or audio.sample_rate, audio.source_channels or audio.channels, audio.duration)


def normalize_audio(audio: AudioBuffer, *, sample_rate: int = TARGET_SAMPLE_RATE,
                    channels: int = TARGET_CHANNELS) -> AudioBuffer:
    """Convert channel count and sample rate while retaining source timing metadata."""
    if channels != 1:
        raise ValueError("normalization currently targets mono audio")
    mono = mixdown(audio)
    normalized = _resample(mono, sample_rate)
    # Preserve exact source duration for timeline conversion despite integer frame rounding.
    return normalized


def decode_audio_file(path: str | Path, *, sample_rate: int = TARGET_SAMPLE_RATE,
                      channels: int = TARGET_CHANNELS) -> AudioBuffer:
    """Decode WAV directly; decode other formats via optional FFmpeg to s16le WAV."""
    path = Path(path)
    if not path.is_file():
        raise AudioFormatError(f"audio file does not exist: {path}")
    try:
        source = decode_wav(path)
    except AudioFormatError:
        if path.suffix.lower() == ".wav":
            raise
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise FFmpegUnavailableError("FFmpeg is required to decode this audio format; install ffmpeg and retry")
        try:
            result = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-f", "wav", "-acodec", "pcm_s16le", "-"],
                                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            source = decode_wav(result.stdout)
        except (subprocess.CalledProcessError, AudioFormatError) as exc:
            detail = exc.stderr.decode("utf-8", "replace") if isinstance(exc, subprocess.CalledProcessError) else str(exc)
            raise AudioFormatError(f"FFmpeg could not decode {path.name}: {detail}") from exc
    return normalize_audio(source, sample_rate=sample_rate, channels=channels)
