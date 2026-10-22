"""Offline ASR over normalized audio with bounded chunks and source timestamps."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from backend.app.audio.core import AudioBuffer, decode_audio_file, normalize_audio
from backend.app.engines.base import EngineConfigError


class OfflineRecognizer(Protocol):
    model_version: str | None

    def transcribe(self, samples: Any, sample_rate: int,
                   hotwords: list[str] | None = None) -> str: ...


class DiarizationProvider(Protocol):
    """T09 integration seam; ranges use the original source-audio timeline."""

    def segment(self, audio: AudioBuffer) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class ASRSegment:
    segment_id: int
    start: float
    end: float
    text: str
    local_speaker_id: str | None = None
    registered_speaker_id: str | None = None
    overlap: bool = False
    revision: int = 1
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class FileASRResult:
    duration: float
    segments: tuple[ASRSegment, ...]
    model_version: str | None

    @property
    def text(self) -> str:
        parts = [segment.text.strip() for segment in self.segments if segment.text.strip()]
        if not parts:
            return ""

        def is_cjk(char: str) -> bool:
            codepoint = ord(char)
            return (0x3400 <= codepoint <= 0x9FFF or 0xF900 <= codepoint <= 0xFAFF
                    or 0x3040 <= codepoint <= 0x30FF or 0xAC00 <= codepoint <= 0xD7AF)

        combined = parts[0]
        for part in parts[1:]:
            separator = "" if is_cjk(combined[-1]) or is_cjk(part[0]) else " "
            combined += separator + part
        return combined


def _deduplicate_overlap(previous: str, current: str, max_chars: int) -> str:
    """Drop the largest exact text suffix/prefix duplicated by adjacent windows."""
    previous, current = previous.strip(), current.strip()
    limit = min(len(previous), len(current), max_chars)
    for size in range(limit, 1, -1):
        if previous[-size:] == current[:size]:
            return current[size:].lstrip()
    return current


def _chunk_windows(frame_count: int, sample_rate: int, chunk_seconds: float,
                   overlap_seconds: float) -> list[tuple[int, int]]:
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be non-negative and smaller than chunk_seconds")
    maximum = max(1, round(chunk_seconds * sample_rate))
    overlap = round(overlap_seconds * sample_rate)
    step = maximum - overlap
    windows: list[tuple[int, int]] = []
    start = 0
    while start < frame_count:
        end = min(frame_count, start + maximum)
        windows.append((start, end))
        if end == frame_count:
            break
        start += step
    return windows


class FileASRService:
    """Decode files using an injected model adapter; no model means explicit failure."""

    def __init__(self, recognizer: OfflineRecognizer | None, *,
                 max_chunk_seconds: float = 60.0, overlap_seconds: float = 0.8,
                 diarizer: DiarizationProvider | None = None):
        if max_chunk_seconds <= 0:
            raise ValueError("max_chunk_seconds must be positive")
        if overlap_seconds < 0 or overlap_seconds >= max_chunk_seconds:
            raise ValueError("overlap_seconds must be smaller than max_chunk_seconds")
        self.recognizer = recognizer
        self.max_chunk_seconds = max_chunk_seconds
        self.overlap_seconds = overlap_seconds
        self.diarizer = diarizer

    @property
    def model_version(self) -> str | None:
        return getattr(self.recognizer, "model_version", None)

    def transcribe_file(self, path: str | Path, *, hotwords: list[str] | None = None,
                        progress: Callable[[float], None] | None = None) -> FileASRResult:
        if self.recognizer is None:
            raise EngineConfigError("离线 ASR 模型未配置或未安装")
        audio = decode_audio_file(path)
        return self.transcribe_audio(audio, hotwords=hotwords, progress=progress)

    def transcribe_audio(self, source: AudioBuffer, *, hotwords: list[str] | None = None,
                         progress: Callable[[float], None] | None = None) -> FileASRResult:
        if self.recognizer is None:
            raise EngineConfigError("离线 ASR 模型未配置或未安装")
        audio = normalize_audio(source)
        if audio.sample_rate != 16_000 or audio.channels != 1:
            raise EngineConfigError("离线 ASR 仅接受标准化的 16000 Hz 单声道音频")
        windows = _chunk_windows(audio.frame_count, audio.sample_rate,
                                 self.max_chunk_seconds, self.overlap_seconds)
        if not windows:
            return FileASRResult(audio.duration, (), self.model_version)

        decoded: list[tuple[int, int, str]] = []
        for index, (start, end) in enumerate(windows):
            samples = audio.samples[start:end]
            text = self.recognizer.transcribe(samples, audio.sample_rate, hotwords or None).strip()
            decoded.append((start, end, text))
            if progress:
                progress((index + 1) / len(windows))

        segments: list[ASRSegment] = []
        for index, (start, end, text) in enumerate(decoded):
            if index:
                _, prev_end, prev_text = decoded[index - 1]
                max_chars = max(2, round((prev_end - start) / audio.sample_rate * 20))
                text = _deduplicate_overlap(prev_text, text, max_chars)
            # Assign only the non-overlap portion of each window to its segment.
            owned_start = start if index == 0 else (decoded[index - 1][1] + start) / 2
            owned_end = end if index + 1 == len(decoded) else (end + decoded[index + 1][0]) / 2
            start_seconds = max(0.0, owned_start / audio.sample_rate)
            end_seconds = audio.duration if end == audio.frame_count else min(audio.duration, owned_end / audio.sample_rate)
            if text:
                segments.append(ASRSegment(
                    segment_id=len(segments) + 1, start=start_seconds, end=max(start_seconds, end_seconds),
                    text=text, metadata={"chunk_index": index, "chunk_start": start / audio.sample_rate,
                                         "chunk_end": audio.duration if end == audio.frame_count else end / audio.sample_rate},
                ))
        return FileASRResult(audio.duration, tuple(segments), self.model_version)
