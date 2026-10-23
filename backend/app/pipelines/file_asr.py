"""Offline ASR over normalized audio with bounded chunks and source timestamps."""
from __future__ import annotations

import array
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from backend.app.audio.core import AudioBuffer, AudioFormatError, decode_audio_file, normalize_audio, separate_stereo
from backend.app.engines.base import EngineConfigError


class OfflineRecognizer(Protocol):
    model_version: str | None

    def transcribe(self, samples: Any, sample_rate: int,
                   hotwords: list[str | list[str]] | None = None) -> str: ...


class DiarizationProvider(Protocol):
    """T09 integration seam; ranges use the original source-audio timeline."""

    def segment(self, audio: AudioBuffer) -> list[dict[str, Any]]: ...


class SpeakerMatcherProvider(Protocol):
    def match(self, audio: AudioBuffer, segments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]: ...


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
    speaker_diarization: str = "disabled"
    speaker_mappings: tuple[dict[str, Any], ...] = ()

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
                 diarizer: DiarizationProvider | None = None,
                 speaker_matcher: SpeakerMatcherProvider | None = None):
        if max_chunk_seconds <= 0:
            raise ValueError("max_chunk_seconds must be positive")
        if overlap_seconds < 0 or overlap_seconds >= max_chunk_seconds:
            raise ValueError("overlap_seconds must be smaller than max_chunk_seconds")
        self.recognizer = recognizer
        self.max_chunk_seconds = max_chunk_seconds
        self.overlap_seconds = overlap_seconds
        self.diarizer = diarizer
        self.speaker_matcher = speaker_matcher

    @property
    def model_version(self) -> str | None:
        return getattr(self.recognizer, "model_version", None)

    def transcribe_file(self, path: str | Path, *, hotwords: list[str | list[str]] | None = None,
                        channel_mode: str = "mixdown",
                        progress: Callable[[float], None] | None = None) -> FileASRResult:
        if self.recognizer is None:
            raise EngineConfigError("离线 ASR 模型未配置或未安装")
        if channel_mode not in {"mixdown", "separate"}:
            raise ValueError("channel_mode 必须是 mixdown 或 separate")
        audio = decode_audio_file(path, preserve_channels=channel_mode == "separate")
        return self.transcribe_audio(audio, hotwords=hotwords, channel_mode=channel_mode,
                                     progress=progress)

    def transcribe_audio(self, source: AudioBuffer, *, hotwords: list[str | list[str]] | None = None,
                         channel_mode: str = "mixdown",
                         progress: Callable[[float], None] | None = None) -> FileASRResult:
        if self.recognizer is None:
            raise EngineConfigError("离线 ASR 模型未配置或未安装")
        if channel_mode not in {"mixdown", "separate"}:
            raise ValueError("channel_mode 必须是 mixdown 或 separate")
        if channel_mode == "separate":
            if source.channels != 2:
                raise AudioFormatError("channel_mode=separate 需要恰好两个声道的音频")
            left, right = separate_stereo(source)
            combined: list[ASRSegment] = []
            mappings: list[dict[str, Any]] = []
            channel_results: list[FileASRResult] = []
            for channel_index, channel in enumerate((left, right)):
                channel_audio = normalize_audio(channel)

                def update_channel_progress(value: float, *, index: int = channel_index) -> None:
                    if progress:
                        progress((index + value) / 2)

                result = self.transcribe_audio(
                    channel_audio, hotwords=hotwords, channel_mode="mixdown",
                    progress=update_channel_progress,
                )
                if (result.speaker_diarization == "no_speech" and result.text.strip()
                        and self.speaker_matcher is not None):
                    result = self._fallback_match_transcribed_channel(channel_audio, result)
                channel_results.append(result)
                local_id_map: dict[str, str] = {}
                for mapping in result.speaker_mappings:
                    old_local_id = str(mapping["local_speaker_id"])
                    new_local_id = f"channel_{channel_index}:{old_local_id}"
                    local_id_map[old_local_id] = new_local_id
                    mappings.append({**mapping, "local_speaker_id": new_local_id,
                                     "channel_index": channel_index})
                for segment in result.segments:
                    metadata = dict(segment.metadata or {})
                    metadata["channel_index"] = channel_index
                    local_id = segment.local_speaker_id
                    if local_id is not None:
                        local_id = local_id_map.get(local_id, f"channel_{channel_index}:{local_id}")
                    combined.append(ASRSegment(
                        segment_id=0, start=segment.start, end=segment.end, text=segment.text,
                        local_speaker_id=local_id,
                        registered_speaker_id=segment.registered_speaker_id,
                        overlap=segment.overlap, revision=segment.revision, metadata=metadata,
                    ))

            # Keep one deterministic timeline across both independent microphones.
            combined.sort(key=lambda segment: (
                segment.start, segment.end, int((segment.metadata or {}).get("channel_index", 0)),
            ))
            channel_zero = [(index, segment) for index, segment in enumerate(combined)
                            if (segment.metadata or {}).get("channel_index") == 0]
            channel_one = [(index, segment) for index, segment in enumerate(combined)
                           if (segment.metadata or {}).get("channel_index") == 1]
            cross_overlaps: set[int] = set()
            left_index = right_index = 0
            while left_index < len(channel_zero) and right_index < len(channel_one):
                left_pos, left = channel_zero[left_index]
                right_pos, right = channel_one[right_index]
                if min(left.end, right.end) > max(left.start, right.start):
                    cross_overlaps.update((left_pos, right_pos))
                if left.end <= right.end:
                    left_index += 1
                if right.end <= left.end:
                    right_index += 1
            for index in cross_overlaps:
                segment = combined[index]
                metadata = {**(segment.metadata or {}), "cross_channel_overlap": True}
                combined[index] = ASRSegment(
                    segment_id=segment.segment_id, start=segment.start, end=segment.end,
                    text=segment.text, local_speaker_id=segment.local_speaker_id,
                    registered_speaker_id=segment.registered_speaker_id,
                    overlap=True, revision=segment.revision, metadata=metadata,
                )
            combined = [ASRSegment(
                segment_id=index, start=segment.start, end=segment.end, text=segment.text,
                local_speaker_id=segment.local_speaker_id,
                registered_speaker_id=segment.registered_speaker_id,
                overlap=segment.overlap, revision=segment.revision, metadata=segment.metadata,
            ) for index, segment in enumerate(combined, start=1)]
            statuses = {result.speaker_diarization for result in channel_results}
            if statuses <= {"disabled"}:
                diarization_status = "disabled"
            elif statuses <= {"no_speech", "disabled"}:
                diarization_status = "no_speech"
            elif statuses <= {"unmatched", "no_speech", "disabled"}:
                diarization_status = "unmatched"
            else:
                diarization_status = "completed"
            if progress:
                progress(1.0)
            return FileASRResult(source.duration, tuple(combined), self.model_version,
                                 diarization_status, tuple(mappings))
        audio = normalize_audio(source)
        if audio.sample_rate != 16_000 or audio.channels != 1:
            raise EngineConfigError("离线 ASR 仅接受标准化的 16000 Hz 单声道音频")
        if self.diarizer is None:
            segments = self._transcribe_plain(audio, hotwords=hotwords, progress=progress)
            return FileASRResult(audio.duration, tuple(segments), self.model_version)
        if not audio.frame_count:
            if progress:
                progress(1.0)
            return FileASRResult(audio.duration, (), self.model_version, "no_speech", ())

        diarized = self.diarizer.segment(audio)
        diarized = self._normalize_diarization_ranges(diarized, audio.duration)
        if not diarized:
            # Diarization can miss speech even when the recognizer can still
            # recover it. Preserve the transcript and leave speaker identity
            # unset instead of treating an empty diarization result as silence.
            segments = self._transcribe_plain(audio, hotwords=hotwords, progress=progress)
            return FileASRResult(audio.duration, tuple(segments), self.model_version,
                                 "no_speech", ())

        mappings = self.speaker_matcher.match(audio, diarized) if self.speaker_matcher else {}
        local_ids = list(dict.fromkeys(item["local_speaker_id"] for item in diarized))
        for index, local_id in enumerate(local_ids, start=1):
            mappings.setdefault(local_id, {
                "local_speaker_id": local_id,
                "registered_speaker_id": None,
                "speaker_name": f"未知说话人 {index}",
                "similarity": None,
                "margin": None,
                "match_status": "disabled" if self.speaker_matcher is None else "unknown",
                "model_version": None,
                "clean_sample_count": 0,
            })

        output: list[ASRSegment] = []
        total = len(diarized)
        for range_index, item in enumerate(diarized):
            start_frame = max(0, min(audio.frame_count, round(item["start"] * audio.sample_rate)))
            end_frame = max(start_frame, min(audio.frame_count, round(item["end"] * audio.sample_rate)))
            samples = audio.samples[start_frame:end_frame]
            crop = AudioBuffer(array.array("f", samples), audio.sample_rate, 1,
                               audio.source_sample_rate or audio.sample_rate, 1,
                               item["end"] - item["start"])

            def update_range_progress(value: float) -> None:
                if progress:
                    progress((range_index + value) / total)

            local_segments = self._transcribe_plain(
                crop, hotwords=hotwords, progress=update_range_progress,
            ) if samples else []
            if not local_segments:
                local_segments = [ASRSegment(0, 0.0, crop.duration, "")]
            mapping = mappings[item["local_speaker_id"]]
            for segment in local_segments:
                metadata = dict(segment.metadata or {})
                metadata.update({
                    "speaker_name": mapping["speaker_name"],
                    "speaker_match_status": mapping["match_status"],
                    "speaker_similarity": mapping.get("similarity"),
                    "speaker_margin": mapping.get("margin"),
                    "diarization_start": item["start"],
                    "diarization_end": item["end"],
                })
                start = item["start"] + segment.start
                end = min(item["end"], item["start"] + segment.end)
                text = segment.text
                segment_overlap = any(
                    other["local_speaker_id"] != item["local_speaker_id"]
                    and min(end, other["end"]) > max(start, other["start"])
                    for other in diarized
                )
                duplicate = any(
                    existing.text and text and existing.text.strip() == text.strip()
                    and min(existing.end, end) > max(existing.start, start)
                    for existing in output
                ) if segment_overlap else False
                if duplicate:
                    text = ""
                    metadata["duplicate_overlap_text_suppressed"] = True
                output.append(ASRSegment(
                    len(output) + 1, start, max(start, end), text,
                    local_speaker_id=item["local_speaker_id"],
                    registered_speaker_id=mapping.get("registered_speaker_id"),
                    overlap=segment_overlap, metadata=metadata,
                ))

        if progress:
            progress(1.0)
        status = "completed" if self.speaker_matcher else "unmatched"
        return FileASRResult(audio.duration, tuple(output), self.model_version,
                             status, tuple(mappings[local_id] for local_id in local_ids))

    def _fallback_match_transcribed_channel(self, audio: AudioBuffer,
                                            result: FileASRResult) -> FileASRResult:
        """Best-effort speaker match from voiced windows when diarization is empty.

        ASR has already run. This fallback only examines those existing transcript
        time ranges and never reruns recognition or changes segment text/timestamps.
        """
        matcher = self.speaker_matcher
        if matcher is None:
            return result
        minimum_seconds = float(getattr(matcher, "min_segment_seconds", 1.0))
        minimum_count = int(getattr(matcher, "min_segments_per_speaker", 2))
        maximum_count = int(getattr(matcher, "max_segments_per_speaker", 5))
        windows = self._voiced_embedding_windows(
            audio, result.segments, min_duration=minimum_seconds,
            minimum_count=minimum_count, maximum_count=maximum_count,
        )
        if len(windows) < minimum_count:
            return result

        local_id = "fallback_speaker_0"
        candidates = [
            {"start": start, "end": end, "local_speaker_id": local_id, "overlap": False}
            for start, end in windows
        ]
        try:
            matches = matcher.match(audio, candidates)
        except Exception:
            # Speaker enrichment must not turn an already completed transcript into
            # a failed job when an optional embedding extraction fails.
            return result
        mapping = matches.get(local_id) if isinstance(matches, dict) else None
        if not isinstance(mapping, dict):
            return result
        mapping = dict(mapping)
        clean_count = int(mapping.get("clean_sample_count") or 0)
        if (mapping.get("match_status") != "matched"
                or not mapping.get("registered_speaker_id")
                or clean_count < minimum_count):
            mapping.update(registered_speaker_id=None, match_status="unknown")
        if not mapping.get("speaker_name"):
            mapping["speaker_name"] = "未知说话人 1"
        mapping["local_speaker_id"] = local_id
        mapping["fallback_method"] = "asr_energy_windows"

        segments: list[ASRSegment] = []
        for segment in result.segments:
            if not segment.text.strip():
                segments.append(segment)
                continue
            metadata = dict(segment.metadata or {})
            metadata.update({
                "speaker_name": mapping["speaker_name"],
                "speaker_match_status": mapping["match_status"],
                "speaker_similarity": mapping.get("similarity"),
                "speaker_margin": mapping.get("margin"),
                "speaker_fallback": mapping["fallback_method"],
                "speaker_fallback_window_count": len(windows),
            })
            segments.append(ASRSegment(
                segment_id=segment.segment_id, start=segment.start, end=segment.end,
                text=segment.text, local_speaker_id=local_id,
                registered_speaker_id=mapping.get("registered_speaker_id"),
                overlap=segment.overlap, revision=segment.revision, metadata=metadata,
            ))
        status = "completed" if mapping["match_status"] == "matched" else "unmatched"
        return FileASRResult(result.duration, tuple(segments), result.model_version,
                             status, (mapping,))

    @staticmethod
    def _voiced_embedding_windows(audio: AudioBuffer, segments: tuple[ASRSegment, ...], *,
                                  min_duration: float, minimum_count: int,
                                  maximum_count: int) -> list[tuple[float, float]]:
        """Find non-overlapping, sufficiently loud windows inside nonempty ASR ranges."""
        if (audio.channels != 1 or audio.sample_rate <= 0 or min_duration <= 0
                or minimum_count < 1 or maximum_count < minimum_count):
            return []
        frame_size = max(1, round(audio.sample_rate * 0.03))
        min_frames = max(1, math.ceil(min_duration * audio.sample_rate))
        max_gap_frames = round(0.15 * audio.sample_rate)
        voiced_runs: list[tuple[int, int]] = []

        for segment in segments:
            if not segment.text.strip():
                continue
            lower = max(0, min(audio.frame_count, round(segment.start * audio.sample_rate)))
            upper = max(lower, min(audio.frame_count, round(segment.end * audio.sample_rate)))
            active: list[tuple[int, int]] = []
            for start in range(lower, upper, frame_size):
                end = min(upper, start + frame_size)
                if end <= start:
                    continue
                samples = audio.samples[start:end]
                rms = math.sqrt(sum(float(value) ** 2 for value in samples) / len(samples))
                if rms >= 0.008:
                    active.append((start, end))
            for start, end in active:
                if voiced_runs and start - voiced_runs[-1][1] <= max_gap_frames:
                    voiced_runs[-1] = (voiced_runs[-1][0], end)
                else:
                    voiced_runs.append((start, end))

        windows: list[tuple[float, float]] = []
        for start, end in voiced_runs:
            frame_count = end - start
            count = min(maximum_count - len(windows), int(frame_count / min_frames + 1e-9))
            if count <= 0:
                continue
            # Equal partitioning avoids a too-short tail and ensures every window
            # meets the matcher’s duration gate; boundaries never overlap.
            for index in range(count):
                window_start = start + round(frame_count * index / count)
                window_end = start + round(frame_count * (index + 1) / count)
                if window_end - window_start >= min_frames:
                    windows.append((window_start / audio.sample_rate,
                                    window_end / audio.sample_rate))
                    if len(windows) == maximum_count:
                        return windows
        return windows

    @staticmethod
    def _normalize_diarization_ranges(ranges: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in ranges:
            try:
                start, end = float(item["start"]), float(item["end"])
                local_id = str(item.get("local_speaker_id", ""))
            except (KeyError, TypeError, ValueError) as exc:
                raise EngineConfigError(f"说话人分段器返回无效区间: {item!r}") from exc
            if not local_id or not 0 <= start < end <= duration + 1e-3:
                raise EngineConfigError(f"说话人分段器返回越界或空区间: {item!r}")
            normalized.append({**item, "start": max(0.0, start),
                               "end": min(duration, end), "local_speaker_id": local_id,
                               "overlap": bool(item.get("overlap", False))})
        normalized.sort(key=lambda item: (item["start"], item["end"], item["local_speaker_id"]))
        for index, current in enumerate(normalized):
            if any(other["local_speaker_id"] != current["local_speaker_id"]
                   and min(current["end"], other["end"]) > max(current["start"], other["start"])
                   for other_index, other in enumerate(normalized) if other_index != index):
                current["overlap"] = True
        return normalized

    def _transcribe_plain(self, audio: AudioBuffer, *, hotwords: list[str | list[str]] | None,
                          progress: Callable[[float], None] | None) -> list[ASRSegment]:
        windows = _chunk_windows(audio.frame_count, audio.sample_rate,
                                 self.max_chunk_seconds, self.overlap_seconds)
        if not windows:
            return []

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
        return segments
