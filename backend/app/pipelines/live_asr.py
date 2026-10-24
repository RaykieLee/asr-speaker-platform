"""Per-WebSocket streaming ASR sessions with bounded audio buffering."""
from __future__ import annotations

import asyncio
import array
import inspect
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from backend.app.audio.core import pcm_s16le_to_float32
from backend.app.engines.base import EngineConfigError
from backend.app.speakers.live_diarization import LiveSpeakerTracker


class LiveProtocolError(ValueError):
    """A client sent a message that violates the streaming protocol."""


class LiveQueueFull(RuntimeError):
    """The bounded audio queue cannot keep up with an incoming client."""


class OnlineStream(Protocol):
    def accept_waveform(self, sample_rate: int, samples: Any) -> None: ...
    def input_finished(self) -> None: ...


class OnlineRecognizer(Protocol):
    def create_stream(self, hotwords: list[str] | None = None) -> OnlineStream: ...
    def is_ready(self, stream: OnlineStream) -> bool: ...
    def decode_stream(self, stream: OnlineStream) -> None: ...
    def get_result(self, stream: OnlineStream) -> str: ...
    def is_endpoint(self, stream: OnlineStream) -> bool: ...


@dataclass(frozen=True)
class StartOptions:
    sample_rate: int = 16_000
    format: str = "s16le"
    hotwords: tuple[str, ...] = ()
    hotword_set_id: str | None = None
    hotword_set_version: int | str | None = None
    offline_hotwords: tuple[str, ...] = ()


@dataclass(frozen=True)
class FinalAudio:
    """Private audio snapshot for an offline correction; never sent over WS."""

    session_id: str
    segment_id: int
    revision: int
    start: float
    end: float
    samples: array.array
    hotwords: tuple[str, ...]
    speaker: dict[str, Any]


class LiveASRSession:
    """Owns one recognizer stream; recognizer/model weights may be shared."""

    def __init__(self, recognizer: OnlineRecognizer, *, session_id: str | None = None,
                 options: StartOptions | None = None,
                 speaker_tracker: LiveSpeakerTracker | None = None,
                 max_segment_seconds: float = 30.0):
        self.recognizer = recognizer
        self.session_id = session_id or f"sess_{uuid.uuid4().hex}"
        self.options = options or StartOptions()
        if max_segment_seconds <= 0:
            raise ValueError("max_segment_seconds 必须大于零")
        self.max_segment_samples = round(max_segment_seconds * self.options.sample_rate)
        if self.max_segment_samples < 1:
            raise ValueError("max_segment_seconds 太短")
        self.stream = recognizer.create_stream(list(self.options.hotwords) or None)
        self.samples_seen = 0
        self.segment_id = 1
        self.revision = 0
        self.last_text = ""
        self.segment_start = 0.0
        self.segment_start_sample = 0
        self.segment_samples = array.array("f")
        self.closed = False
        self.speaker_tracker = speaker_tracker or LiveSpeakerTracker()
        self._final_audio: list[FinalAudio] = []

    @property
    def duration(self) -> float:
        return self.samples_seen / self.options.sample_rate

    def _event(self, status: str, text: str, *, start: float, end: float) -> dict[str, Any]:
        return {
            "type": "segment", "session_id": self.session_id,
            "segment_id": self.segment_id, "revision": self.revision,
            "status": status, "start": start, "end": max(start, end),
            "text": text, "local_speaker_id": None,
            "registered_speaker_id": None, "speaker_name": None, "overlap": False,
        }

    def feed(self, pcm: bytes) -> list[dict[str, Any]]:
        if self.closed:
            raise LiveProtocolError("会话已结束")
        samples = pcm_s16le_to_float32(pcm, sample_rate=self.options.sample_rate)
        if not samples:
            return []
        all_events: list[dict[str, Any]] = []
        offset = 0
        while offset < len(samples):
            # Flush exactly at the segment limit, including when callers send
            # unusually large PCM frames that span several segment boundaries.
            if len(self.segment_samples) >= self.max_segment_samples:
                all_events.extend(self._rollover_segment())
            capacity = self.max_segment_samples - len(self.segment_samples)
            piece = samples[offset:offset + capacity]
            start = self.duration
            self.samples_seen += len(piece)
            end = self.duration
            self.segment_samples.extend(piece)
            offset += len(piece)

            speaker_changes = self.speaker_tracker.feed(piece, self.samples_seen)
            # The first confident identity labels the still-open segment; only
            # a transition away from an established identity creates a boundary.
            speaker_changes = [change for change in speaker_changes
                               if (change.get("previous", {}).get("local_speaker_id") or
                                   change.get("previous", {}).get("registered_speaker_id"))]
            if speaker_changes:
                # The tracker confirms changes after looking at overlapping
                # windows. Re-decode buffered audio at each estimated boundary.
                events: list[dict[str, Any]] = []
                for change in speaker_changes:
                    events.extend(self._split_at_speaker_change(change))
            else:
                self.stream.accept_waveform(self.options.sample_rate, piece)
                events = self._decode(start, end, final=False)
            all_events.extend(events)
            all_events.extend(self._rollover_if_needed())
        return all_events

    def finish(self) -> list[dict[str, Any]]:
        if self.closed:
            return []
        self.closed = True
        self.stream.input_finished()
        self.speaker_tracker.flush(self.samples_seen)
        return self._decode(self.duration, self.duration, final=True)

    def _decode(self, start: float, end: float, *, final: bool) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        text = self.recognizer.get_result(self.stream).strip()
        endpoint = self.recognizer.is_endpoint(self.stream)
        should_finalize = (endpoint and bool(text)) or (final and bool(text))
        if should_finalize:
            self.revision += 1
            event = self._event("final", text, start=self.segment_start, end=end)
            self._apply_speaker(event)
            events.append(event)
            self._queue_final_audio(event, self.segment_samples)
            self.last_text = ""
            if not final:
                self.segment_id += 1
                self.revision = 0
                self.segment_start = end
                self.segment_start_sample = self.samples_seen
                self.segment_samples = array.array("f")
                self.stream = self.recognizer.create_stream(list(self.options.hotwords) or None)
        elif text and text != self.last_text:
            self.revision += 1
            event = self._event("partial", text, start=self.segment_start, end=end)
            self._apply_speaker(event)
            events.append(event)
            self.last_text = text
        elif endpoint or final:
            # Preserve the boundary PCM even when online decoding is empty or
            # unchanged; offline decoding may still recover a usable transcript.
            event = self._event("final", text, start=self.segment_start, end=end)
            self._apply_speaker(event)
            self._queue_final_audio(event, self.segment_samples)
            if endpoint and not final:
                self.segment_id += 1
                self.revision = 0
                self.segment_start = end
                self.segment_start_sample = self.samples_seen
                self.segment_samples = array.array("f")
                self.last_text = ""
                self.stream = self.recognizer.create_stream(list(self.options.hotwords) or None)
        return events

    def _split_at_speaker_change(self, change: dict[str, Any]) -> list[dict[str, Any]]:
        boundary_sample = round(float(change["start"]) * self.options.sample_rate)
        boundary_sample = max(self.segment_start_sample,
                              min(self.samples_seen, boundary_sample))
        split_index = boundary_sample - self.segment_start_sample
        split_index = max(0, min(len(self.segment_samples), split_index))
        before = array.array("f", self.segment_samples[:split_index])
        after = array.array("f", self.segment_samples[split_index:])

        # Rebuild and flush the old side so its final text cannot include any
        # post-boundary samples that were already seen while the change settled.
        old_text = self._recognize_final(before)
        events: list[dict[str, Any]] = []
        if old_text:
            self.revision += 1
            event = self._event("final", old_text, start=self.segment_start,
                                end=boundary_sample / self.options.sample_rate)
            event.update(change.get("previous") or {})
            events.append(event)
            self._queue_final_audio(event, before)
        elif before:
            # Offline ASR may recover speech that the online stream missed.
            self._queue_final_audio({
                "segment_id": self.segment_id, "revision": self.revision,
                "start": self.segment_start,
                "end": boundary_sample / self.options.sample_rate,
                **(change.get("previous") or {}),
            }, before)

        self.segment_id += 1
        self.revision = 0
        self.segment_start_sample = boundary_sample
        self.segment_start = boundary_sample / self.options.sample_rate
        self.segment_samples = after
        self.last_text = ""
        self.stream = self.recognizer.create_stream(list(self.options.hotwords) or None)
        if after:
            self.stream.accept_waveform(self.options.sample_rate, after)
            events.extend(self._decode(self.segment_start, self.duration, final=False))
        return events

    def _recognize_final(self, samples: array.array) -> str:
        if not samples:
            return ""
        stream = self.recognizer.create_stream(list(self.options.hotwords) or None)
        stream.accept_waveform(self.options.sample_rate, samples)
        stream.input_finished()
        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)
        return self.recognizer.get_result(stream).strip()

    def _queue_final_audio(self, event: dict[str, Any], samples: array.array) -> None:
        if not samples:
            return
        speaker_keys = ("local_speaker_id", "registered_speaker_id", "speaker_name", "overlap")
        self._final_audio.append(FinalAudio(
            session_id=self.session_id,
            segment_id=int(event["segment_id"]), revision=int(event["revision"]),
            start=float(event["start"]), end=float(event["end"]),
            samples=array.array("f", samples), hotwords=self.options.offline_hotwords,
            speaker={key: event.get(key) for key in speaker_keys},
        ))

    def drain_final_audio(self) -> list[FinalAudio]:
        pending, self._final_audio = self._final_audio, []
        return pending

    def _rollover_if_needed(self) -> list[dict[str, Any]]:
        if len(self.segment_samples) < self.max_segment_samples:
            return []
        return self._rollover_segment()

    def _rollover_segment(self) -> list[dict[str, Any]]:
        """Finalize a long open segment and continue with a fresh ASR stream."""
        end = self.duration
        self.stream.input_finished()
        events = self._decode(self.segment_start, end, final=True)
        self.segment_id += 1
        self.revision = 0
        self.segment_start = end
        self.segment_start_sample = self.samples_seen
        self.segment_samples = array.array("f")
        self.last_text = ""
        self.stream = self.recognizer.create_stream(list(self.options.hotwords) or None)
        return events

    def _apply_speaker(self, event: dict[str, Any]) -> None:
        speaker = self.speaker_tracker._as_dict(self.speaker_tracker.current)
        event.update(speaker)

class LiveASRService:
    """Factory seam used by the API and fake recognizers in tests."""

    def __init__(self, recognizer: OnlineRecognizer, *, queue_size: int = 16,
                 hotword_resolver: Any = None, vad: Any = None,
                 speaker_extractor: Any = None, speaker_matcher: Any = None,
                 speaker_profiles: Any = (), speaker_options: dict[str, Any] | None = None,
                 offline_hotword_resolver: Any = None):
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self.recognizer = recognizer
        self.queue_size = queue_size
        self.hotword_resolver = hotword_resolver
        self.vad = vad
        self.speaker_extractor = speaker_extractor
        self.speaker_matcher = speaker_matcher
        self.speaker_profiles = speaker_profiles
        self.speaker_options = speaker_options or {}
        self.offline_hotword_resolver = offline_hotword_resolver

    async def create_session(self, options: StartOptions | None = None) -> LiveASRSession:
        if options and options.hotword_set_id and not self.hotword_resolver:
            raise EngineConfigError("热词集解析器未配置，不能忽略 hotword_set_id")
        if options and options.hotword_set_id and self.hotword_resolver:
            try:
                resolved = self.hotword_resolver(options.hotword_set_id)
                if inspect.isawaitable(resolved):
                    resolved = await resolved
                words, version = resolved
                if not isinstance(words, (tuple, list)) or any(
                    not isinstance(word, str) or not word.strip() for word in words
                ):
                    raise ValueError("热词集内容无效")
            except Exception as exc:
                raise EngineConfigError(f"热词集解析失败: {exc}") from exc
            offline_words: tuple[str, ...] = ()
            if self.offline_hotword_resolver is not None:
                try:
                    offline_resolved = self.offline_hotword_resolver(options.hotword_set_id, version)
                    if inspect.isawaitable(offline_resolved):
                        offline_resolved = await offline_resolved
                    offline_words, offline_version = offline_resolved
                    if offline_version != version:
                        raise ValueError("在线与离线热词版本不一致")
                    if not isinstance(offline_words, (tuple, list)) or any(
                        not isinstance(word, str) or not word.strip() for word in offline_words
                    ):
                        raise ValueError("离线热词快照内容无效")
                except Exception as exc:
                    raise EngineConfigError(f"离线热词快照失败: {exc}") from exc
            options = StartOptions(sample_rate=options.sample_rate, format=options.format,
                                   hotwords=tuple(word.strip() for word in words), hotword_set_id=options.hotword_set_id,
                                   hotword_set_version=version,
                                   offline_hotwords=tuple(word.strip() for word in offline_words))
        tracker = LiveSpeakerTracker(vad=self.vad, extractor=self.speaker_extractor,
                                     matcher=self.speaker_matcher, profiles=self.speaker_profiles,
                                     sample_rate=options.sample_rate if options else 16_000,
                                     **self.speaker_options)
        return LiveASRSession(self.recognizer, options=options, speaker_tracker=tracker)

    def create_queue(self) -> asyncio.Queue[bytes | None]:
        return asyncio.Queue(maxsize=self.queue_size)


def parse_start_message(message: Any) -> StartOptions:
    if not isinstance(message, dict) or message.get("type") != "start":
        raise LiveProtocolError("首条消息必须是 JSON start")
    version = message.get("protocol_version", message.get("version", 1))
    if version != 1:
        raise LiveProtocolError("不支持的协议版本")
    if message.get("sample_rate", 16_000) != 16_000:
        raise LiveProtocolError("实时 PCM 采样率必须为 16000 Hz")
    if message.get("format", "s16le") != "s16le":
        raise LiveProtocolError("实时音频格式必须为 s16le")
    if message.get("channels", 1) != 1:
        raise LiveProtocolError("实时 PCM 必须为单声道")
    words = message.get("hotwords", [])
    if not isinstance(words, list) or any(not isinstance(word, str) or not word.strip() for word in words):
        raise LiveProtocolError("hotwords 必须是非空字符串数组")
    hotword_id = message.get("hotword_set_id")
    if hotword_id is not None and not isinstance(hotword_id, str):
        raise LiveProtocolError("hotword_set_id 必须是字符串")
    return StartOptions(hotwords=tuple(words), hotword_set_id=hotword_id,
                        hotword_set_version=message.get("hotword_set_version"))
