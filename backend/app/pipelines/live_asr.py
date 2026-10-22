"""Per-WebSocket streaming ASR sessions with bounded audio buffering."""
from __future__ import annotations

import asyncio
import array
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from app.audio.core import pcm_s16le_to_float32
from app.engines.base import EngineConfigError


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


class LiveASRSession:
    """Owns one recognizer stream; recognizer/model weights may be shared."""

    def __init__(self, recognizer: OnlineRecognizer, *, session_id: str | None = None,
                 options: StartOptions | None = None):
        self.recognizer = recognizer
        self.session_id = session_id or f"sess_{uuid.uuid4().hex}"
        self.options = options or StartOptions()
        self.stream = recognizer.create_stream(list(self.options.hotwords) or None)
        self.samples_seen = 0
        self.segment_id = 1
        self.revision = 0
        self.last_text = ""
        self.segment_start = 0.0
        self.closed = False

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
        start = self.duration
        self.samples_seen += len(samples)
        end = self.duration
        self.stream.accept_waveform(self.options.sample_rate, samples)
        return self._decode(start, end, final=False)

    def finish(self) -> list[dict[str, Any]]:
        if self.closed:
            return []
        self.closed = True
        self.stream.input_finished()
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
            events.append(self._event("final", text, start=self.segment_start, end=end))
            self.last_text = ""
            if not final:
                self.segment_id += 1
                self.revision = 0
                self.segment_start = end
                self.stream = self.recognizer.create_stream(list(self.options.hotwords) or None)
        elif text and text != self.last_text:
            self.revision += 1
            events.append(self._event("partial", text, start=self.segment_start, end=end))
            self.last_text = text
        return events


class LiveASRService:
    """Factory seam used by the API and fake recognizers in tests."""

    def __init__(self, recognizer: OnlineRecognizer, *, queue_size: int = 16, hotword_resolver: Any = None):
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self.recognizer = recognizer
        self.queue_size = queue_size
        self.hotword_resolver = hotword_resolver

    def create_session(self, options: StartOptions | None = None) -> LiveASRSession:
        if options and options.hotword_set_id and not self.hotword_resolver:
            raise EngineConfigError("热词集解析器未配置，不能忽略 hotword_set_id")
        if options and options.hotword_set_id and self.hotword_resolver:
            resolved = self.hotword_resolver(options.hotword_set_id)
            if hasattr(resolved, "__await__"):
                raise TypeError("异步 hotword_resolver 请在 API 层解析")
            words, version = resolved
            options = StartOptions(sample_rate=options.sample_rate, format=options.format,
                                   hotwords=tuple(word.strip() for word in words), hotword_set_id=options.hotword_set_id,
                                   hotword_set_version=version)
        return LiveASRSession(self.recognizer, options=options)

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
