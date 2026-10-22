"""Session-scoped rolling-window speaker mapping for live audio."""
from __future__ import annotations

import array
import math
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from backend.app.audio import AudioBuffer


class LiveVAD(Protocol):
    def is_speech(self, samples: Sequence[float], sample_rate: int) -> bool: ...


class LiveEmbeddingExtractor(Protocol):
    model_version: str
    def extract(self, audio: AudioBuffer) -> Sequence[float]: ...


@dataclass(frozen=True)
class SpeakerIdentity:
    local_speaker_id: str | None = None
    registered_speaker_id: str | None = None
    speaker_name: str | None = None


class LiveSpeakerTracker:
    """Detect speaker changes from overlapping windows with confirmation hysteresis.

    VAD and model operations are injected so each WebSocket owns only its state.
    The matcher receives a normalized embedding and registered profile rows; it
    must return an identity mapping or ``None`` when the score is inconclusive.
    """

    def __init__(self, *, vad: LiveVAD | Callable[..., bool] | None = None,
                 extractor: LiveEmbeddingExtractor | None = None,
                 matcher: Callable[[Sequence[float], Sequence[dict[str, Any]]], Any] | None = None,
                 profiles: Sequence[dict[str, Any]] | Callable[[], Sequence[dict[str, Any]]] = (),
                 sample_rate: int = 16_000, window_seconds: float = 1.5,
                 step_seconds: float = 0.5, min_speech_seconds: float = 1.0,
                 min_dwell_seconds: float = 1.0, confirmations: int = 2):
        if sample_rate <= 0 or window_seconds <= 0 or step_seconds <= 0 or min_speech_seconds <= 0:
            raise ValueError("实时声纹窗口参数必须大于零")
        self.vad, self.extractor, self.matcher = vad, extractor, matcher
        self.profiles = profiles
        self.sample_rate = sample_rate
        self.window_samples = round(window_seconds * sample_rate)
        self.step_samples = round(step_seconds * sample_rate)
        self.min_speech_samples = round(min_speech_seconds * sample_rate)
        self.min_dwell_samples = round(min_dwell_seconds * sample_rate)
        self.confirmations = max(1, confirmations)
        self.samples = array.array("f")
        self.processed = 0
        self.base_sample = 0
        self.current = SpeakerIdentity()
        self.current_since = 0
        self.pending: SpeakerIdentity | None = None
        self.pending_count = 0
        self._local_vectors: list[list[float]] = []

    @property
    def available(self) -> bool:
        return self.vad is not None and self.extractor is not None and self.matcher is not None

    def feed(self, samples: Sequence[float], end_sample: int) -> list[dict[str, Any]]:
        if not samples:
            return []
        self.samples.extend(samples)
        events: list[dict[str, Any]] = []
        while len(self.samples) - self.processed >= self.step_samples:
            self.processed += self.step_samples
            left = max(0, self.processed - self.window_samples)
            window = self.samples[left:self.processed]
            if len(window) < self.min_speech_samples or not self._is_speech(window):
                continue
            candidate = self._identify(window)
            if candidate is None:
                continue
            if candidate == self.current:
                self.pending, self.pending_count = None, 0
                continue
            self.pending_count = self.pending_count + 1 if candidate == self.pending else 1
            self.pending = candidate
            absolute_processed = self.base_sample + self.processed
            if (self.pending_count >= self.confirmations and
                    absolute_processed - self.current_since >= self.min_dwell_samples):
                previous = self.current
                self.current = candidate
                self.current_since = absolute_processed
                self.pending, self.pending_count = None, 0
                events.append({"type": "speaker_change", "start": (self.base_sample + left) / self.sample_rate,
                               "end": (self.base_sample + self.processed) / self.sample_rate,
                               "previous": self._as_dict(previous),
                               "speaker": self._as_dict(candidate), "confirmed_at": end_sample / self.sample_rate})
        # Keep enough history for the next overlapping window and absolute indices.
        discard = max(0, self.processed - self.window_samples)
        if discard:
            del self.samples[:discard]
            self.processed -= discard
            self.base_sample += discard
        return events

    def flush(self, end_sample: int) -> list[dict[str, Any]]:
        # Deliberately do not infer identity from an undersized tail window.
        return []

    def _is_speech(self, samples: Sequence[float]) -> bool:
        try:
            if hasattr(self.vad, "is_speech"):
                return bool(self.vad.is_speech(samples, self.sample_rate))
            return bool(self.vad(samples, self.sample_rate))  # type: ignore[misc]
        except Exception:
            return False

    def _identify(self, samples: Sequence[float]) -> SpeakerIdentity | None:
        if not self.available:
            return None
        audio = AudioBuffer(array.array("f", samples), self.sample_rate, 1)
        try:
            vector = _normalize(self.extractor.extract(audio))  # type: ignore[union-attr]
            rows = self.profiles() if callable(self.profiles) else self.profiles
            extractor_version = getattr(self.extractor, "model_version", None)
            if extractor_version:
                # Unknown-version profiles cannot be safely compared either:
                # embeddings from different model releases are not compatible.
                rows = [row for row in rows
                        if row.get("model_version") == extractor_version]
            result = self.matcher(vector, rows)  # type: ignore[misc]
        except Exception:
            return None
        identity = _identity(result)
        if identity and identity.registered_speaker_id:
            return identity
        # Unmatched voices still receive stable session-local ids when their
        # embeddings are sufficiently close; this never invents a registered name.
        for index, centroid in enumerate(self._local_vectors):
            if len(centroid) == len(vector) and _cosine(centroid, vector) >= 0.82:
                return SpeakerIdentity(local_speaker_id=f"speaker_{index}")
        self._local_vectors.append(vector)
        return SpeakerIdentity(local_speaker_id=f"speaker_{len(self._local_vectors)-1}")

    @staticmethod
    def _as_dict(identity: SpeakerIdentity) -> dict[str, str | None]:
        return {"local_speaker_id": identity.local_speaker_id,
                "registered_speaker_id": identity.registered_speaker_id,
                "speaker_name": identity.speaker_name}


def _identity(value: Any) -> SpeakerIdentity | None:
    if value is None:
        return None
    if isinstance(value, SpeakerIdentity):
        return value
    if isinstance(value, dict):
        return SpeakerIdentity(value.get("local_speaker_id"), value.get("registered_speaker_id"), value.get("speaker_name"))
    if isinstance(value, (tuple, list)):
        values = list(value) + [None, None, None]
        return SpeakerIdentity(values[0], values[1], values[2])
    return None


def _normalize(values: Sequence[float]) -> list[float]:
    result = [float(v) for v in values]
    norm = math.sqrt(sum(v * v for v in result))
    if not result or not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("无效声纹向量")
    return [v / norm for v in result]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def profile_cosine_matcher(*, min_similarity: float, min_margin: float) -> Callable[[Sequence[float], Sequence[dict[str, Any]]], Any]:
    """Build a conservative matcher for rows containing id/name/embedding.

    Thresholds are mandatory because they require calibration on authorized,
    labeled audio for the actual embedding model and deployment environment.
    """
    if not -1.0 <= min_similarity <= 1.0 or not 0.0 <= min_margin <= 2.0:
        raise ValueError("声纹匹配阈值超出有效范围")

    def match(vector: Sequence[float], profiles: Sequence[dict[str, Any]]) -> SpeakerIdentity | None:
        scores = []
        for row in profiles:
            profile = row.get("embedding")
            if not profile or len(profile) != len(vector):
                continue
            try:
                score = _cosine(vector, _normalize(profile))
            except (TypeError, ValueError):
                continue
            scores.append((score, row))
        scores.sort(key=lambda item: item[0], reverse=True)
        if not scores or scores[0][0] < min_similarity:
            return None
        if len(scores) > 1 and scores[0][0] - scores[1][0] < min_margin:
            return None
        winner = scores[0][1]
        return SpeakerIdentity(local_speaker_id=winner.get("local_speaker_id"),
                               registered_speaker_id=winner.get("id"),
                               speaker_name=winner.get("name"))
    return match
