"""Match diarized local speakers to registered profiles conservatively."""
from __future__ import annotations

import array
import math
from typing import Any, Protocol

from backend.app.audio.core import AudioBuffer
from backend.app.engines.base import EngineConfigError
from backend.app.speakers.service import aggregate_embeddings, l2_normalize


class EmbeddingExtractor(Protocol):
    model_version: str

    def extract(self, audio: AudioBuffer) -> list[float]: ...


class SpeakerMatcher:
    """Build one embedding per local speaker and match only comparable profiles.

    The top candidate must exceed ``min_similarity`` and lead the runner-up by
    ``min_margin``. Samples that overlap another speaker are excluded.
    """

    def __init__(self, database: Any, extractor: EmbeddingExtractor, *,
                 min_similarity: float = 0.55, min_margin: float = 0.05,
                 min_segment_seconds: float = 1.0, max_segments_per_speaker: int = 5,
                 min_segments_per_speaker: int = 2):
        if not -1.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be between -1 and 1")
        if not 0.0 <= min_margin <= 2.0:
            raise ValueError("min_margin must be between 0 and 2")
        if min_segment_seconds <= 0 or min_segments_per_speaker < 1 or max_segments_per_speaker < min_segments_per_speaker:
            raise ValueError("invalid clean segment selection limits")
        self.database = database
        self.extractor = extractor
        self.min_similarity = min_similarity
        self.min_margin = min_margin
        self.min_segment_seconds = min_segment_seconds
        self.max_segments_per_speaker = max_segments_per_speaker
        self.min_segments_per_speaker = min_segments_per_speaker

    @staticmethod
    def _has_overlap(segment: dict[str, Any], segments: list[dict[str, Any]]) -> bool:
        if segment.get("overlap"):
            return True
        return any(
            other is not segment
            and other.get("local_speaker_id") != segment.get("local_speaker_id")
            and min(float(segment["end"]), float(other["end"]))
            > max(float(segment["start"]), float(other["start"]))
            for other in segments
        )

    def _local_embedding(self, audio: AudioBuffer, segments: list[dict[str, Any]],
                         local_id: str) -> tuple[list[float], int] | None:
        candidates = [item for item in segments
                      if item.get("local_speaker_id") == local_id
                      and not self._has_overlap(item, segments)
                      and float(item["end"]) - float(item["start"]) >= self.min_segment_seconds]
        # Prefer longer, cleaner opportunities, then keep temporal order stable.
        candidates.sort(key=lambda item: (-(
            float(item["end"]) - float(item["start"])), float(item["start"])))
        chosen: list[dict[str, Any]] = []
        for candidate in candidates:
            if any(min(float(candidate["end"]), float(other["end"]))
                   > max(float(candidate["start"]), float(other["start"])) for other in chosen):
                continue
            chosen.append(candidate)
            if len(chosen) == self.max_segments_per_speaker:
                break
        if len(chosen) < self.min_segments_per_speaker:
            return None
        vectors: list[list[float]] = []
        for segment in sorted(chosen, key=lambda item: float(item["start"])):
            start = max(0, min(audio.frame_count, round(float(segment["start"]) * audio.sample_rate)))
            end = max(start, min(audio.frame_count, round(float(segment["end"]) * audio.sample_rate)))
            if end - start < round(self.min_segment_seconds * audio.sample_rate):
                continue
            samples = audio.samples[start:end]
            if not samples:
                continue
            rms = math.sqrt(sum(float(value) ** 2 for value in samples) / len(samples))
            if rms < 0.008:
                continue
            crop = AudioBuffer(array.array("f", samples), audio.sample_rate, 1,
                               audio.source_sample_rate or audio.sample_rate, 1,
                               (end - start) / audio.sample_rate)
            vectors.append(l2_normalize(self.extractor.extract(crop)))
        if len(vectors) < self.min_segments_per_speaker:
            return None
        return aggregate_embeddings(vectors), len(vectors)

    def match(self, audio: AudioBuffer, segments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        model_version = getattr(self.extractor, "model_version", None)
        if not model_version:
            raise EngineConfigError("声纹提取器缺少明确的 model_version")
        profiles: list[tuple[str, str, list[float]]] = []
        for person in self.database.list_speakers():
            profile = self.database.get_speaker_profile(person["id"])
            if (not profile or profile.get("model_version") != model_version
                    or not profile.get("embedding")):
                continue
            profiles.append((person["id"], person["name"], l2_normalize(profile["embedding"])))

        local_ids = list(dict.fromkeys(item["local_speaker_id"] for item in segments
                                       if item.get("local_speaker_id")))
        result: dict[str, dict[str, Any]] = {}
        for local_id in local_ids:
            local_profile = self._local_embedding(audio, segments, local_id)
            ranked: list[tuple[float, str, str]] = []
            if local_profile is not None:
                embedding, sample_count = local_profile
                for speaker_id, name, profile_embedding in profiles:
                    if len(embedding) != len(profile_embedding):
                        continue
                    similarity = sum(a * b for a, b in zip(embedding, profile_embedding))
                    ranked.append((similarity, speaker_id, name))
            ranked.sort(key=lambda item: (-item[0], item[1]))
            best = ranked[0] if ranked else None
            runner_up = ranked[1] if len(ranked) > 1 else None
            margin = best[0] - runner_up[0] if best and runner_up else None
            accepted = bool(best and best[0] >= self.min_similarity and
                            (margin is None or margin >= self.min_margin))
            result[local_id] = {
                "local_speaker_id": local_id,
                "registered_speaker_id": best[1] if accepted and best else None,
                "speaker_name": best[2] if accepted and best else f"未知说话人 {len(result) + 1}",
                "similarity": best[0] if best else None,
                "margin": margin,
                "match_status": "matched" if accepted else "unknown",
                "model_version": model_version,
                "clean_sample_count": local_profile[1] if local_profile is not None else 0,
            }
        return result
