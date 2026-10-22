"""Real sherpa-onnx offline diarization adapter.

The adapter returns only model-derived boundaries and stable, session-local
speaker IDs. It never fabricates a speaker count or a transcript.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.app.audio.core import AudioBuffer, normalize_audio
from backend.app.engines.base import EngineConfigError


class SherpaOfflineDiarizer:
    """Lazily loaded wrapper for sherpa-onnx ``OfflineSpeakerDiarization``."""

    def __init__(self, diarization: Any):
        self._diarization = diarization
        self.sample_rate = int(diarization.sample_rate)
        if self.sample_rate <= 0:
            raise EngineConfigError("sherpa 说话人分段模型返回了无效采样率")

    @classmethod
    def load(cls, segmentation_model: str | Path, embedding_model: str | Path, *,
             num_threads: int = 2, provider: str = "cpu", num_speakers: int = -1,
             cluster_threshold: float = 0.5, min_duration_on: float = 0.3,
             min_duration_off: float = 0.5) -> "SherpaOfflineDiarizer":
        segmentation_path = Path(segmentation_model)
        embedding_path = Path(embedding_model)
        missing = [str(path) for path in (segmentation_path, embedding_path) if not path.is_file()]
        if missing:
            raise EngineConfigError("离线说话人分段模型文件缺失: " + ", ".join(missing))
        if num_threads < 1:
            raise EngineConfigError("说话人分段 num_threads 必须是正整数")
        if num_speakers == 0 or num_speakers < -1:
            raise EngineConfigError("num_speakers 必须为 -1 或正整数")
        if not 0.0 < cluster_threshold < 1.0:
            raise EngineConfigError("cluster_threshold 必须在 0 到 1 之间")
        if min_duration_on < 0 or min_duration_off < 0:
            raise EngineConfigError("min_duration_on/off 不能为负数")
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise EngineConfigError("未安装 sherpa-onnx，无法执行离线说话人分段") from exc
        try:
            config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                        model=str(segmentation_path), window_shift_ratio=0.1,
                    ),
                    num_threads=num_threads, provider=provider,
                ),
                embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=str(embedding_path), num_threads=num_threads, provider=provider,
                ),
                clustering=sherpa_onnx.FastClusteringConfig(
                    num_clusters=num_speakers, threshold=cluster_threshold,
                ),
                min_duration_on=min_duration_on,
                min_duration_off=min_duration_off,
            )
            if not config.validate():
                raise EngineConfigError("离线说话人分段配置无效")
            return cls(sherpa_onnx.OfflineSpeakerDiarization(config))
        except EngineConfigError:
            raise
        except Exception as exc:
            raise EngineConfigError(f"创建 sherpa 离线说话人分段器失败: {exc}") from exc

    def segment(self, audio: AudioBuffer) -> list[dict[str, Any]]:
        normalized = normalize_audio(audio, sample_rate=self.sample_rate)
        try:
            result = self._diarization.process(normalized.samples)
            raw_segments = result.sort_by_start_time()
            ordered_ids: dict[int, str] = {}
            segments: list[dict[str, Any]] = []
            for item in raw_segments:
                start = max(0.0, min(normalized.duration, float(item.start)))
                end = max(start, min(normalized.duration, float(item.end)))
                raw_id = int(item.speaker)
                local_id = ordered_ids.setdefault(raw_id, f"speaker_{len(ordered_ids)}")
                if end <= start:
                    continue
                segments.append({
                    "start": start,
                    "end": end,
                    "local_speaker_id": local_id,
                    "overlap": False,
                    "metadata": {"diarization_model": "sherpa-onnx-offline"},
                })
            for index, current in enumerate(segments):
                current["overlap"] = any(
                    other["local_speaker_id"] != current["local_speaker_id"]
                    and min(current["end"], other["end"]) > max(current["start"], other["start"])
                    for other_index, other in enumerate(segments) if index != other_index
                )
            return segments
        except Exception as exc:
            raise EngineConfigError(f"sherpa 离线说话人分段失败: {exc}") from exc
