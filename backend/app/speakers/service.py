"""Speaker enrollment service with explicit model and sample quality contracts."""
from __future__ import annotations

import io
import math
import shutil
import subprocess
import tempfile
import uuid
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from backend.app.audio import AudioBuffer, decode_audio_file, decode_wav, normalize_audio
from backend.app.storage import FileCleanupError, FileStore, SQLiteStore, StorageService


class SpeakerError(RuntimeError):
    """Base class for speaker enrollment failures."""


class MissingSpeakerModelError(SpeakerError):
    """A real speaker embedding model is not configured or available."""


class EmbeddingModelVersionError(SpeakerError):
    """Stored samples/profile were produced by a different model version."""


class SpeakerNotFoundError(SpeakerError):
    """Requested speaker identity does not exist."""


class SampleQualityError(SpeakerError):
    """Sample failed enrollment quality requirements."""

    def __init__(self, message: str, quality: dict[str, float | bool]):
        self.quality = quality
        super().__init__(message)


class SampleUploadTooLargeError(SpeakerError):
    """Enrollment upload exceeded the configured byte limit."""


class SampleDurationTooLongError(SpeakerError):
    """Enrollment audio exceeded the configured duration limit."""


class EmbeddingExtractor(Protocol):
    model_version: str

    def extract(self, audio: AudioBuffer) -> Sequence[float]: ...


@dataclass(frozen=True)
class SpeakerModel:
    path: Path
    version: str
    num_threads: int = 1
    provider: str = "cpu"


class SherpaSpeakerEmbeddingAdapter:
    """Thin lazy-loaded wrapper around sherpa-onnx SpeakerEmbeddingExtractor."""

    def __init__(self, model: SpeakerModel):
        self.model = model
        self.model_version = model.version
        if not self.model.version.strip():
            raise MissingSpeakerModelError("声纹模型版本必须显式配置")
        if not model.path.is_file():
            raise MissingSpeakerModelError(f"声纹模型文件缺失: {model.path}")
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise MissingSpeakerModelError("未安装 sherpa-onnx，无法提取真实声纹向量") from exc
        try:
            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(model.path), num_threads=model.num_threads,
                debug=False, provider=model.provider,
            )
            if not config.validate():
                raise MissingSpeakerModelError(f"声纹模型配置无效: {model.path}")
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        except MissingSpeakerModelError:
            raise
        except Exception as exc:
            raise MissingSpeakerModelError(f"加载声纹模型失败: {exc}") from exc

    def extract(self, audio: AudioBuffer) -> Sequence[float]:
        try:
            import numpy as np
        except ImportError as exc:
            raise MissingSpeakerModelError("未安装 numpy，无法调用 sherpa-onnx 声纹模型") from exc
        if audio.channels != 1:
            raise ValueError("声纹提取要求单声道音频")
        stream = self._extractor.create_stream()
        try:
            stream.accept_waveform(sample_rate=audio.sample_rate,
                                   waveform=np.ascontiguousarray(audio.samples, dtype=np.float32))
            stream.input_finished()
            embedding = self._extractor.compute(stream)
            if embedding is None or len(embedding) == 0:
                raise SampleQualityError("声纹模型未能从该样本提取有效向量", {})
            return [float(value) for value in embedding]
        finally:
            release = getattr(stream, "release", None)
            if release is not None:
                release()


def l2_normalize(vector: Sequence[float]) -> list[float]:
    """Return a finite unit-length embedding; reject empty/invalid model output."""
    values = [float(value) for value in vector]
    if not values or any(not math.isfinite(value) for value in values):
        raise SpeakerError("声纹模型返回了空向量或非有限数值")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        raise SpeakerError("声纹模型返回了零向量")
    return [value / norm for value in values]


def aggregate_embeddings(embeddings: Sequence[Sequence[float]]) -> list[float]:
    """Average unit-normalized embeddings and normalize the resulting profile."""
    if not embeddings:
        raise SpeakerError("没有可用于聚合的声纹样本")
    normalized = [l2_normalize(v) for v in embeddings]
    dimension = len(normalized[0])
    if any(len(item) != dimension for item in normalized):
        raise SpeakerError("声纹模型返回的向量维度不一致")
    mean = [sum(item[i] for item in normalized) / len(normalized) for i in range(dimension)]
    return l2_normalize(mean)


def assess_audio_quality(audio: AudioBuffer, *, min_duration_seconds: float = 1.0,
                         frame_seconds: float = 0.03, silence_rms: float = 0.008,
                         clipping_level: float = 0.999, max_clipped_ratio: float = 0.01
                         ) -> dict[str, float | bool]:
    """Estimate speech presence and clipping from short-time energy.

    This deterministic energy gate catches silence and unusably short audio. It is
    not a speaker-count/noise classifier; deployments can add a VAD-quality model.
    """
    if audio.channels != 1:
        raise SampleQualityError("声纹样本必须是单声道", {"duration_seconds": audio.duration})
    frame_length = max(1, round(audio.sample_rate * frame_seconds))
    total_frames = audio.frame_count
    if total_frames == 0:
        quality = {"duration_seconds": 0.0, "speech_seconds": 0.0,
                   "speech_ratio": 0.0, "rms": 0.0, "clipped_ratio": 0.0,
                   "is_silent": True}
        raise SampleQualityError("样本为空或没有可检测语音", quality)
    voiced_frames = 0
    energy = 0.0
    clipped = 0
    for value in audio.samples:
        energy += value * value
        if abs(value) >= clipping_level:
            clipped += 1
    rms = math.sqrt(energy / total_frames)
    for start in range(0, total_frames, frame_length):
        block = audio.samples[start:min(start + frame_length, total_frames)]
        block_rms = math.sqrt(sum(value * value for value in block) / len(block))
        if block_rms >= silence_rms:
            voiced_frames += len(block)
    speech_seconds = voiced_frames / audio.sample_rate
    clipped_ratio = clipped / total_frames
    quality: dict[str, float | bool] = {
        "duration_seconds": audio.duration,
        "speech_seconds": speech_seconds,
        "speech_ratio": voiced_frames / total_frames,
        "rms": rms,
        "clipped_ratio": clipped_ratio,
        "is_silent": speech_seconds == 0.0,
    }
    if speech_seconds < min_duration_seconds:
        reason = "静音样本" if speech_seconds == 0 else "有效语音过短"
        raise SampleQualityError(f"{reason}：至少需要 {min_duration_seconds:g} 秒有效语音", quality)
    if clipped_ratio > max_clipped_ratio:
        raise SampleQualityError("样本削波过多，请降低录音音量后重录", quality)
    return quality


class SpeakerRegistrationService:
    """CRUD, sample enrollment and profile maintenance over the shared stores."""

    def __init__(self, database: SQLiteStore, files: FileStore,
                 extractor: EmbeddingExtractor | None = None, *,
                 storage: StorageService | None = None,
                 min_duration_seconds: float = 1.0,
                 max_sample_bytes: int = 64 * 1024 * 1024,
                 max_sample_duration_seconds: float = 120.0):
        if max_sample_bytes < 1:
            raise ValueError("max_sample_bytes must be positive")
        if not math.isfinite(max_sample_duration_seconds) or max_sample_duration_seconds <= 0:
            raise ValueError("max_sample_duration_seconds must be positive and finite")
        self.database = database
        self.files = files
        self.storage = storage or StorageService(database, files)
        self.extractor = extractor
        self.min_duration_seconds = min_duration_seconds
        self.max_sample_bytes = max_sample_bytes
        self.max_sample_duration_seconds = max_sample_duration_seconds

    def _require_model(self) -> EmbeddingExtractor:
        if self.extractor is None:
            raise MissingSpeakerModelError("声纹模型未配置；请先安装并配置 sherpa-onnx 声纹模型")
        return self.extractor

    def create_speaker(self, name: str) -> dict[str, Any]:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("人员姓名不能为空")
        return self.database.create_speaker(clean_name)

    def rename_speaker(self, speaker_id: str, name: str) -> dict[str, Any]:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("人员姓名不能为空")
        with self.database.transaction() as db:
            cur = db.execute("UPDATE speakers SET name=?,updated_at=datetime('now') WHERE id=?",
                             (clean_name, speaker_id))
            if not cur.rowcount:
                raise SpeakerNotFoundError(speaker_id)
        return self.get_speaker(speaker_id)

    def get_speaker(self, speaker_id: str) -> dict[str, Any]:
        speaker = self.database.get_speaker(speaker_id)
        if speaker is None:
            raise SpeakerNotFoundError(speaker_id)
        samples = self.database.list_samples(speaker_id)
        speaker["samples"] = samples
        speaker["sample_count"] = len(samples)
        speaker["profile"] = self.database.get_speaker_profile(speaker_id)
        speaker["registration_status"] = "registered" if speaker["profile"] else "incomplete"
        return speaker

    def list_speakers(self) -> list[dict[str, Any]]:
        result = []
        for row in self.database.list_speakers():
            samples = self.database.list_samples(row["id"])
            profile = self.database.get_speaker_profile(row["id"])
            row.update(sample_count=len(samples), registration_status="registered" if profile else "incomplete")
            result.append(row)
        return result

    def delete_speaker(self, speaker_id: str) -> bool:
        return self.storage.delete_speaker(speaker_id)

    @staticmethod
    def _decode_upload(data: bytes, filename: str) -> AudioBuffer:
        try:
            if filename.lower().endswith(".wav") or data[:4] == b"RIFF":
                return normalize_audio(decode_wav(data))
            suffix = Path(filename).suffix or ".audio"
            with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
                temporary.write(data)
                temporary.flush()
                return decode_audio_file(temporary.name)
        except Exception as exc:
            if isinstance(exc, SampleQualityError):
                raise
            raise SampleQualityError(f"无法解码音频样本：{exc}", {}) from exc

    @staticmethod
    def _probe_upload_duration(data: bytes, filename: str) -> float:
        """Read duration metadata before decoding all audio samples.

        WAV duration is available from its header. Other supported formats use
        ffprobe, which ships with the FFmpeg installation used by the decoder.
        """
        try:
            if filename.lower().endswith(".wav") or data[:4] == b"RIFF":
                with wave.open(io.BytesIO(data), "rb") as wav:
                    rate = wav.getframerate()
                    if rate < 1:
                        raise ValueError("invalid sample rate")
                    return wav.getnframes() / rate
            ffprobe = shutil.which("ffprobe")
            if not ffprobe:
                raise SampleQualityError(
                    "检查非 WAV 样本时长需要 ffprobe；请安装 FFmpeg 后重试", {}
                )
            suffix = Path(filename).suffix or ".audio"
            with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
                temporary.write(data)
                temporary.flush()
                result = subprocess.run(
                    [ffprobe, "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", temporary.name],
                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    timeout=15,
                )
            duration = float(result.stdout.decode("ascii").strip())
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("invalid duration")
            return duration
        except SampleQualityError:
            raise
        except (wave.Error, EOFError, OSError, ValueError, subprocess.SubprocessError) as exc:
            raise SampleQualityError(f"无法预先检查音频时长：{exc}", {}) from exc

    def add_sample(self, speaker_id: str, filename: str, data: bytes) -> dict[str, Any]:
        if len(data) > self.max_sample_bytes:
            raise SampleUploadTooLargeError(
                f"声纹样本超过 {self.max_sample_bytes} 字节限制"
            )
        duration = self._probe_upload_duration(data, filename)
        if duration > self.max_sample_duration_seconds:
            raise SampleDurationTooLongError(
                f"声纹样本时长超过 {self.max_sample_duration_seconds:g} 秒限制"
            )
        if self.database.get_speaker(speaker_id) is None:
            raise SpeakerNotFoundError(speaker_id)
        extractor = self._require_model()
        audio = self._decode_upload(data, filename)
        quality = assess_audio_quality(audio, min_duration_seconds=self.min_duration_seconds)
        current_profile = self.database.get_speaker_profile(speaker_id)
        samples = self.database.list_samples(speaker_id)
        versions = {sample["model_version"] for sample in samples if sample["model_version"]}
        if current_profile and current_profile["model_version"] != extractor.model_version:
            raise EmbeddingModelVersionError(
                f"当前声纹模型版本 {extractor.model_version} 与已注册版本 {current_profile['model_version']} 不一致；需用保留样本统一重算"
            )
        if versions and versions != {extractor.model_version}:
            raise EmbeddingModelVersionError("人员样本包含不同声纹模型版本，不能混合聚合")
        embedding = l2_normalize(extractor.extract(audio))
        if current_profile and len(current_profile["embedding"]) != len(embedding):
            raise EmbeddingModelVersionError("新旧声纹向量维度不同，不能聚合")
        all_embeddings = [sample["embedding"] for sample in samples if sample["embedding"] is not None]
        all_embeddings.append(embedding)
        profile = aggregate_embeddings(all_embeddings)
        sample_id = f"smp_{uuid.uuid4().hex}"
        suffix = Path(filename).suffix.lower()
        if len(suffix) > 10 or not suffix[1:].isalnum():
            suffix = ".audio"
        relative_path = f"speakers/{speaker_id}/{sample_id}{suffix}"
        self.files.write(relative_path, data)
        try:
            sample = self.database.create_sample(
                speaker_id, relative_path, sample_id=sample_id,
                duration_seconds=audio.duration, quality=quality,
                model_version=extractor.model_version, embedding=embedding,
            )
            self.database.set_speaker_profile(
                speaker_id, profile, sample_count=len(all_embeddings),
                model_version=extractor.model_version,
            )
        except BaseException:
            self.files.delete(relative_path)
            raise
        return {"sample": sample, "profile": self.database.get_speaker_profile(speaker_id),
                "quality": quality, "registration_status": "registered"}

    def delete_sample(self, speaker_id: str, sample_id: str) -> bool:
        sample = self.database.get_sample(sample_id)
        if sample is None or sample["speaker_id"] != speaker_id:
            return False
        deleted = self.storage.delete_sample(sample_id)
        if not deleted:
            return False
        remaining = self.database.list_samples(speaker_id)
        if not remaining:
            return True
        versions = {item["model_version"] for item in remaining}
        if len(versions) != 1 or None in versions:
            raise EmbeddingModelVersionError("保留样本模型版本不一致或缺失，无法安全重算人员声纹")
        embeddings = [item["embedding"] for item in remaining if item["embedding"] is not None]
        if len(embeddings) != len(remaining):
            raise SpeakerError("保留样本缺少向量，无法安全重算人员声纹")
        profile = aggregate_embeddings(embeddings)
        self.database.set_speaker_profile(speaker_id, profile, sample_count=len(embeddings),
                                          model_version=next(iter(versions)))
        return True
