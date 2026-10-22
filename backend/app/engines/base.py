"""Small dependency-free interfaces that remain testable without model weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class EngineConfigError(RuntimeError):
    """An engine configuration cannot be honored as requested."""


class MissingModelError(EngineConfigError):
    """A required model artifact is missing or incomplete."""


class HotwordsUnsupportedError(EngineConfigError):
    """The selected recognizer cannot apply native decoding hotwords."""


@dataclass(frozen=True)
class ModelArtifact:
    model_id: str
    architecture: str
    version: str | None
    files: tuple[Path, ...]

    def require_files(self) -> tuple[Path, ...]:
        missing = [str(path) for path in self.files if not path.is_file()]
        if missing:
            version = self.version or "版本待固定"
            raise MissingModelError(
                f"模型 {self.model_id} ({version}) 文件缺失: {', '.join(missing)}。"
                "请按 models/README.md 固定版本与 SHA-256 后下载。"
            )
        return self.files


class RecognizerAdapter:
    """Engine-neutral construction seam; concrete inference is added by ASR tasks."""

    def __init__(self, model: ModelArtifact):
        self.model = model

    def validate_request(self, *, hotwords: bool = False) -> None:
        from .capabilities import capabilities_for

        capabilities_for(self.model.architecture).require(hotwords=hotwords)

    def validate_model(self) -> tuple[Path, ...]:
        return self.model.require_files()
