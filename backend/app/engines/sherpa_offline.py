"""Optional sherpa-onnx adapter for offline Transducer recognition."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import EngineConfigError, ModelArtifact


@dataclass(frozen=True)
class OfflineTransducerFiles:
    encoder: Path
    decoder: Path
    joiner: Path
    tokens: Path

    def require_files(self) -> None:
        missing = [str(path) for path in (self.encoder, self.decoder, self.joiner, self.tokens)
                   if not path.is_file()]
        if missing:
            raise EngineConfigError("离线 ASR 模型文件缺失: " + ", ".join(missing))


class SherpaOfflineRecognizer:
    """A loaded offline recognizer; model configuration never fabricates output."""

    def __init__(self, recognizer: Any, *, model_version: str | None = None):
        self._recognizer = recognizer
        self.model_version = model_version
        self.decoding_method = "modified_beam_search"

    @classmethod
    def load(cls, model: ModelArtifact, files: OfflineTransducerFiles, *,
             num_threads: int = 2, provider: str = "cpu",
             decoding_method: str = "modified_beam_search",
             hotwords_score: float = 1.5,
             modeling_unit: str = "cjkchar",
             bpe_vocab: str | Path | None = None) -> "SherpaOfflineRecognizer":
        model.require_files()
        files.require_files()
        if model.architecture != "transducer":
            raise EngineConfigError("离线 sherpa adapter 仅支持 Transducer 架构")
        if decoding_method not in {"greedy_search", "modified_beam_search"}:
            raise EngineConfigError(f"离线 Transducer 不支持解码方式: {decoding_method}")
        if not modeling_unit.strip():
            raise EngineConfigError("必须指定 Transducer modeling_unit")
        if "bpe" in modeling_unit.lower() and (bpe_vocab is None or not Path(bpe_vocab).is_file()):
            raise EngineConfigError(
                f"modeling_unit={modeling_unit!r} 需要存在且可读的 bpe_vocab 文件"
            )
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise EngineConfigError("未安装 sherpa-onnx，无法执行离线识别") from exc
        try:
            recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(files.encoder), decoder=str(files.decoder),
                joiner=str(files.joiner), tokens=str(files.tokens),
                num_threads=num_threads, sample_rate=16_000, feature_dim=80,
                decoding_method=decoding_method, provider=provider,
                hotwords_score=hotwords_score,
                modeling_unit=modeling_unit,
                bpe_vocab=str(bpe_vocab) if bpe_vocab is not None else "",
            )
        except Exception as exc:
            raise EngineConfigError(f"创建 sherpa 离线识别器失败: {exc}") from exc
        adapter = cls(recognizer, model_version=model.version)
        adapter.decoding_method = decoding_method
        return adapter

    def transcribe(self, samples: Any, sample_rate: int,
                   hotwords: list[str | list[str]] | None = None) -> str:
        if sample_rate != 16_000:
            raise EngineConfigError("离线识别输入必须为 16000 Hz")
        if hotwords and self.decoding_method != "modified_beam_search":
            raise EngineConfigError("离线 Transducer 热词要求 modified_beam_search")
        encoded_hotwords = self._format_hotwords(hotwords) if hotwords else None
        try:
            stream = (self._recognizer.create_stream(hotwords=encoded_hotwords)
                      if encoded_hotwords else self._recognizer.create_stream())
            stream.accept_waveform(sample_rate, samples)
            self._recognizer.decode_stream(stream)
            result = stream.result
        except Exception as exc:
            raise RuntimeError(f"sherpa 离线识别失败: {exc}") from exc
        return str(getattr(result, "text", result)).strip()

    @staticmethod
    def _format_hotwords(hotwords: list[str | list[str]]) -> str:
        """Encode tokenized phrases in sherpa's ``TOKEN TOKEN/OTHER`` format."""
        phrases: list[str] = []
        for hotword in hotwords:
            if isinstance(hotword, str):
                tokens = hotword.split() if any(char.isspace() for char in hotword) else list(hotword)
            else:
                tokens = hotword
            clean_tokens = [str(token).strip() for token in tokens]
            if not clean_tokens or any(not token for token in clean_tokens):
                raise EngineConfigError("热词必须包含一个或多个非空模型 token")
            if any("/" in token or any(char.isspace() for char in token)
                   for token in clean_tokens):
                raise EngineConfigError("热词 token 含有动态热词格式保留字符")
            phrases.append(" ".join(clean_tokens))
        if not phrases:
            raise EngineConfigError("热词列表不能为空")
        return "/".join(phrases)
