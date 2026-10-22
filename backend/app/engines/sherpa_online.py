"""Optional sherpa-onnx adapter for online Transducer recognition."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import EngineConfigError, ModelArtifact


@dataclass(frozen=True)
class OnlineTransducerFiles:
    encoder: Path
    decoder: Path
    joiner: Path
    tokens: Path

    def require_files(self) -> None:
        missing = [str(path) for path in (self.encoder, self.decoder, self.joiner, self.tokens) if not path.is_file()]
        if missing:
            raise EngineConfigError("在线 ASR 模型文件缺失: " + ", ".join(missing))


class SherpaOnlineRecognizer:
    """A shared loaded recognizer. Each caller gets a distinct sherpa stream."""

    def __init__(self, recognizer: Any):
        self._recognizer = recognizer

    @classmethod
    def load(cls, model: ModelArtifact, files: OnlineTransducerFiles, *,
             num_threads: int = 2, provider: str = "cpu",
             decoding_method: str = "modified_beam_search",
             hotwords_score: float = 1.5,
             modeling_unit: str = "cjkchar",
             bpe_vocab: Path | None = None) -> "SherpaOnlineRecognizer":
        model.require_files()
        files.require_files()
        if model.architecture != "transducer":
            raise EngineConfigError("在线 sherpa adapter 仅支持 Transducer 架构")
        if decoding_method not in {"greedy_search", "modified_beam_search"}:
            raise EngineConfigError(f"在线 Transducer 不支持解码方式: {decoding_method}")
        if modeling_unit not in {"cjkchar", "bpe", "cjkchar+bpe"}:
            raise EngineConfigError(f"在线 Transducer 不支持建模单元: {modeling_unit}")
        if "bpe" in modeling_unit and (bpe_vocab is None or not bpe_vocab.is_file()):
            raise EngineConfigError("BPE 热词必须配置与当前模型匹配的 bpe_vocab 文件")
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise EngineConfigError("未安装 sherpa-onnx，无法启动在线识别") from exc
        try:
            recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                encoder=str(files.encoder), decoder=str(files.decoder),
                joiner=str(files.joiner), tokens=str(files.tokens),
                num_threads=num_threads, sample_rate=16_000, feature_dim=80,
                decoding_method=decoding_method, provider=provider,
                hotwords_score=hotwords_score, modeling_unit=modeling_unit,
                bpe_vocab=str(bpe_vocab) if bpe_vocab else "",
            )
        except Exception as exc:
            raise EngineConfigError(f"创建 sherpa 在线识别器失败: {exc}") from exc
        return cls(recognizer)

    def create_stream(self, hotwords: list[str] | None = None) -> Any:
        if hotwords and not hasattr(self._recognizer, "create_stream"):
            raise EngineConfigError("当前 sherpa-onnx 不支持 create_stream 热词接口")
        if hotwords:
            if any("/" in item or "\n" in item for item in hotwords):
                raise EngineConfigError("热词 token 序列不能包含 '/' 或换行")
            return self._recognizer.create_stream(hotwords="/".join(hotwords))
        return self._recognizer.create_stream()

    def is_ready(self, stream: Any) -> bool:
        return bool(self._recognizer.is_ready(stream))

    def decode_stream(self, stream: Any) -> None:
        self._recognizer.decode_stream(stream)

    def get_result(self, stream: Any) -> str:
        result = self._recognizer.get_result(stream)
        return str(getattr(result, "text", result))

    def is_endpoint(self, stream: Any) -> bool:
        return bool(self._recognizer.is_endpoint(stream))
