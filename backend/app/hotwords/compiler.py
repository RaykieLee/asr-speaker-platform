"""Compile editable hotword entries into validated sherpa-onnx snapshots.

The tokenizer is injected by the model integration layer. A BPE model must
provide its own vocabulary-backed tokenizer; guessing byte-pair units would
silently produce unusable hotwords.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from backend.app.engines import HotwordsUnsupportedError, capabilities_for


class HotwordCompilationError(ValueError):
    """A phrase could not be represented by the configured model tokenizer."""


class HotwordVocabularyUnavailableError(HotwordCompilationError):
    """No model vocabulary was configured, so phrase coverage cannot be checked."""


Tokenizer = Callable[[str], Sequence[str]]


class HotwordCompiler:
    """Model-bound compiler for the sherpa-onnx contextual-biasing interface.

    ``tokenizer`` must be the tokenizer used by the selected model. For BPE
    models it must use that model's BPE vocabulary; for CJK character models
    it may emit one character token per code point. The generated ``text`` is
    suitable for sherpa's one-hotword-per-line hotwords file; ``tokens`` keep
    the validated token sequence available for runtime integrations and audit.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        architecture: str = "transducer",
        decoding_method: str = "modified_beam_search",
        mode: str = "offline",
        tokenizer_id: str = "model-tokenizer",
        modeling_unit: str = "cjkchar",
        vocabulary: Iterable[str] | None = None,
    ) -> None:
        if not callable(tokenizer):
            raise TypeError("tokenizer must be a callable using the selected model vocabulary")
        self.tokenizer = tokenizer
        self.architecture = architecture
        self.decoding_method = decoding_method
        self.mode = mode
        self.tokenizer_id = tokenizer_id
        if modeling_unit not in {"cjkchar", "bpe", "cjkchar+bpe"}:
            raise ValueError(f"不支持的模型建模单元: {modeling_unit}")
        self.modeling_unit = modeling_unit
        self.vocabulary = frozenset(vocabulary) if vocabulary is not None else None

    def compile(self, entries: Iterable[dict[str, Any]], weight: float) -> dict[str, Any]:
        if self.decoding_method != "modified_beam_search":
            raise HotwordsUnsupportedError(
                "原生热词仅支持 Transducer modified_beam_search 解码；"
                f"当前解码方法为 {self.decoding_method!r}。"
            )
        capabilities_for(self.architecture).require(hotwords=True, mode=self.mode)
        if self.vocabulary is None:
            if "bpe" in self.modeling_unit:
                raise HotwordVocabularyUnavailableError(
                    "BPE 热词需要对应模型的 BPE vocabulary；未配置时不能编译或声称词表覆盖。"
                )
            raise HotwordVocabularyUnavailableError(
                "热词编译需要所选模型的 token 词表以验证词条覆盖。"
            )
        compiled_entries: list[dict[str, Any]] = []
        seen: set[str] = set()
        for entry in entries:
            if not entry.get("enabled", True):
                continue
            source_word = str(entry["word"])
            phrases = [source_word, *(str(alias) for alias in entry.get("aliases", []))]
            for phrase in phrases:
                normalized = phrase.strip()
                key = normalized.casefold()
                if not normalized or key in seen:
                    continue
                if "/" in normalized:
                    raise HotwordCompilationError(
                        f"热词 {normalized!r} 包含保留分隔符 '/'，不能用于动态 stream 热词。"
                    )
                seen.add(key)
                try:
                    tokens = tuple(str(token) for token in self.tokenizer(normalized))
                except Exception as exc:
                    raise HotwordCompilationError(
                        f"热词 {normalized!r} 无法通过模型分词器编码: {exc}"
                    ) from exc
                if not tokens or any(not token for token in tokens):
                    raise HotwordCompilationError(
                        f"热词 {normalized!r} 没有有效 token；请检查模型词表覆盖。"
                    )
                if any("/" in token or any(char.isspace() for char in token) for token in tokens):
                    raise HotwordCompilationError(
                        f"热词 {normalized!r} 的模型 token 含有动态热词格式保留字符。"
                    )
                missing = [token for token in tokens if token not in self.vocabulary]
                if missing:
                    raise HotwordCompilationError(
                        f"热词 {normalized!r} 包含模型词表未覆盖的 token: {', '.join(missing)}。"
                    )
                compiled_entries.append({
                    "text": normalized,
                    "tokens": list(tokens),
                    "source_word": source_word,
                    "language": entry.get("language"),
                })
        if not compiled_entries:
            raise HotwordCompilationError("热词集中没有可启用的词条。")
        return {
            "format": "sherpa-onnx-hotwords-v1",
            "architecture": self.architecture,
            "decoding_method": self.decoding_method,
            "mode": self.mode,
            "tokenizer_id": self.tokenizer_id,
            "modeling_unit": self.modeling_unit,
            "weight": float(weight),
            "hotwords_file_content": "\n".join(item["text"] for item in compiled_entries) + "\n",
            "stream_hotwords": "/".join(" ".join(item["tokens"]) for item in compiled_entries),
            "entries": compiled_entries,
        }
