"""Tokenize editable phrases with the exact vocabulary of a selected ASR model."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from backend.app.hotwords.compiler import HotwordCompilationError

_RUN = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9]+")


def load_vocabulary(tokens_path: str | Path) -> frozenset[str]:
    """Read sherpa's ``token id`` file, preserving its model token spelling."""
    lines = Path(tokens_path).read_text(encoding="utf-8").splitlines()
    result = set()
    for line in lines:
        token, separator, index = line.rpartition(" ")
        if not separator or not token or not index.isdecimal():
            raise ValueError(f"无效的模型 token 行: {line!r}")
        result.add(token)
    return frozenset(result)


def mixed_cjk_bpe_tokenizer(model_path: str | Path) -> Callable[[str], list[str]]:
    """Use one Han character per token and model SentencePiece for ASCII words.

    The mixed Chinese/English Zipformer models uppercase Latin input before BPE.
    Unsupported punctuation is rejected so callers cannot save a hotword whose
    tokenization would silently differ from the model's decoder.
    """
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("混合中英热词需要安装 sentencepiece") from exc
    processor = spm.SentencePieceProcessor(model_file=str(model_path))

    def tokenize(phrase: str) -> list[str]:
        tokens: list[str] = []
        position = 0
        for match in _RUN.finditer(phrase):
            gap = phrase[position:match.start()]
            if gap.strip():
                raise HotwordCompilationError(f"热词包含无法编码的字符: {gap!r}")
            part = match.group()
            if "\u3400" <= part[0] <= "\u9fff" and len(part) == 1:
                tokens.append(part)
            else:
                tokens.extend(processor.encode(part.upper(), out_type=str))
            position = match.end()
        if phrase[position:].strip():
            raise HotwordCompilationError(f"热词包含无法编码的字符: {phrase[position:]!r}")
        return tokens

    return tokenize
