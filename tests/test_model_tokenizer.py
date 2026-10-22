from pathlib import Path

import pytest

from backend.app.hotwords.compiler import HotwordCompilationError
from backend.app.hotwords.model_tokenizer import load_vocabulary, mixed_cjk_bpe_tokenizer


def test_load_vocabulary_and_reject_invalid_lines(tmp_path: Path):
    path = tmp_path / "tokens.txt"
    path.write_text("张 0\n▁HE 1\nLL 2\nO 3\n", encoding="utf-8")
    assert load_vocabulary(path) == {"张", "▁HE", "LL", "O"}
    path.write_text("broken\n", encoding="utf-8")
    with pytest.raises(ValueError, match="无效"):
        load_vocabulary(path)


def test_mixed_tokenizer_uses_model_sentencepiece(tmp_path: Path, monkeypatch):
    class FakeProcessor:
        def __init__(self, model_file):
            assert model_file == str(tmp_path / "bpe.model")

        def encode(self, text, out_type):
            assert out_type is str
            return {"HELLO": ["▁HE", "LL", "O"]}[text]

    import sys
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, "sentencepiece", SimpleNamespace(SentencePieceProcessor=FakeProcessor))
    tokenize = mixed_cjk_bpe_tokenizer(tmp_path / "bpe.model")
    assert tokenize("张三 Hello") == ["张", "三", "▁HE", "LL", "O"]
    with pytest.raises(HotwordCompilationError):
        tokenize("张三/Hello")
