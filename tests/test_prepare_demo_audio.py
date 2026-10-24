from __future__ import annotations

import hashlib
import json
import shutil
import wave
from pathlib import Path

import pytest

from scripts.prepare_demo_audio import (
    DEFAULT_ASSET_DIR,
    concatenate_wavs,
    prepare_assets,
    transcript_text,
)


def _write_wav(path: Path, *, rate: int = 8000, frames: bytes = b"\x01\x00" * 800) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(frames)


def test_reference_text_comes_from_alternating_character_pinyin_tokens() -> None:
    assert transcript_text("不 bu4 如 ru2 一 yi2 块儿 kuair4 玩儿 wanr2 吧 ba5") == "不如一块儿玩儿吧"
    with pytest.raises(ValueError):
        transcript_text("不 bu4 如")


def test_concatenation_preserves_pcm_and_records_exact_clip_intervals(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    _write_wav(first, frames=b"\x01\x00" * 5)
    _write_wav(second, frames=b"\x02\x00" * 3)
    output = tmp_path / "joined.wav"
    intervals = concatenate_wavs([("a", first), ("b", second)], output, silence_ms=1)

    with wave.open(str(output), "rb") as joined:
        assert (joined.getnchannels(), joined.getsampwidth(), joined.getframerate()) == (1, 2, 8000)
        assert joined.getnframes() == 5 + 8 + 3
        pcm = joined.readframes(joined.getnframes())
    assert pcm[:10] == b"\x01\x00" * 5
    assert pcm[10:26] == b"\x00" * 16
    assert pcm[26:] == b"\x02\x00" * 3
    assert intervals == [
        {"clip_id": "a", "start_seconds": 0.0, "end_seconds": 0.000625,
         "source_clip_duration_seconds": 0.000625, "silence_after_seconds": 0.001},
        {"clip_id": "b", "start_seconds": 0.001625, "end_seconds": 0.002,
         "source_clip_duration_seconds": 0.000375, "silence_after_seconds": 0.0},
    ]


def test_concatenation_rejects_incompatible_pcm(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    _write_wav(first, rate=8000)
    _write_wav(second, rate=16000)
    with pytest.raises(ValueError, match="identical PCM"):
        concatenate_wavs([("a", first), ("b", second)], tmp_path / "out.wav", 350)


def test_offline_rebuild_matches_checked_in_assets_and_manifest(tmp_path: Path) -> None:
    source_assets = DEFAULT_ASSET_DIR
    target = tmp_path / "aishell3"
    shutil.copytree(source_assets / "source", target / "source")
    shutil.copy2(source_assets / "source_selection.json", target / "source_selection.json")

    generated = prepare_assets(target)
    for relative in (
        "registration/speaker-a.wav",
        "registration/speaker-b.wav",
        "two-speaker-abab.wav",
        "manifest.json",
    ):
        expected = (source_assets / relative).read_bytes()
        actual = (target / relative).read_bytes()
        assert hashlib.sha256(actual).hexdigest() == hashlib.sha256(expected).hexdigest()
    assert generated["demo_audio"]["turn_order"] == ["A", "B", "A", "B"]
    assert [segment["reference_text"] for segment in generated["demo_audio"]["segments"]] == [
        "你别那样好吗", "桌上是什么", "你能看到她吗", "不如一块儿玩儿吧",
    ]
    checked_manifest = json.loads((source_assets / "manifest.json").read_text(encoding="utf-8"))
    assert generated == checked_manifest
