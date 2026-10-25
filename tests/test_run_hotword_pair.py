from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_hotword_pair import _format_hotwords, parse_hotword, run_pair, score_hypothesis


def test_parse_hotword_requires_finite_positive_trailing_score():
    assert parse_hotword("客流量:1.25") == ("客流量", 1.25)
    assert parse_hotword("OpenAI") == ("OpenAI", 1.0)
    assert parse_hotword("客流量:20") == ("客流量", 20.0)
    for value in (" :1", "客流量:NaN", "客流量:0", "客流量:-1", "客流量:20.01", "客流量:oops"):
        with pytest.raises(ValueError):
            parse_hotword(value)


def test_hotword_encoding_keeps_score_as_final_separate_item():
    encoded = _format_hotwords([
        {"tokens": ["商", "场"], "score": 1.0},
        {"tokens": ["▁SALE", "S"], "score": 2.5},
    ])
    assert encoded == ["商 场 :1", "▁SALE S :2.5"]


def test_model_level_metrics_count_cer_and_present_or_absent_targets():
    result = score_hypothesis("商场客流量", "商场客流", ["商场", "家电"])
    assert result["cer"] == {"errors": 1, "reference_characters": 5, "value": 0.2}
    assert result["hotwords"]["target_hits"] == 1
    assert result["hotwords"]["targets_present_in_reference"] == 1
    assert result["hotwords"]["false_insertions"] == 0
    assert result["hotwords"]["targets_absent_from_reference"] == 1


def test_run_pair_reuses_identical_samples_writes_raw_and_hashed_manifest(tmp_path, monkeypatch):
    import numpy as np
    import sys
    from types import SimpleNamespace

    wav = tmp_path / "clip.wav"
    annotation = tmp_path / "reference.json"
    out = tmp_path / "results"
    wav.write_bytes(b"synthetic test audio")
    fake_soundfile = SimpleNamespace(
        info=lambda _path: SimpleNamespace(samplerate=16_000, channels=1, duration=0.2),
        read=lambda _path, dtype, always_2d: (np.zeros(3200, dtype=np.float32), 16_000),
    )
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)
    annotation.write_text(json.dumps({
        "segments": [{"start": 0, "end": 0.2, "text": "商场客流量"}],
        "hotword_targets": ["商场", "家电"],
        "source": {"dataset": "AISHELL-4"},
    }, ensure_ascii=False), encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in ("encoder-epoch-20-avg-1.int8.onnx", "decoder-epoch-20-avg-1.int8.onnx",
                 "joiner-epoch-20-avg-1.int8.onnx", "tokens.txt", "bpe.model"):
        (model_dir / name).write_bytes(name.encode())
    monkeypatch.setattr("scripts.run_hotword_pair._compile_hotwords",
                        lambda _dir, _args: ([{"phrase": "商场", "tokens": ["商", "场"], "score": 1.0},
                                              {"phrase": "家电", "tokens": ["家", "电"], "score": 1.0}],
                                             ["商 场 :1", "家 电 :1"]))
    observed = []

    class FakeStream:
        def __init__(self, phrase):
            self.phrase = phrase
            self.result = type("Result", (), {"text": "商场客流量"})()

        def accept_waveform(self, rate, samples):
            observed.append((rate, samples.copy(), self.phrase))

    class FakeRecognizer:
        def create_stream(self, hotwords=None):
            return FakeStream(hotwords)

        def decode_stream(self, _stream):
            pass

    tick = iter([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    result = run_pair(wav, annotation, out, model_dir=model_dir,
                      hotword_args=["商场", "家电"],
                      recognizer_factory=lambda *_args, **_kwargs: FakeRecognizer(),
                      now=lambda: next(tick))
    assert len(observed) == 2
    assert np.array_equal(observed[0][1], observed[1][1])
    assert [row[2] for row in observed] == [None, "商 场 :1/家 电 :1"]
    assert out.stat().st_mode & 0o777 == 0o700
    assert (out / "enabled.raw.json").stat().st_mode & 0o777 == 0o600
    assert json.loads((out / "disabled.raw.json").read_text()) == {
        "mode": "disabled", "text": "商场客流量"}
    assert result["inputs"]["wav_sha256"]
    assert result["model"]["files_sha256"]["tokens.txt"]
    saved_manifest = json.loads((out / "manifest.json").read_text())
    assert "text" not in saved_manifest["results"]["enabled"]
    with pytest.raises(FileExistsError):
        run_pair(wav, annotation, out, model_dir=model_dir, hotword_args=["商场", "家电"],
                 recognizer_factory=lambda *_args, **_kwargs: FakeRecognizer())


def test_reference_rejects_overlap_and_pair_does_not_flatten_ambiguous_order(tmp_path):
    from scripts.run_hotword_pair import _reference

    with pytest.raises(ValueError, match="overlapping"):
        _reference({"segments": [{"start": 0, "end": 2, "text": "甲"},
                                  {"start": 1, "end": 3, "text": "乙"}]})
