from __future__ import annotations

import array
import asyncio
import io
import sys
import wave
from types import SimpleNamespace

import pytest

from backend.app.audio.core import AudioBuffer
from backend.app.engines.base import EngineConfigError
from backend.app.jobs.service import TranscriptionService
from backend.app.pipelines.file_asr import FileASRService
from backend.app.speakers.diarization import SherpaOfflineDiarizer
from backend.app.speakers.matching import SpeakerMatcher
from backend.app.storage import FileStore, SQLiteStore


def audio(seconds=4.0):
    rate = 16000
    return AudioBuffer(array.array("f", [0.1] * round(seconds * rate)), rate, 1, rate, 1, seconds)


class FakeDiarizationResult:
    def sort_by_start_time(self):
        return [
            SimpleNamespace(start=0.0, end=2.0, speaker=4),
            SimpleNamespace(start=1.5, end=3.0, speaker=9),
            SimpleNamespace(start=3.0, end=4.0, speaker=4),
        ]


class FakeRecognizer:
    model_version = "asr-test"

    def __init__(self, text="same words"):
        self.text = text
        self.calls = []

    def transcribe(self, samples, sample_rate, hotwords=None):
        self.calls.append((len(samples), sample_rate, hotwords))
        return self.text


class FakeExtractor:
    model_version = "embedding-v1"

    def __init__(self, vector=(1.0, 0.0)):
        self.vector = vector
        self.durations = []

    def extract(self, clip):
        self.durations.append(clip.duration)
        return list(self.vector)


def test_sherpa_adapter_loads_official_config_and_stabilizes_ids(tmp_path, monkeypatch):
    segmentation, embedding = tmp_path / "seg.onnx", tmp_path / "emb.onnx"
    segmentation.touch()
    embedding.touch()
    calls = {}

    class Config:
        def __init__(self, **kwargs):
            calls["diarization_config"] = kwargs

        def validate(self):
            return True

    class Diarizer:
        sample_rate = 16000

        def __init__(self, config):
            calls["diarizer_config"] = config

        def process(self, samples):
            calls["samples"] = len(samples)
            return FakeDiarizationResult()

    fake = SimpleNamespace(
        OfflineSpeakerDiarizationConfig=Config,
        OfflineSpeakerSegmentationModelConfig=lambda **kw: ("segmentation", kw),
        OfflineSpeakerSegmentationPyannoteModelConfig=lambda **kw: ("pyannote", kw),
        SpeakerEmbeddingExtractorConfig=lambda **kw: ("embedding", kw),
        FastClusteringConfig=lambda **kw: ("clustering", kw),
        OfflineSpeakerDiarization=Diarizer,
    )
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake)
    diarizer = SherpaOfflineDiarizer.load(segmentation, embedding, num_speakers=2)
    result = diarizer.segment(audio())

    assert calls["diarization_config"]["segmentation"][0] == "segmentation"
    assert calls["diarization_config"]["embedding"][0] == "embedding"
    assert calls["diarization_config"]["clustering"][1]["num_clusters"] == 2
    assert [item["local_speaker_id"] for item in result] == ["speaker_0", "speaker_1", "speaker_0"]
    assert [item["overlap"] for item in result] == [True, True, False]
    assert result[-1]["end"] == 4.0


def test_diarization_missing_model_fails_explicitly(tmp_path):
    with pytest.raises(EngineConfigError, match="模型文件缺失"):
        SherpaOfflineDiarizer.load(tmp_path / "missing-seg.onnx", tmp_path / "missing-emb.onnx")


def test_speaker_matcher_aggregates_clean_segments_and_applies_model_and_margin_gates(tmp_path):
    database = SQLiteStore(tmp_path / "speakers.sqlite3")
    database.initialize()
    registered = database.create_speaker("已注册")
    database.set_speaker_profile(registered["id"], [1.0, 0.0], sample_count=2,
                                 model_version="embedding-v1")
    wrong_version = database.create_speaker("旧版本")
    database.set_speaker_profile(wrong_version["id"], [1.0, 0.0], sample_count=2,
                                 model_version="embedding-v0")
    source = audio(7)
    segments = [
        {"start": 0.0, "end": 1.5, "local_speaker_id": "speaker_0", "overlap": False},
        {"start": 2.0, "end": 3.5, "local_speaker_id": "speaker_0", "overlap": False},
        {"start": 4.0, "end": 5.5, "local_speaker_id": "speaker_0", "overlap": True},
        {"start": 4.5, "end": 6.0, "local_speaker_id": "speaker_1", "overlap": True},
    ]
    extractor = FakeExtractor()
    mapped = SpeakerMatcher(database, extractor, min_similarity=0.75).match(source, segments)
    assert mapped["speaker_0"]["registered_speaker_id"] == registered["id"]
    assert mapped["speaker_0"]["clean_sample_count"] == 2
    assert extractor.durations == [1.5, 1.5]
    assert mapped["speaker_1"]["registered_speaker_id"] is None
    assert mapped["speaker_1"]["match_status"] == "unknown"

    database.set_speaker_profile(registered["id"], [1.0, 0.0], sample_count=2,
                                 model_version="embedding-v0")
    version_filtered = SpeakerMatcher(database, FakeExtractor()).match(source, segments)
    assert version_filtered["speaker_0"]["registered_speaker_id"] is None
    database.close()


def test_matcher_uses_top_two_margin_and_reports_unknown_when_ambiguous(tmp_path):
    database = SQLiteStore(tmp_path / "ambiguous.sqlite3")
    database.initialize()
    for name, vector in (("one", [1.0, 0.0]), ("two", [0.99, 0.01])):
        person = database.create_speaker(name)
        database.set_speaker_profile(person["id"], vector, sample_count=2,
                                     model_version="embedding-v1")
    segments = [
        {"start": 0.0, "end": 1.5, "local_speaker_id": "speaker_0"},
        {"start": 2.0, "end": 3.5, "local_speaker_id": "speaker_0"},
    ]
    result = SpeakerMatcher(database, FakeExtractor(), min_similarity=0.5,
                            min_margin=0.05).match(audio(), segments)["speaker_0"]
    assert result["match_status"] == "unknown"
    assert result["registered_speaker_id"] is None
    assert result["margin"] < 0.05
    database.close()


def test_file_asr_transcribes_diarized_ranges_and_suppresses_duplicate_overlap_text():
    recognizer = FakeRecognizer()
    diarizer = SimpleNamespace(segment=lambda _: [
        {"start": 0.0, "end": 2.0, "local_speaker_id": "speaker_0"},
        {"start": 1.5, "end": 3.0, "local_speaker_id": "speaker_1"},
        {"start": 3.0, "end": 4.0, "local_speaker_id": "speaker_0"},
    ])
    matcher = SimpleNamespace(match=lambda *_: {
        "speaker_0": {"speaker_name": "Alice", "registered_speaker_id": "spk_a",
                      "match_status": "matched", "similarity": 0.9, "margin": 0.2},
        "speaker_1": {"speaker_name": "未知说话人 1", "registered_speaker_id": None,
                      "match_status": "unknown", "similarity": 0.3, "margin": 0.01},
    })
    service = FileASRService(recognizer, diarizer=diarizer, speaker_matcher=matcher)
    result = service.transcribe_audio(audio())
    assert len(recognizer.calls) == 3
    assert [item.local_speaker_id for item in result.segments] == [
        "speaker_0", "speaker_1", "speaker_0"]
    assert [item.overlap for item in result.segments] == [True, True, False]
    assert result.segments[0].text == "same words"
    assert result.segments[1].text == ""
    assert result.segments[1].metadata["duplicate_overlap_text_suppressed"] is True
    assert result.segments[0].registered_speaker_id == "spk_a"
    assert result.segments[-1].end == 4.0
    assert result.speaker_diarization == "completed"


def test_file_asr_without_diarizer_marks_disabled_not_pending():
    result = FileASRService(FakeRecognizer()).transcribe_audio(audio(1.0))
    assert result.speaker_diarization == "disabled"


def test_empty_diarization_falls_back_to_plain_transcription():
    """A diarizer miss must not discard otherwise recognizable audio."""
    recognizer = FakeRecognizer("speech that the diarizer missed")
    diarizer = SimpleNamespace(segment=lambda _: [])

    result = FileASRService(recognizer, diarizer=diarizer).transcribe_audio(audio(2.0))

    assert result.text == "speech that the diarizer missed"
    assert result.segments
    assert result.segments[0].local_speaker_id is None
    assert result.speaker_diarization == "no_speech"


def test_job_persists_diarization_status_and_mapping(tmp_path):
    samples = array.array("h", [1000] * 16000)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(samples.tobytes())
    diarizer = SimpleNamespace(segment=lambda _: [
        {"start": 0.0, "end": 1.0, "local_speaker_id": "speaker_0"},
    ])
    matcher = SimpleNamespace(match=lambda *_: {
        "speaker_0": {"local_speaker_id": "speaker_0", "registered_speaker_id": None,
                      "speaker_name": "未知说话人 1", "match_status": "unknown",
                      "similarity": None, "margin": None, "clean_sample_count": 0},
    })
    database = SQLiteStore(tmp_path / "jobs.sqlite3")
    database.initialize()
    files = FileStore(tmp_path / "objects")
    service = TranscriptionService(
        database, files,
        FileASRService(FakeRecognizer(), diarizer=diarizer, speaker_matcher=matcher),
    )
    try:
        job = asyncio.run(service.create_job(buffer.getvalue(), "diarized.wav"))
        completed = service.worker.wait(job["id"], timeout=3)
        assert completed["status"] == "succeeded"
        metadata = completed["result_metadata"]
        assert metadata["speaker_diarization"] == "completed"
        assert metadata["speaker_mappings"][0]["match_status"] == "unknown"
        assert metadata["speaker_mappings"][0]["registered_speaker_id"] is None
        assert "pending_t09" not in str(metadata)
    finally:
        service.close()
        database.close()
