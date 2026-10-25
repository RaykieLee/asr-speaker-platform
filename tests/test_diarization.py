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
from backend.app.main import _diarization_cluster_threshold
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


def test_diarization_cluster_threshold_setting_defaults_and_parses(monkeypatch):
    monkeypatch.delenv("ASR_DIARIZATION_CLUSTER_THRESHOLD", raising=False)
    assert _diarization_cluster_threshold() == 0.5

    monkeypatch.setenv("ASR_DIARIZATION_CLUSTER_THRESHOLD", "0.65")
    assert _diarization_cluster_threshold() == 0.65


@pytest.mark.parametrize("value", ["invalid", "nan", "inf", "0", "1", "-0.1", "1.1"])
def test_diarization_cluster_threshold_setting_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("ASR_DIARIZATION_CLUSTER_THRESHOLD", value)
    with pytest.raises(EngineConfigError, match="ASR_DIARIZATION_CLUSTER_THRESHOLD"):
        _diarization_cluster_threshold()


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


def _stereo_levels(seconds: float, left: float, right: float) -> AudioBuffer:
    frames = round(seconds * 16_000)
    samples = array.array("f")
    for _ in range(frames):
        samples.extend((left, right))
    return AudioBuffer(samples, 16_000, 2, 16_000, 2, seconds)


def test_separate_empty_diarization_uses_nonoverlapping_voiced_asr_windows_for_match():
    class ChannelRecognizer:
        model_version = "asr-test"

        def __init__(self):
            self.calls = []

        def transcribe(self, samples, sample_rate, hotwords=None):
            self.calls.append((len(samples), sample_rate, hotwords))
            return "left transcript" if samples[0] > 0.25 else "right transcript"

    class LeftOnlyDiarizer:
        def segment(self, audio):
            if audio.samples[0] > 0.25:
                return [{"start": 0.0, "end": audio.duration,
                         "local_speaker_id": "speaker_0"}]
            return []

    class RegisteredMatcher:
        def __init__(self):
            self.calls = []

        def match(self, audio, segments):
            local_id = segments[0]["local_speaker_id"]
            self.calls.append((audio.samples[0], segments))
            is_right = audio.samples[0] <= 0.25
            return {local_id: {
                "local_speaker_id": local_id,
                "registered_speaker_id": "speaker-b" if is_right else "speaker-a",
                "speaker_name": "B" if is_right else "A",
                "match_status": "matched",
                "similarity": 0.9,
                "margin": 0.2,
                "clean_sample_count": len(segments),
            }}

    recognizer, matcher = ChannelRecognizer(), RegisteredMatcher()
    result = FileASRService(
        recognizer, diarizer=LeftOnlyDiarizer(), speaker_matcher=matcher,
    ).transcribe_audio(_stereo_levels(5.6, 0.3, 0.2), channel_mode="separate")

    assert len(recognizer.calls) == 2  # ASR ran once per channel; fallback did not rerun it.
    fallback_calls = [segments for _, segments in matcher.calls
                      if segments[0]["local_speaker_id"] == "fallback_speaker_0"]
    assert len(fallback_calls) == 1
    windows = fallback_calls[0]
    assert len(windows) >= 2
    assert all(item["end"] - item["start"] >= 1.0 for item in windows)
    assert all(left["end"] <= right["start"]
               for left, right in zip(windows, windows[1:]))
    assert [segment.text for segment in result.segments] == ["left transcript", "right transcript"]
    assert [segment.local_speaker_id for segment in result.segments] == [
        "channel_0:speaker_0", "channel_1:fallback_speaker_0",
    ]
    assert [segment.registered_speaker_id for segment in result.segments] == [
        "speaker-a", "speaker-b",
    ]
    assert result.segments[1].metadata["speaker_fallback"] == "asr_energy_windows"
    assert result.segments[1].metadata["channel_index"] == 1


def test_separate_empty_diarization_does_not_match_silence_or_weak_evidence():
    class FixedRecognizer:
        model_version = "asr-test"

        def __init__(self, text):
            self.text = text

        def transcribe(self, samples, sample_rate, hotwords=None):
            return self.text

    class EmptyDiarizer:
        def segment(self, audio):
            return []

    class NeverMatch:
        min_segment_seconds = 1.0
        min_segments_per_speaker = 2
        max_segments_per_speaker = 5

        def __init__(self):
            self.calls = 0

        def match(self, audio, segments):
            self.calls += 1
            raise AssertionError("insufficient speech must not reach the matcher")

    # Even if ASR emits text on silence, no voice window means no speaker assignment.
    matcher = NeverMatch()
    silent = FileASRService(FixedRecognizer("hallucinated words"), diarizer=EmptyDiarizer(),
                            speaker_matcher=matcher).transcribe_audio(
        _stereo_levels(3.0, 0.0, 0.0), channel_mode="separate",
    )
    assert silent.text == "hallucinated words hallucinated words"
    assert all(segment.local_speaker_id is None and segment.registered_speaker_id is None
               for segment in silent.segments)
    assert not silent.speaker_mappings

    # A single 1.5 second voice run cannot satisfy the matcher's two-window gate.
    matcher = NeverMatch()
    short = FileASRService(FixedRecognizer("short speech"), diarizer=EmptyDiarizer(),
                           speaker_matcher=matcher).transcribe_audio(
        _stereo_levels(1.5, 0.0, 0.2), channel_mode="separate",
    )
    assert short.text == "short speech short speech"
    assert all(segment.local_speaker_id is None for segment in short.segments)
    assert not short.speaker_mappings
    assert matcher.calls == 0


def test_separate_empty_diarization_preserves_asr_text_without_enough_voiced_windows():
    class FixedRecognizer:
        model_version = "asr-test"

        def transcribe(self, samples, sample_rate, hotwords=None):
            return "short but audible speech" if samples[-1] > 0.008 else ""

    class EmptyDiarizer:
        def segment(self, audio):
            return []

    class MustRemainUnknownMatcher:
        min_segment_seconds = 1.0
        min_segments_per_speaker = 2
        max_segments_per_speaker = 5

        def match(self, audio, segments):
            raise AssertionError("only one clean window is not enough evidence")

    result = FileASRService(
        FixedRecognizer(), diarizer=EmptyDiarizer(), speaker_matcher=MustRemainUnknownMatcher(),
    ).transcribe_audio(_stereo_levels(1.5, 0.0, 0.2), channel_mode="separate")

    assert result.text == "short but audible speech"
    right = [segment for segment in result.segments
             if segment.metadata.get("channel_index") == 1]
    assert len(right) == 1
    assert right[0].text == "short but audible speech"
    assert right[0].local_speaker_id is None
    assert right[0].registered_speaker_id is None
    assert not result.speaker_mappings


def test_separate_fallback_keeps_sufficient_but_ambiguous_voice_unknown():
    class FixedRecognizer:
        model_version = "asr-test"

        def transcribe(self, samples, sample_rate, hotwords=None):
            return "voice words" if samples[0] > 0.008 else ""

    class EmptyDiarizer:
        def segment(self, audio):
            return []

    class UnknownMatcher:
        def match(self, audio, segments):
            local_id = segments[0]["local_speaker_id"]
            return {local_id: {
                "local_speaker_id": local_id, "registered_speaker_id": None,
                "speaker_name": "未知说话人 1", "match_status": "unknown",
                "similarity": 0.52, "margin": 0.01,
                "clean_sample_count": len(segments),
            }}

    result = FileASRService(
        FixedRecognizer(), diarizer=EmptyDiarizer(), speaker_matcher=UnknownMatcher(),
    ).transcribe_audio(_stereo_levels(5.6, 0.0, 0.2), channel_mode="separate")

    assert result.text == "voice words"
    assert len(result.speaker_mappings) == 1
    mapping = result.speaker_mappings[0]
    assert mapping["match_status"] == "unknown"
    assert mapping["registered_speaker_id"] is None
    right = [segment for segment in result.segments
             if segment.metadata.get("channel_index") == 1]
    assert right[0].local_speaker_id == "channel_1:fallback_speaker_0"
    assert right[0].registered_speaker_id is None


def test_separate_fallback_rejected_match_does_not_leak_registered_name():
    class FixedRecognizer:
        model_version = "asr-test"

        def transcribe(self, samples, sample_rate, hotwords=None):
            return "voice words" if samples[0] > 0.008 else ""

    class EmptyDiarizer:
        def segment(self, audio):
            return []

    class TooFewSamplesMatcher:
        def match(self, audio, segments):
            local_id = segments[0]["local_speaker_id"]
            return {local_id: {
                "local_speaker_id": local_id,
                "registered_speaker_id": "speaker-b",
                "speaker_name": "真人姓名乙",
                "match_status": "matched",
                "similarity": 0.99,
                "margin": 0.5,
                "clean_sample_count": 1,
            }}

    result = FileASRService(
        FixedRecognizer(), diarizer=EmptyDiarizer(), speaker_matcher=TooFewSamplesMatcher(),
    ).transcribe_audio(_stereo_levels(5.6, 0.0, 0.2), channel_mode="separate")

    mapping = next(item for item in result.speaker_mappings
                   if item["local_speaker_id"] == "channel_1:fallback_speaker_0")
    right = [segment for segment in result.segments
             if segment.metadata.get("channel_index") == 1]
    assert mapping["match_status"] == "unknown"
    assert mapping["registered_speaker_id"] is None
    assert mapping["speaker_name"] == "未知说话人 1"
    assert right[0].registered_speaker_id is None
    assert right[0].metadata["speaker_name"] == "未知说话人 1"
    assert "真人姓名乙" not in repr(result)


def test_separate_fallback_matcher_error_is_diagnostic_but_nonfatal():
    class FixedRecognizer:
        model_version = "asr-test"

        def transcribe(self, samples, sample_rate, hotwords=None):
            return "voice words" if samples[0] > 0.008 else ""

    class EmptyDiarizer:
        def segment(self, audio):
            return []

    class BrokenMatcher:
        def match(self, audio, segments):
            raise RuntimeError("sensitive internal details")

    result = FileASRService(
        FixedRecognizer(), diarizer=EmptyDiarizer(), speaker_matcher=BrokenMatcher(),
    ).transcribe_audio(_stereo_levels(5.6, 0.0, 0.2), channel_mode="separate")

    right = [segment for segment in result.segments
             if segment.metadata.get("channel_index") == 1]
    assert result.text == "voice words"
    assert right[0].text == "voice words"
    assert right[0].metadata["speaker_fallback_error"] == "RuntimeError"
    assert "sensitive internal details" not in repr(result)


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
