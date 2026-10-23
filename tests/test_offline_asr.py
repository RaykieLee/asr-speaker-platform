"""Offline file ASR, durable jobs and upload API behavior."""
from __future__ import annotations

import array
import asyncio
import io
import sys
import threading
import wave
from types import SimpleNamespace
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.audio.core import AudioBuffer, AudioFormatError
from backend.app.engines.base import EngineConfigError, ModelArtifact
from backend.app.engines.sherpa_offline import OfflineTransducerFiles, SherpaOfflineRecognizer
from backend.app.api.transcriptions import router
from backend.app.jobs.service import TranscriptionService
from backend.app.pipelines.file_asr import FileASRService
from backend.app.storage import FileStore, SQLiteStore


def wav_bytes(duration: float = 0.25, sample_rate: int = 16_000) -> bytes:
    stream = io.BytesIO()
    frames = int(duration * sample_rate)
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((b"\x01\x00" * frames))
    return stream.getvalue()


def stereo_wav_bytes(duration: float = 0.5, sample_rate: int = 24_000) -> bytes:
    stream = io.BytesIO()
    frames = int(duration * sample_rate)
    interleaved = array.array("h")
    for _ in range(frames):
        interleaved.extend((16000, -12000))
    if sys.byteorder != "little":
        interleaved.byteswap()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(interleaved.tobytes())
    return stream.getvalue()


class FakeRecognizer:
    model_version = "fake-offline-v1"

    def __init__(self, texts=None, *, fail_count=0):
        self.texts = list(texts or ["recognized sentence"])
        self.fail_count = fail_count
        self.calls = []

    def transcribe(self, samples, sample_rate, hotwords=None):
        self.calls.append((len(samples), sample_rate, hotwords, samples[-1] if samples else None))
        if self.fail_count:
            self.fail_count -= 1
            raise RuntimeError("temporary model failure")
        return self.texts[min(len(self.calls) - 1, len(self.texts) - 1)]


def make_storage(tmp_path: Path):
    database = SQLiteStore(tmp_path / "db.sqlite3")
    database.initialize()
    files = FileStore(tmp_path / "objects")
    return database, files


def test_long_audio_is_chunked_deduplicated_and_keeps_tail_timestamp():
    rate = 16_000
    total_frames = round(2.45 * rate)
    samples = array.array("f", [0.0] * total_frames)
    samples[-1] = 0.75
    audio = AudioBuffer(samples, rate, 1, rate, 1, 2.45)
    recognizer = FakeRecognizer(["前句重复", "重复中句", "中句结尾"])
    progress = []
    result = FileASRService(recognizer, max_chunk_seconds=1.2,
                            overlap_seconds=0.5).transcribe_audio(
        audio, hotwords=["李子熙"], progress=progress.append,
    )

    assert len(recognizer.calls) == 3
    assert recognizer.calls[-1][3] == pytest.approx(0.75)
    assert all(call[1] == rate and call[2] == ["李子熙"] for call in recognizer.calls)
    assert progress == [pytest.approx(1 / 3), pytest.approx(2 / 3), 1.0]
    assert result.duration == pytest.approx(2.45)
    assert result.text == "前句重复中句结尾"
    assert result.segments[0].start == 0
    assert result.segments[-1].end == pytest.approx(2.45)
    assert all(segment.end >= segment.start for segment in result.segments)


def test_separate_stereo_transcribes_each_channel_with_source_timeline_and_index():
    recognizer = FakeRecognizer(["左声道内容", "右声道内容"])
    # Exercise the same decode-preserving path used by file jobs without a filesystem fixture.
    from backend.app.audio.core import decode_wav
    source = decode_wav(stereo_wav_bytes(duration=1001 / 24_000))
    result = FileASRService(recognizer).transcribe_audio(source, channel_mode="separate")

    assert len(recognizer.calls) == 2
    assert all(call[1] == 16_000 for call in recognizer.calls)
    assert recognizer.calls[0][0] == recognizer.calls[1][0] == 667
    assert recognizer.calls[0][3] > 0 and recognizer.calls[1][3] < 0
    assert result.duration == pytest.approx(1001 / 24_000)
    assert result.text == "左声道内容右声道内容"
    assert [segment.metadata["channel_index"] for segment in result.segments] == [0, 1]
    assert [segment.start for segment in result.segments] == [0, 0]
    assert [segment.end for segment in result.segments] == [
        pytest.approx(1001 / 24_000), pytest.approx(1001 / 24_000),
    ]
    assert all(segment.overlap for segment in result.segments)
    assert all(segment.metadata["cross_channel_overlap"] for segment in result.segments)


def test_separate_stereo_namespaces_diarization_ids_per_channel():
    from backend.app.audio.core import decode_wav

    class Diarizer:
        def segment(self, audio):
            return [{"start": 0.1, "end": 0.4, "local_speaker_id": "speaker_0"}]

    class Matcher:
        def match(self, audio, segments):
            label = "左侧人员" if audio.samples[0] > 0 else "右侧人员"
            return {"speaker_0": {
                "local_speaker_id": "speaker_0", "registered_speaker_id": None,
                "speaker_name": label, "match_status": "unknown", "clean_sample_count": 0,
            }}

    result = FileASRService(FakeRecognizer(["left", "right"]), diarizer=Diarizer(),
                            speaker_matcher=Matcher()).transcribe_audio(
        decode_wav(stereo_wav_bytes(duration=1.0, sample_rate=16_000)),
        channel_mode="separate",
    )

    assert [segment.local_speaker_id for segment in result.segments] == [
        "channel_0:speaker_0", "channel_1:speaker_0",
    ]
    assert [item["local_speaker_id"] for item in result.speaker_mappings] == [
        "channel_0:speaker_0", "channel_1:speaker_0",
    ]
    assert [segment.metadata["channel_index"] for segment in result.segments] == [0, 1]
    assert [segment.metadata["speaker_name"] for segment in result.segments] == ["左侧人员", "右侧人员"]
    assert all(segment.overlap and segment.metadata["cross_channel_overlap"]
               for segment in result.segments)


def test_separate_rejects_mono_with_explicit_error():
    from backend.app.audio.core import decode_wav

    with pytest.raises(AudioFormatError, match="需要恰好两个声道"):
        FileASRService(FakeRecognizer()).transcribe_audio(
            decode_wav(wav_bytes()), channel_mode="separate",
        )


def test_missing_offline_model_fails_without_fake_transcript(tmp_path):
    service = FileASRService(None)
    with pytest.raises(EngineConfigError, match="模型未配置"):
        service.transcribe_file(tmp_path / "audio.wav")


def test_missing_model_marks_upload_job_failed_without_transcript(tmp_path):
    database, files = make_storage(tmp_path)
    service = TranscriptionService(database, files, FileASRService(None))
    try:
        job = asyncio.run(service.create_job(wav_bytes(), "no-model.wav"))
        completed = service.worker.wait(job["id"], timeout=3)
        assert completed["status"] == "failed"
        assert completed["error_code"] == "model_unavailable"
        assert completed["error_message"]
        assert database.list_segments(job["id"]) == []
    finally:
        service.close()
        database.close()


def test_sherpa_adapter_validates_all_model_files_before_loading(tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    model = ModelArtifact("offline", "transducer", "v1", (model_path,))
    files = OfflineTransducerFiles(*(tmp_path / name for name in (
        "encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt")))
    with pytest.raises(EngineConfigError, match="模型文件缺失"):
        SherpaOfflineRecognizer.load(model, files)


def test_sherpa_adapter_uses_official_stream_hotword_interface(tmp_path, monkeypatch):
    paths = [tmp_path / name for name in ("model.onnx", "encoder.onnx", "decoder.onnx",
                                           "joiner.onnx", "tokens.txt")]
    for path in paths:
        path.touch()
    calls = {}

    class FakeStream:
        result = SimpleNamespace(text="real fake-boundary result")

        def accept_waveform(self, rate, samples):
            calls["audio"] = (rate, list(samples))

    class FakeRecognizer:
        def create_stream(self, hotwords=None):
            calls["hotwords"] = hotwords
            return FakeStream()

        def decode_stream(self, stream):
            calls["decoded"] = stream

    class Offline:
        @staticmethod
        def from_transducer(**kwargs):
            calls["configuration"] = kwargs
            return FakeRecognizer()

    monkeypatch.setitem(sys.modules, "sherpa_onnx", SimpleNamespace(OfflineRecognizer=Offline))
    model = ModelArtifact("offline", "transducer", "offline-v2", (paths[0],))
    files = OfflineTransducerFiles(*paths[1:])
    adapter = SherpaOfflineRecognizer.load(model, files, decoding_method="modified_beam_search")
    assert adapter.transcribe([0.1, -0.1], 16_000, ["李子熙", "项目名"]) == "real fake-boundary result"
    assert calls["hotwords"] == "李 子 熙/项 目 名"
    assert calls["audio"] == (16_000, [0.1, -0.1])
    assert calls["configuration"]["decoding_method"] == "modified_beam_search"
    assert calls["configuration"]["modeling_unit"] == "cjkchar"
    assert calls["configuration"]["bpe_vocab"] == ""
    assert adapter.model_version == "offline-v2"
    adapter.transcribe([0.1], 16_000, [["Open", "AI"], ["project", "name"]])
    assert calls["hotwords"] == "Open AI/project name"

    greedy = SherpaOfflineRecognizer(FakeRecognizer())
    greedy.decoding_method = "greedy_search"
    with pytest.raises(EngineConfigError, match="热词要求 modified_beam_search"):
        greedy.transcribe([0.0], 16_000, ["term"])


def test_bpe_model_requires_and_passes_its_vocabulary(tmp_path, monkeypatch):
    paths = [tmp_path / name for name in ("model.onnx", "encoder.onnx", "decoder.onnx",
                                           "joiner.onnx", "tokens.txt")]
    for path in paths:
        path.touch()
    model = ModelArtifact("offline", "transducer", "offline-bpe-v1", (paths[0],))
    files = OfflineTransducerFiles(*paths[1:])
    calls = {}

    class FakeRecognizer:
        pass

    class Offline:
        @staticmethod
        def from_transducer(**kwargs):
            calls.update(kwargs)
            return FakeRecognizer()

    monkeypatch.setitem(sys.modules, "sherpa_onnx", SimpleNamespace(OfflineRecognizer=Offline))
    with pytest.raises(EngineConfigError, match="需要存在且可读"):
        SherpaOfflineRecognizer.load(model, files, modeling_unit="cjkchar+bpe")
    assert calls == {}

    vocab = tmp_path / "bpe.vocab"
    vocab.write_text("▁张 0\n三 -1\n", encoding="utf-8")
    adapter = SherpaOfflineRecognizer.load(
        model, files, modeling_unit="cjkchar+bpe", bpe_vocab=vocab,
    )
    assert isinstance(adapter, SherpaOfflineRecognizer)
    assert calls["modeling_unit"] == "cjkchar+bpe"
    assert calls["bpe_vocab"] == str(vocab)


def test_offline_hotword_format_rejects_unrepresentable_tokens():
    with pytest.raises(EngineConfigError, match="保留字符"):
        SherpaOfflineRecognizer._format_hotwords([["hello/world"]])
    with pytest.raises(EngineConfigError, match="保留字符"):
        SherpaOfflineRecognizer._format_hotwords([["hello world"]])


def test_upload_persists_hotword_snapshot_and_query_returns_segments(tmp_path):
    database, files = make_storage(tmp_path)
    hotword_set = database.create_hotword_set(
        "names", ["李子熙"],
        compiled={"entries": [{"text": "李子熙", "tokens": ["李", "子", "熙"]}]},
    )

    async def resolve(set_id):
        assert set_id == hotword_set["set_id"]
        return database.get_hotword_version(set_id)

    recognizer = FakeRecognizer()
    asr = FileASRService(recognizer)
    service = TranscriptionService(database, files, asr, hotword_resolver=resolve)
    app = FastAPI()
    app.include_router(router)
    app.state.transcription_service = service
    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/transcriptions", files={
                "file": ("meeting.wav", wav_bytes(), "audio/wav"),
            }, data={"language": "zh", "channel_mode": "mixdown",
                    "hotword_set_id": hotword_set["set_id"]})
            assert response.status_code == 202, response.text
            created = response.json()
            job = service.worker.wait(created["job_id"], timeout=3)
            assert job["status"] == "succeeded"
            assert job["hotword_version"] == 1
            assert job["options"]["hotwords"] == [["李", "子", "熙"]]
            assert recognizer.calls[0][2] == [["李", "子", "熙"]]
            assert job["model_version"] == "fake-offline-v1"
            assert files.exists(job["file_path"])
            queried = client.get(created["status_url"])
            assert queried.status_code == 200
            result = queried.json()
            assert result["status"] == "succeeded"
            assert len(result["segments"]) == 1
            assert result["segments"][0]["start_seconds"] == 0
            assert result["segments"][0]["end_seconds"] == pytest.approx(0.25)
            assert result["result_metadata"]["speaker_diarization"] == "disabled"
    finally:
        service.close()
        database.close()


def test_stereo_separate_upload_persists_ordered_channel_metadata_and_hotwords(tmp_path):
    database, files = make_storage(tmp_path)
    hotword_set = database.create_hotword_set(
        "separate test", ["李子熙"],
        compiled={"entries": [{"text": "李子熙", "tokens": ["李", "子", "熙"]}]},
    )

    def resolve(set_id):
        assert set_id == hotword_set["set_id"]
        return database.get_hotword_version(set_id)

    recognizer = FakeRecognizer(["左声道内容", "右声道内容"])
    service = TranscriptionService(
        database, files, FileASRService(recognizer), hotword_resolver=resolve,
    )
    app = FastAPI()
    app.include_router(router)
    app.state.transcription_service = service
    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/transcriptions", files={
                "file": ("stereo.wav", stereo_wav_bytes(), "audio/wav"),
            }, data={"channel_mode": "separate", "hotword_set_id": hotword_set["set_id"]})
            assert response.status_code == 202, response.text
            created = response.json()
            completed = service.worker.wait(created["job_id"], timeout=3)
            assert completed["status"] == "succeeded"
            queried = client.get(created["status_url"]).json()
            assert queried["result_metadata"]["channel_mode"] == "separate"
            assert queried["hotword_set_version"] == 1
            assert "options" not in queried
            assert queried["hotword_version"] == 1
            assert [item["metadata"]["channel_index"] for item in queried["segments"]] == [0, 1]
            assert [item["start_seconds"] for item in queried["segments"]] == [0, 0]
            assert all(item["end_seconds"] == pytest.approx(0.5) for item in queried["segments"])
            assert [call[2] for call in recognizer.calls] == [
                [["李", "子", "熙"]], [["李", "子", "熙"]],
            ]
            export = client.get(f"/api/v1/transcriptions/{created['job_id']}/export?format=json")
            assert export.status_code == 200
            assert [item["metadata"]["channel_index"] for item in export.json()["segments"]] == [0, 1]
    finally:
        service.close()
        database.close()


def test_mono_separate_upload_fails_clearly_without_creating_segments(tmp_path):
    database, files = make_storage(tmp_path)
    service = TranscriptionService(database, files, FileASRService(FakeRecognizer()))
    app = FastAPI()
    app.include_router(router)
    app.state.transcription_service = service
    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/transcriptions", files={
                "file": ("mono.wav", wav_bytes(), "audio/wav"),
            }, data={"channel_mode": "separate"})
            assert response.status_code == 202, response.text
            completed = service.worker.wait(response.json()["job_id"], timeout=3)
            assert completed["status"] == "failed"
            assert completed["error_code"] == "invalid_audio"
            assert "需要恰好两个声道" in completed["error_message"]
            assert database.list_segments(completed["id"]) == []
    finally:
        service.close()
        database.close()


def test_failed_job_can_retry_and_running_job_recovers_on_restart(tmp_path):
    database, files = make_storage(tmp_path)
    recognizer = FakeRecognizer(fail_count=1)
    service = TranscriptionService(database, files, FileASRService(recognizer))
    try:
        first = asyncio.run(service.create_job(wav_bytes(), "retry.wav"))
        failed = service.worker.wait(first["id"], timeout=3)
        assert failed["status"] == "failed"
        assert failed["error_code"] == "recognition_error"
        retried = service.retry_job(first["id"])
        assert retried["status"] in {"queued", "running", "succeeded"}
        succeeded = service.worker.wait(first["id"], timeout=3)
        assert succeeded["status"] == "succeeded"
    finally:
        service.close()

    recovery_path = "transcriptions/interrupted.wav"
    files.write(recovery_path, wav_bytes())
    interrupted = database.create_job(file_path=recovery_path, status="running", options={})
    recovered_service = TranscriptionService(database, files, FileASRService(FakeRecognizer()))
    try:
        recovered = recovered_service.worker.wait(interrupted["id"], timeout=3)
        assert recovered["status"] == "succeeded"
        assert recovered["progress"] == 1
    finally:
        recovered_service.close()
        database.close()


def test_job_exposes_running_state_while_model_is_busy(tmp_path):
    database, files = make_storage(tmp_path)
    entered, release = threading.Event(), threading.Event()

    class BlockingRecognizer(FakeRecognizer):
        def transcribe(self, samples, sample_rate, hotwords=None):
            entered.set()
            assert release.wait(timeout=3)
            return "done"

    service = TranscriptionService(database, files, FileASRService(BlockingRecognizer()))
    try:
        job = asyncio.run(service.create_job(wav_bytes(), "busy.wav"))
        assert entered.wait(timeout=2)
        assert database.get_job(job["id"])["status"] == "running"
        release.set()
        assert service.worker.wait(job["id"], timeout=3)["status"] == "succeeded"
    finally:
        release.set()
        service.close()
        database.close()


def test_hotword_set_requires_a_resolver_and_unknown_job_is_404(tmp_path):
    database, files = make_storage(tmp_path)
    service = TranscriptionService(database, files, FileASRService(FakeRecognizer()))
    app = FastAPI()
    app.include_router(router)
    app.state.transcription_service = service
    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/transcriptions", files={
                "file": ("meeting.wav", wav_bytes(), "audio/wav"),
            }, data={"hotword_set_id": "missing"})
            assert response.status_code == 422
            assert response.json()["code"] == "hotword_snapshot_failed"
            assert client.get("/api/v1/transcriptions/unknown").status_code == 404
    finally:
        service.close()
        database.close()
