"""Cross-process worker integration using a fake recognizer and SQLite only."""
from __future__ import annotations

import asyncio
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.app.jobs.service import JobQueueFull, TranscriptionService
from backend.app.jobs.worker import FileWorkerLock, JobWorker, WorkerAlreadyRunning
from backend.app.main import app
from backend.app.storage import FileStore, SQLiteStore


class FakeFileASR:
    model_version = "fake-process-model"
    diarizer = None

    def transcribe_file(self, path, *, hotwords=None, channel_mode="mixdown", progress=None):
        assert Path(path).is_file()
        if progress:
            progress(0.5)
        return SimpleNamespace(
            duration=1.0, text="fake result", segments=(),
            speaker_diarization="disabled", speaker_mappings=(),
        )


def _run_one_worker(database_path: str, files_root: str, result_queue) -> None:
    database = SQLiteStore(database_path)
    database.initialize()
    files = FileStore(files_root)
    worker = JobWorker(database, files, FakeFileASR())
    try:
        result_queue.put(worker.run_once())
    finally:
        worker.close()
        database.close()


def _claim_from_process(database_path: str, job_id: str, result_queue) -> None:
    database = SQLiteStore(database_path)
    database.initialize()
    try:
        result_queue.put(database.claim_job(job_id) is not None)
    finally:
        database.close()


def _hold_process_lock(lock_path: str, ready, release) -> None:
    lock = FileWorkerLock(Path(lock_path))
    lock.acquire()
    ready.set()
    release.wait(10)
    lock.release()


def test_external_api_queues_job_and_separate_process_recovers_and_finishes(tmp_path):
    database_path = tmp_path / "platform.sqlite3"
    files_root = tmp_path / "files"
    database = SQLiteStore(database_path)
    database.initialize()
    files = FileStore(files_root)
    service = TranscriptionService(
        database, files, None, worker_mode="external", model_version="api-snapshot",
    )
    try:
        created = asyncio.run(service.create_job(b"fake audio bytes", "meeting.wav"))
        assert service.worker is None
        assert created["status"] == "queued"
        assert created["model_version"] == "api-snapshot"

        # Simulate a process exit after claim but before task completion.
        database.update_job(created["id"], status="running", progress=0.3)
        context = multiprocessing.get_context("spawn")
        result_queue = context.Queue()
        process = context.Process(target=_run_one_worker,
                                  args=(str(database_path), str(files_root), result_queue))
        process.start()
        process.join(15)
        assert not process.is_alive()
        assert process.exitcode == 0
        assert result_queue.get(timeout=2) is True
        completed = database.get_job(created["id"])
        assert completed["status"] == "succeeded"
        assert completed["result_metadata"]["text"] == "fake result"
    finally:
        service.close()
        database.close()


def test_sqlite_claim_is_single_winner_across_processes(tmp_path):
    database_path = tmp_path / "jobs.sqlite3"
    database = SQLiteStore(database_path)
    database.initialize()
    job = database.create_job()
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [context.Process(target=_claim_from_process,
                                 args=(str(database_path), job["id"], result_queue))
                 for _ in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(15)
        assert not process.is_alive()
        assert process.exitcode == 0
    results = [result_queue.get(timeout=2) for _ in processes]
    assert sorted(results) == [False, True]
    assert database._connection.execute("PRAGMA user_version").fetchone()[0] == 2
    assert database.get_job(job["id"])["status"] == "running"
    database.close()


def test_worker_process_lock_rejects_second_worker(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    lock_path = tmp_path / "file-worker.lock"
    process = context.Process(target=_hold_process_lock,
                              args=(str(lock_path), ready, release))
    process.start()
    try:
        assert ready.wait(10)
        with pytest.raises(WorkerAlreadyRunning, match="其他进程"):
            FileWorkerLock(lock_path).acquire()
    finally:
        release.set()
        process.join(10)
    assert process.exitcode == 0


def test_external_api_rejects_full_queue_atomically(tmp_path):
    database = SQLiteStore(tmp_path / "jobs.sqlite3")
    database.initialize()
    files = FileStore(tmp_path / "files")
    service = TranscriptionService(database, files, None, worker_mode="external",
                                   max_pending_jobs=1)
    try:
        first = asyncio.run(service.create_job(b"first", "first.wav"))
        assert database.get_job(first["id"])["status"] == "queued"
        with pytest.raises(JobQueueFull):
            asyncio.run(service.create_job(b"second", "second.wav"))
        assert database.count_jobs(status="queued") == 1
        assert len(list(files.root.glob("transcriptions/*"))) == 1
    finally:
        service.close()
        database.close()


def test_external_api_lifespan_skips_file_model_and_worker_startup(monkeypatch, tmp_path):
    monkeypatch.setenv("ASR_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ASR_FILE_WORKER_MODE", "external")
    monkeypatch.setenv("ASR_PROCESS_ROLE", "api")
    model_dir = tmp_path / "offline-model"
    model_dir.mkdir()
    monkeypatch.setenv("ASR_OFFLINE_MODEL_DIR", str(model_dir))
    for name in ("ASR_ONLINE_MODEL_DIR", "ASR_SPEAKER_MODEL_PATH", "ASR_DIARIZATION_MODEL_PATH"):
        monkeypatch.delenv(name, raising=False)

    class FakeHotwords:
        def resolve(self, _set_id, *, mode):
            assert mode == "offline"
            return {"version": 1, "entries": ["fake"]}

    monkeypatch.setattr("backend.app.main._asr_paths",
                        lambda *_args, **_kwargs: (SimpleNamespace(), None, None))
    monkeypatch.setattr("backend.app.main._hotword_service",
                        lambda *_args, **_kwargs: FakeHotwords())
    monkeypatch.setattr("backend.app.main.SherpaOfflineRecognizer.load",
                        lambda *_args, **_kwargs: pytest.fail("API attempted to load offline ASR"))

    with TestClient(app) as client:
        service = app.state.transcription_service
        assert service.worker_mode == "external"
        assert service.asr is None
        assert service.worker is None
        assert service.model_version == "offline-model"
        response = client.post("/api/v1/transcriptions", files={
            "file": ("queued.wav", b"fake audio", "audio/wav"),
        })
        assert response.status_code == 202, response.text
        created = response.json()
        assert app.state.database.get_job(created["job_id"])["status"] == "queued"
        status = client.get("/api/v1/status").json()
        assert status["file_queue"]["queued"] == 1
        assert status["checks"]["file_worker"] is False
