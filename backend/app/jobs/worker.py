"""Thread-pool worker for durable transcription jobs."""
from __future__ import annotations

import fcntl
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from datetime import datetime, timezone
from typing import Any

from backend.app.audio.core import AudioFormatError, FFmpegUnavailableError
from backend.app.engines.base import EngineConfigError
from backend.app.pipelines.file_asr import FileASRService
from backend.app.storage import FileStore, SQLiteStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class WorkerAlreadyRunning(RuntimeError):
    """Raised when another embedded or external file worker owns the queue."""


class FileWorkerLock:
    """Advisory process lock shared by embedded and external workers."""

    def __init__(self, path):
        self.path = path
        self._fd: int | None = None

    def acquire(self) -> None:
        if self._fd is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise WorkerAlreadyRunning("文件任务 Worker 已由其他进程启动") from exc
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()}\n".encode())
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)
        self._fd = None

    @classmethod
    def is_locked(cls, path) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
                return False
        finally:
            os.close(fd)


class JobWorker:
    """Runs queued rows and requeues interrupted ``running`` rows on startup."""

    def __init__(self, database: SQLiteStore, files: FileStore, asr: FileASRService, *,
                 max_workers: int = 1):
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.database = database
        self.files = files
        self.asr = asr
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="asr-job")
        self._lock = threading.RLock()
        self._futures: dict[str, Future[None]] = {}
        self._started = False
        self._closed = False
        self._recovered = False
        self._process_lock = FileWorkerLock(files.root.parent / "file-worker.lock")

    def start(self) -> None:
        """Resume durable work after process startup; safe to call once per worker."""
        with self._lock:
            if self._started:
                return
            if self._closed:
                raise RuntimeError("job worker is closed")
            self._started = True
        try:
            self._process_lock.acquire()
            self._recover_interrupted_jobs()
        except BaseException:
            with self._lock:
                self._started = False
            self._process_lock.release()
            raise
        for job in reversed(self.database.list_jobs(status="queued", limit=100_000)):
            self.enqueue(job["id"])

    def _recover_interrupted_jobs(self) -> None:
        if not self._recovered:
            self.database.recover_running_jobs()
            self._recovered = True

    def enqueue(self, job_id: str) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("job worker is closed")
            previous = self._futures.get(job_id)
            if previous is not None and not previous.done():
                return
            self._futures[job_id] = self._executor.submit(self._run, job_id)

    def wait(self, job_id: str, timeout: float | None = None) -> dict[str, Any] | None:
        with self._lock:
            future = self._futures.get(job_id)
        if future is not None:
            try:
                future.result(timeout=timeout)
            except TimeoutError:
                pass
        return self.database.get_job(job_id)

    def retry(self, job_id: str) -> dict[str, Any] | None:
        job = self.database.get_job(job_id)
        if job is None or job["status"] != "failed":
            return None
        self.database.update_job(job_id, status="queued", progress=0, error_code="",
                                 error_message="", started_at="", finished_at="")
        self.enqueue(job_id)
        return self.database.get_job(job_id)

    def close(self, *, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=not wait)
        if wait:
            self._process_lock.release()
            return
        with self._lock:
            pending = [future for future in self._futures.values() if not future.done()]
        if not pending:
            self._process_lock.release()
            return
        callback_lock = threading.Lock()
        remaining = len(pending)

        def release_after_last(_future: Future[None]) -> None:
            nonlocal remaining
            with callback_lock:
                remaining -= 1
                release = remaining == 0
            if release:
                self._process_lock.release()

        for future in pending:
            future.add_done_callback(release_after_last)

    def run_once(self) -> bool:
        """Claim and process one row in a standalone worker process."""
        self._process_lock.acquire()
        try:
            self._recover_interrupted_jobs()
            job = self.database.claim_job()
            if job is None:
                return False
            self._run_claimed(job)
            return True
        finally:
            self._process_lock.release()

    def run_forever(self, *, stop_event: threading.Event | None = None,
                    poll_interval: float = 0.5) -> None:
        """Poll the persistent queue serially until stopped; holds singleton lock."""
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        stop_event = stop_event or threading.Event()
        self._process_lock.acquire()
        try:
            self._recover_interrupted_jobs()
            while not stop_event.is_set():
                job = self.database.claim_job()
                if job is None:
                    stop_event.wait(poll_interval)
                    continue
                self._run_claimed(job)
        finally:
            self._process_lock.release()

    def _clear_segments(self, job_id: str) -> None:
        with self.database.transaction() as db:
            db.execute("DELETE FROM transcription_segments WHERE job_id=?", (job_id,))

    def _run(self, job_id: str) -> None:
        job = self.database.claim_job(job_id)
        if job is None:
            return
        self._run_claimed(job)

    def _run_claimed(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        try:
            if not job.get("file_path"):
                raise AudioFormatError("任务缺少上传音频文件")
            source_path = self.files._resolve(job["file_path"])
            if not source_path.is_file():
                raise AudioFormatError("上传音频文件已丢失")
            self._clear_segments(job_id)
            options = job.get("options") or {}

            def update_progress(progress: float) -> None:
                self.database.update_job(job_id, progress=max(0.0, min(0.99, progress)))

            result = self.asr.transcribe_file(
                source_path, hotwords=options.get("hotwords") or None,
                channel_mode=options.get("channel_mode", "mixdown"), progress=update_progress,
            )
            for segment in result.segments:
                self.database.upsert_segment(
                    job_id, segment.segment_id, segment.start, segment.end, segment.text,
                    local_speaker_id=segment.local_speaker_id,
                    registered_speaker_id=segment.registered_speaker_id,
                    overlap=segment.overlap, revision=segment.revision,
                    metadata=segment.metadata or {},
                )
            metadata = {
                "duration": result.duration,
                "text": result.text,
                "hotword_set_id": job.get("hotword_set_id"),
                "hotword_set_version": job.get("hotword_version"),
                "channel_mode": options.get("channel_mode", "mixdown"),
                "speaker_diarization": result.speaker_diarization,
                "speaker_mappings": list(result.speaker_mappings),
            }
            self.database.update_job(job_id, status="succeeded", progress=1,
                                     finished_at=_now(), result_metadata=metadata)
        except Exception as exc:
            if isinstance(exc, EngineConfigError):
                code = "model_unavailable"
            elif isinstance(exc, (AudioFormatError, FFmpegUnavailableError, ValueError)):
                code = "invalid_audio"
            else:
                code = "recognition_error"
            self.database.update_job(job_id, status="failed", error_code=code,
                                     error_message=str(exc), finished_at=_now())
