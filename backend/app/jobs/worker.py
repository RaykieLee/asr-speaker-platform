"""Thread-pool worker for durable transcription jobs."""
from __future__ import annotations

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

    def start(self) -> None:
        """Resume durable work after process startup; safe to call once per worker."""
        with self._lock:
            if self._started:
                return
            if self._closed:
                raise RuntimeError("job worker is closed")
            self._started = True
        for job in self.database.list_jobs(status="running", limit=100_000):
            self.database.update_job(job["id"], status="queued", progress=0,
                                     error_code="", error_message="", started_at="", finished_at="")
        for job in reversed(self.database.list_jobs(status="queued", limit=100_000)):
            self.enqueue(job["id"])

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

    def _clear_segments(self, job_id: str) -> None:
        with self.database.transaction() as db:
            db.execute("DELETE FROM transcription_segments WHERE job_id=?", (job_id,))

    def _run(self, job_id: str) -> None:
        job = self.database.get_job(job_id)
        if job is None:
            return
        started = _now()
        self.database.update_job(job_id, status="running", progress=0, error_code="",
                                 error_message="", started_at=started, finished_at="")
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
                source_path, hotwords=options.get("hotwords") or None, progress=update_progress,
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
