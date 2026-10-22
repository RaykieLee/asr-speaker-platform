"""Upload, hotword snapshot and query operations for transcription jobs."""
from __future__ import annotations

import hashlib
import inspect
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import PurePath
from typing import Any

from backend.app.pipelines.file_asr import FileASRService
from backend.app.storage import FileStore, SQLiteStore

from .worker import JobWorker


class UploadTooLarge(ValueError):
    """An upload exceeds the configured per-file limit."""


class HotwordSnapshotError(ValueError):
    """A requested hotword set cannot be snapshotted for this job."""


_ALLOWED_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".aac"}


def _resolve_hotword_value(resolved: Any) -> tuple[list[str], int]:
    if isinstance(resolved, Mapping):
        words, version = resolved.get("entries", resolved.get("hotwords")), resolved.get("version")
    else:
        try:
            words, version = resolved
        except (TypeError, ValueError) as exc:
            raise HotwordSnapshotError("热词解析器必须返回词条和版本") from exc
    if isinstance(words, list) and words and isinstance(words[0], Mapping):
        words = [entry.get("word") for entry in words if entry.get("enabled", True)]
    if not isinstance(words, (list, tuple)) or any(not isinstance(word, str) or not word.strip() for word in words):
        raise HotwordSnapshotError("热词集内容无效")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise HotwordSnapshotError("热词集版本必须是正整数")
    return [word.strip() for word in words], version


class TranscriptionService:
    """Owns uploaded files, immutable options snapshots and the background worker."""

    def __init__(self, database: SQLiteStore, files: FileStore, asr: FileASRService, *,
                 hotword_resolver: Callable[[str], Any] | None = None,
                 max_upload_bytes: int = 512 * 1024 * 1024,
                 max_workers: int = 1):
        if max_upload_bytes < 1:
            raise ValueError("max_upload_bytes must be positive")
        self.database = database
        self.files = files
        self.asr = asr
        self.hotword_resolver = hotword_resolver
        self.max_upload_bytes = max_upload_bytes
        self.worker = JobWorker(database, files, asr, max_workers=max_workers)
        self.worker.start()

    async def create_job(self, content: bytes, filename: str, *, language: str = "auto",
                         channel_mode: str = "mixdown", hotword_set_id: str | None = None) -> dict[str, Any]:
        if not content:
            raise ValueError("上传文件不能为空")
        if len(content) > self.max_upload_bytes:
            raise UploadTooLarge(f"上传文件超过 {self.max_upload_bytes} 字节限制")
        if channel_mode != "mixdown":
            raise ValueError("当前只支持 channel_mode=mixdown")
        if not language or len(language) > 32:
            raise ValueError("language 必须是 1 到 32 个字符")

        hotwords: list[str] = []
        hotword_version: int | None = None
        if hotword_set_id is not None:
            if not isinstance(hotword_set_id, str) or not hotword_set_id.strip():
                raise ValueError("hotword_set_id 必须是非空字符串")
            if self.hotword_resolver is None:
                raise HotwordSnapshotError("热词解析器未配置，不能创建未固定热词版本的任务")
            try:
                resolved = self.hotword_resolver(hotword_set_id)
                if inspect.isawaitable(resolved):
                    resolved = await resolved
                hotwords, hotword_version = _resolve_hotword_value(resolved)
            except HotwordSnapshotError:
                raise
            except Exception as exc:
                raise HotwordSnapshotError(f"热词集快照失败: {exc}") from exc

        job_id = f"job_{uuid.uuid4().hex}"
        suffix = PurePath(filename or "audio").suffix.lower()
        if suffix not in _ALLOWED_SUFFIXES:
            suffix = ".audio"
        relative_path = f"transcriptions/{job_id}{suffix}"
        checksum = hashlib.sha256(content).hexdigest()
        options = {
            "language": language,
            "channel_mode": channel_mode,
            "hotwords": hotwords,
            "hotword_set_version": hotword_version,
            "snapshot_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        }
        model_version = self.asr.model_version
        self.files.write(relative_path, content)
        try:
            job = self.database.create_job(
                job_id=job_id, file_path=relative_path, checksum=checksum,
                hotword_set_id=hotword_set_id, hotword_version=hotword_version,
                options=options, model_version=model_version,
            )
        except Exception:
            self.files.delete(relative_path)
            raise
        self.worker.enqueue(job_id)
        return job

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        job = self.database.get_job(job_id)
        if job is None:
            return None
        job["segments"] = self.database.list_segments(job_id)
        if not job.get("error_code"):
            job["error_code"] = None
        if not job.get("error_message"):
            job["error_message"] = None
        return job

    def retry_job(self, job_id: str) -> dict[str, Any] | None:
        self.worker.retry(job_id)
        return self.get_job(job_id)

    def close(self, *, wait: bool = True) -> None:
        self.worker.close(wait=wait)
