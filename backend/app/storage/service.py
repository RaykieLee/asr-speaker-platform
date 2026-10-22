"""Coordinated database and file operations used by application services."""

from __future__ import annotations

import threading
from collections.abc import Iterable

from .database import SQLiteStore
from .files import FileStore


class FileCleanupError(RuntimeError):
    """Raised after rows are removed when one or more owned files could not be deleted."""

    def __init__(self, failed_paths: Iterable[str], cause: BaseException):
        self.failed_paths = tuple(failed_paths)
        super().__init__(f"database rows were removed but file cleanup failed for {self.failed_paths!r}: {cause}")


class StorageService:
    """Application-level storage policy combining SQLite and local files.

    Sample and speaker deletion first commits the relational cascade, then
    removes the corresponding unique sample audio paths through ``FileStore``.
    Paths still referenced by another sample or transcription job are retained.
    Missing files are harmless. Cleanup attempts every unreferenced file and
    raises ``FileCleanupError`` with failed paths if an I/O or safety check
    prevents complete cleanup; this makes post-commit cleanup failures visible
    to callers rather than silently leaving biometric audio behind.

    As SQLite and a local filesystem do not share a transaction, a process crash
    between row deletion and file cleanup can leave an orphan file. Deployments
    can periodically reconcile files against ``speaker_samples.audio_path``.
    """

    def __init__(self, database: SQLiteStore, files: FileStore):
        self.database = database
        self.files = files
        self._delete_lock = threading.RLock()

    def _delete_unreferenced(self, paths: Iterable[str]) -> None:
        unique_paths = tuple(dict.fromkeys(path for path in paths if path))
        failed: list[str] = []
        first_error: BaseException | None = None
        for path in unique_paths:
            # Do not remove shared paths that remain referenced by another
            # registration sample or by an uploaded transcription job.
            with self.database._lock:
                referenced = self.database._connection.execute(
                    """SELECT 1 FROM speaker_samples WHERE audio_path=?
                       UNION ALL SELECT 1 FROM transcription_jobs WHERE file_path=? LIMIT 1""",
                    (path, path),
                ).fetchone()
            if referenced:
                continue
            try:
                self.files.delete(path)
            except Exception as exc:  # attempt remaining paths before reporting
                failed.append(path)
                if first_error is None:
                    first_error = exc
        if failed:
            raise FileCleanupError(failed, first_error or OSError("unknown cleanup failure"))

    def delete_sample(self, sample_id: str) -> bool:
        """Delete one sample row and its now-unreferenced audio file."""
        with self._delete_lock:
            sample = self.database.get_sample(sample_id)
            if sample is None or not self.database.delete_sample(sample_id):
                return False
            self._delete_unreferenced([sample["audio_path"]])
            return True

    def delete_speaker(self, speaker_id: str) -> bool:
        """Delete a speaker and clean all its sample files after FK cascades."""
        with self._delete_lock:
            samples = self.database.list_samples(speaker_id)
            if not self.database.delete_speaker(speaker_id):
                return False
            self._delete_unreferenced(sample["audio_path"] for sample in samples)
            return True
