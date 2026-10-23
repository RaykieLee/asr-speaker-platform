"""Versioned SQLite schema and repositories for speaker, job and hotword data."""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


SCHEMA_VERSION = 1
_JSON_COLUMNS = {
    "speakers": {"metadata"},
    "speaker_samples": {"quality"},
    "transcription_jobs": {"options", "result_metadata"},
    "transcription_segments": {"metadata"},
    "speaker_mapping_overrides": {"metadata"},
    "hotword_set_versions": {"entries", "compiled"},
}
_EMBEDDING_COLUMNS = {
    "speaker_samples": {"embedding"},
    "speaker_profiles": {"embedding"},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _embedding_blob(value: Iterable[float] | bytes | bytearray | None) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        if len(value) % 4:
            raise ValueError("embedding BLOB length must be a multiple of 4")
        return bytes(value)
    values = tuple(float(v) for v in value)
    return struct.pack(f"<{len(values)}f", *values)


def _embedding_values(value: bytes | None) -> list[float] | None:
    if value is None:
        return None
    if len(value) % 4:
        raise ValueError("stored embedding BLOB length is invalid")
    return list(struct.unpack(f"<{len(value) // 4}f", value))


class SQLiteStore:
    """Thread-safe SQLite repository.

    The connection uses WAL, foreign keys and a generous busy timeout. All public
    mutation methods take the same RLock, so callers may safely share one store
    among FastAPI threads. For multi-process deployments SQLite's own locking is
    still active; keep write transactions short.
    """

    def __init__(self, database_path: str | Path):
        self.path = Path(database_path)
        if str(database_path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(database_path), timeout=30.0, check_same_thread=False,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 30000")
        if str(database_path) != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = NORMAL")

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteStore":
        self.initialize()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a caller-supplied group of SQL statements atomically."""
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def initialize(self) -> None:
        """Apply all pending schema migrations; safe to call repeatedly."""
        with self._lock:
            version = self._connection.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"database schema {version} is newer than supported {SCHEMA_VERSION}"
                )
            if version < 1:
                # sqlite3.executescript issues an implicit commit before the
                # script, so keep the migration's explicit transaction inside
                # that script rather than nesting it in transaction().
                try:
                    self._connection.executescript(
                        "BEGIN IMMEDIATE;\n" +
                        """
                        CREATE TABLE IF NOT EXISTS speakers (
                            id TEXT PRIMARY KEY,
                            name TEXT NOT NULL,
                            status TEXT NOT NULL DEFAULT 'active',
                            tenant_id TEXT,
                            metadata TEXT NOT NULL DEFAULT '{}',
                            created_at TEXT NOT NULL,
                            updated_at TEXT NOT NULL
                        );
                        CREATE TABLE IF NOT EXISTS speaker_samples (
                            id TEXT PRIMARY KEY,
                            speaker_id TEXT NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
                            audio_path TEXT NOT NULL,
                            duration_seconds REAL,
                            quality TEXT NOT NULL DEFAULT '{}',
                            model_version TEXT,
                            embedding BLOB,
                            created_at TEXT NOT NULL
                        );
                        CREATE INDEX IF NOT EXISTS idx_speaker_samples_speaker ON speaker_samples(speaker_id);
                        CREATE TABLE IF NOT EXISTS speaker_profiles (
                            speaker_id TEXT PRIMARY KEY REFERENCES speakers(id) ON DELETE CASCADE,
                            embedding BLOB NOT NULL,
                            sample_count INTEGER NOT NULL DEFAULT 0,
                            model_version TEXT,
                            updated_at TEXT NOT NULL
                        );
                        CREATE TABLE IF NOT EXISTS transcription_jobs (
                            id TEXT PRIMARY KEY,
                            file_path TEXT,
                            checksum TEXT,
                            status TEXT NOT NULL DEFAULT 'queued',
                            progress REAL NOT NULL DEFAULT 0 CHECK(progress >= 0 AND progress <= 1),
                            error_code TEXT,
                            error_message TEXT,
                            model_version TEXT,
                            hotword_set_id TEXT REFERENCES hotword_sets(id) ON DELETE SET NULL,
                            hotword_version INTEGER,
                            options TEXT NOT NULL DEFAULT '{}',
                            result_metadata TEXT NOT NULL DEFAULT '{}',
                            created_at TEXT NOT NULL,
                            started_at TEXT,
                            finished_at TEXT,
                            updated_at TEXT NOT NULL
                        );
                        CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON transcription_jobs(status, created_at);
                        CREATE TABLE IF NOT EXISTS transcription_segments (
                            job_id TEXT NOT NULL REFERENCES transcription_jobs(id) ON DELETE CASCADE,
                            segment_id INTEGER NOT NULL,
                            start_seconds REAL NOT NULL,
                            end_seconds REAL NOT NULL,
                            text TEXT NOT NULL DEFAULT '',
                            local_speaker_id TEXT,
                            registered_speaker_id TEXT REFERENCES speakers(id) ON DELETE SET NULL,
                            overlap INTEGER NOT NULL DEFAULT 0,
                            revision INTEGER NOT NULL DEFAULT 1,
                            metadata TEXT NOT NULL DEFAULT '{}',
                            PRIMARY KEY(job_id, segment_id),
                            CHECK(start_seconds >= 0 AND end_seconds >= start_seconds)
                        );
                        CREATE INDEX IF NOT EXISTS idx_segments_speaker ON transcription_segments(registered_speaker_id);
                        CREATE TABLE IF NOT EXISTS speaker_mapping_overrides (
                            id TEXT PRIMARY KEY,
                            job_id TEXT NOT NULL REFERENCES transcription_jobs(id) ON DELETE CASCADE,
                            local_speaker_id TEXT NOT NULL,
                            registered_speaker_id TEXT REFERENCES speakers(id) ON DELETE SET NULL,
                            display_name TEXT,
                            operator TEXT,
                            metadata TEXT NOT NULL DEFAULT '{}',
                            created_at TEXT NOT NULL,
                            UNIQUE(job_id, local_speaker_id)
                        );
                        CREATE TABLE IF NOT EXISTS hotword_sets (
                            id TEXT PRIMARY KEY,
                            name TEXT NOT NULL,
                            current_version INTEGER NOT NULL DEFAULT 0,
                            status TEXT NOT NULL DEFAULT 'active',
                            tenant_id TEXT,
                            created_at TEXT NOT NULL,
                            updated_at TEXT NOT NULL
                        );
                        CREATE TABLE IF NOT EXISTS hotword_set_versions (
                            set_id TEXT NOT NULL REFERENCES hotword_sets(id) ON DELETE CASCADE,
                            version INTEGER NOT NULL,
                            entries TEXT NOT NULL,
                            compiled TEXT NOT NULL DEFAULT 'null',
                            weight REAL NOT NULL DEFAULT 1.0,
                            created_by TEXT,
                            created_at TEXT NOT NULL,
                            PRIMARY KEY(set_id, version)
                        );
                        PRAGMA user_version = 1;
                        COMMIT;
                        """
                    )
                except BaseException:
                    if self._connection.in_transaction:
                        self._connection.rollback()
                    raise

    def _query_one(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(sql, params).fetchone()
            return dict(row) if row else None

    def _query_all(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._connection.execute(sql, params).fetchall()]

    def _decode(self, table: str, row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        for column in _JSON_COLUMNS.get(table, ()):
            if column in row:
                row[column] = json.loads(row[column])
        for column in _EMBEDDING_COLUMNS.get(table, ()):
            if column in row:
                row[column] = _embedding_values(row[column])
        if "overlap" in row:
            row["overlap"] = bool(row["overlap"])
        return row

    def create_speaker(self, name: str, *, speaker_id: str | None = None,
                       status: str = "active", tenant_id: str | None = None,
                       metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        now, speaker_id = _now(), speaker_id or _id("spk")
        with self.transaction() as db:
            db.execute("INSERT INTO speakers VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (speaker_id, name, status, tenant_id, json.dumps(metadata or {}), now, now))
        return self.get_speaker(speaker_id)  # type: ignore[return-value]

    def get_speaker(self, speaker_id: str) -> dict[str, Any] | None:
        return self._decode("speakers", self._query_one("SELECT * FROM speakers WHERE id=?", (speaker_id,)))

    def list_speakers(self, *, status: str | None = None) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM speakers", ()
        if status is not None:
            sql, args = sql + " WHERE status=?", (status,)
        sql += " ORDER BY created_at, id"
        return [self._decode("speakers", r) for r in self._query_all(sql, args)]  # type: ignore[misc]

    def delete_speaker(self, speaker_id: str) -> bool:
        """Delete identity, samples and profile; preserve task rows as unknown."""
        with self.transaction() as db:
            cursor = db.execute("DELETE FROM speakers WHERE id=?", (speaker_id,))
            return cursor.rowcount > 0

    def create_sample(self, speaker_id: str, audio_path: str, *, sample_id: str | None = None,
                      duration_seconds: float | None = None, quality: Mapping[str, Any] | None = None,
                      model_version: str | None = None,
                      embedding: Iterable[float] | bytes | bytearray | None = None) -> dict[str, Any]:
        sample_id, now = sample_id or _id("smp"), _now()
        with self.transaction() as db:
            db.execute("INSERT INTO speaker_samples VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       (sample_id, speaker_id, audio_path, duration_seconds,
                        json.dumps(quality or {}), model_version, _embedding_blob(embedding), now))
        return self.get_sample(sample_id)  # type: ignore[return-value]

    def get_sample(self, sample_id: str) -> dict[str, Any] | None:
        return self._decode("speaker_samples", self._query_one("SELECT * FROM speaker_samples WHERE id=?", (sample_id,)))

    def list_samples(self, speaker_id: str) -> list[dict[str, Any]]:
        return [self._decode("speaker_samples", r) for r in self._query_all(
            "SELECT * FROM speaker_samples WHERE speaker_id=? ORDER BY created_at, id", (speaker_id,))]  # type: ignore[misc]

    def delete_sample(self, sample_id: str) -> bool:
        """Delete one audio sample and invalidate its potentially stale profile."""
        with self.transaction() as db:
            sample = db.execute("SELECT speaker_id FROM speaker_samples WHERE id=?", (sample_id,)).fetchone()
            if sample is None:
                return False
            db.execute("DELETE FROM speaker_samples WHERE id=?", (sample_id,))
            db.execute("DELETE FROM speaker_profiles WHERE speaker_id=?", (sample[0],))
            return True

    def set_speaker_profile(self, speaker_id: str, embedding: Iterable[float] | bytes,
                            *, sample_count: int, model_version: str | None = None) -> dict[str, Any]:
        with self.transaction() as db:
            db.execute("""INSERT INTO speaker_profiles VALUES (?, ?, ?, ?, ?)
                         ON CONFLICT(speaker_id) DO UPDATE SET embedding=excluded.embedding,
                         sample_count=excluded.sample_count, model_version=excluded.model_version,
                         updated_at=excluded.updated_at""",
                       (speaker_id, _embedding_blob(embedding), sample_count, model_version, _now()))
        return self.get_speaker_profile(speaker_id)  # type: ignore[return-value]

    def get_speaker_profile(self, speaker_id: str) -> dict[str, Any] | None:
        return self._decode("speaker_profiles", self._query_one(
            "SELECT * FROM speaker_profiles WHERE speaker_id=?", (speaker_id,)))

    def create_job(self, *, job_id: str | None = None, file_path: str | None = None,
                   checksum: str | None = None, status: str = "queued",
                   hotword_set_id: str | None = None, hotword_version: int | None = None,
                   options: Mapping[str, Any] | None = None,
                   model_version: str | None = None) -> dict[str, Any]:
        job_id, now = job_id or _id("job"), _now()
        with self.transaction() as db:
            db.execute("""INSERT INTO transcription_jobs
                (id,file_path,checksum,status,hotword_set_id,hotword_version,options,model_version,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                       (job_id, file_path, checksum, status, hotword_set_id, hotword_version,
                        json.dumps(options or {}), model_version, now, now))
        return self.get_job(job_id)  # type: ignore[return-value]

    def update_job(self, job_id: str, *, status: str | None = None, progress: float | None = None,
                   error_code: str | None = None, error_message: str | None = None,
                   started_at: str | None = None, finished_at: str | None = None,
                   result_metadata: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        values = {"status": status, "progress": progress, "error_code": error_code,
                  "error_message": error_message, "started_at": started_at,
                  "finished_at": finished_at,
                  "result_metadata": json.dumps(result_metadata) if result_metadata is not None else None}
        updates = [(k, v) for k, v in values.items() if v is not None]
        if not updates:
            return self.get_job(job_id)
        with self.transaction() as db:
            db.execute("UPDATE transcription_jobs SET " + ",".join(f"{k}=?" for k, _ in updates)
                       + ",updated_at=? WHERE id=?", tuple(v for _, v in updates) + (_now(), job_id))
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self._decode("transcription_jobs", self._query_one("SELECT * FROM transcription_jobs WHERE id=?", (job_id,)))

    def list_jobs(self, *, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM transcription_jobs", ()
        if status is not None:
            sql, args = sql + " WHERE status=?", (status,)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args += (limit,)
        return [self._decode("transcription_jobs", r) for r in self._query_all(sql, args)]  # type: ignore[misc]

    def count_jobs(self, *, status: str) -> int:
        """Count jobs in one lifecycle state without loading task contents."""
        row = self._query_one("SELECT COUNT(*) AS count FROM transcription_jobs WHERE status=?", (status,))
        return int(row["count"]) if row is not None else 0

    def upsert_segment(self, job_id: str, segment_id: int, start_seconds: float,
                       end_seconds: float, text: str, *, local_speaker_id: str | None = None,
                       registered_speaker_id: str | None = None, overlap: bool = False,
                       revision: int = 1, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        values = (job_id, segment_id, start_seconds, end_seconds, text, local_speaker_id,
                  registered_speaker_id, int(overlap), revision, json.dumps(metadata or {}))
        with self.transaction() as db:
            db.execute("""INSERT INTO transcription_segments VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(job_id,segment_id) DO UPDATE SET start_seconds=excluded.start_seconds,
                end_seconds=excluded.end_seconds,text=excluded.text,local_speaker_id=excluded.local_speaker_id,
                registered_speaker_id=excluded.registered_speaker_id,overlap=excluded.overlap,
                revision=excluded.revision,metadata=excluded.metadata""", values)
        return self._decode("transcription_segments", self._query_one(
            "SELECT * FROM transcription_segments WHERE job_id=? AND segment_id=?", (job_id, segment_id)))  # type: ignore[return-value]

    def list_segments(self, job_id: str) -> list[dict[str, Any]]:
        return [self._decode("transcription_segments", r) for r in self._query_all(
            "SELECT * FROM transcription_segments WHERE job_id=? ORDER BY start_seconds,segment_id", (job_id,))]  # type: ignore[misc]

    def set_mapping_override(self, job_id: str, local_speaker_id: str, *,
                             registered_speaker_id: str | None = None,
                             display_name: str | None = None, operator: str | None = None,
                             metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        override_id, now = _id("ovr"), _now()
        with self.transaction() as db:
            db.execute("""INSERT INTO speaker_mapping_overrides VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(job_id,local_speaker_id) DO UPDATE SET
                registered_speaker_id=excluded.registered_speaker_id,display_name=excluded.display_name,
                operator=excluded.operator,metadata=excluded.metadata,created_at=excluded.created_at""",
                       (override_id, job_id, local_speaker_id, registered_speaker_id, display_name,
                        operator, json.dumps(metadata or {}), now))
            row = db.execute("SELECT * FROM speaker_mapping_overrides WHERE job_id=? AND local_speaker_id=?",
                             (job_id, local_speaker_id)).fetchone()
        return self._decode("speaker_mapping_overrides", dict(row))  # type: ignore[return-value]

    def get_mapping_override(self, job_id: str, local_speaker_id: str) -> dict[str, Any] | None:
        return self._decode("speaker_mapping_overrides", self._query_one(
            "SELECT * FROM speaker_mapping_overrides WHERE job_id=? AND local_speaker_id=?",
            (job_id, local_speaker_id)))

    def create_hotword_set(self, name: str, entries: list[Any], *,
                           hotword_set_id: str | None = None, weight: float = 1.0,
                           tenant_id: str | None = None, created_by: str | None = None,
                           compiled: Any = None) -> dict[str, Any]:
        hotword_set_id, now = hotword_set_id or _id("hws"), _now()
        with self.transaction() as db:
            db.execute("INSERT INTO hotword_sets VALUES (?,?,1,'active',?,?,?)",
                       (hotword_set_id, name, tenant_id, now, now))
            db.execute("INSERT INTO hotword_set_versions VALUES (?,?,?,?,?,?,?)",
                       (hotword_set_id, 1, json.dumps(entries, ensure_ascii=False),
                        json.dumps(compiled, ensure_ascii=False), weight, created_by, now))
        return self.get_hotword_version(hotword_set_id, 1)  # type: ignore[return-value]

    def append_hotword_version(self, hotword_set_id: str, entries: list[Any], *,
                               expected_version: int, weight: float = 1.0,
                               created_by: str | None = None, compiled: Any = None) -> dict[str, Any]:
        """Append an immutable snapshot with optimistic version checking."""
        now = _now()
        with self.transaction() as db:
            row = db.execute("SELECT current_version FROM hotword_sets WHERE id=?", (hotword_set_id,)).fetchone()
            if row is None:
                raise KeyError(hotword_set_id)
            if row[0] != expected_version:
                raise ValueError(f"stale hotword version: expected {expected_version}, current {row[0]}")
            version = expected_version + 1
            db.execute("INSERT INTO hotword_set_versions VALUES (?,?,?,?,?,?,?)",
                       (hotword_set_id, version, json.dumps(entries, ensure_ascii=False),
                        json.dumps(compiled, ensure_ascii=False), weight, created_by, now))
            db.execute("UPDATE hotword_sets SET current_version=?,updated_at=? WHERE id=?",
                       (version, now, hotword_set_id))
        return self.get_hotword_version(hotword_set_id, version)  # type: ignore[return-value]

    def get_hotword_set(self, hotword_set_id: str) -> dict[str, Any] | None:
        return self._query_one("SELECT * FROM hotword_sets WHERE id=?", (hotword_set_id,))

    def get_hotword_version(self, hotword_set_id: str, version: int | None = None) -> dict[str, Any] | None:
        if version is None:
            hotword_set = self.get_hotword_set(hotword_set_id)
            if hotword_set is None:
                return None
            version = hotword_set["current_version"]
        return self._decode("hotword_set_versions", self._query_one(
            "SELECT * FROM hotword_set_versions WHERE set_id=? AND version=?", (hotword_set_id, version)))

    def set_hotword_set_status(self, hotword_set_id: str, status: str) -> bool:
        with self.transaction() as db:
            cur = db.execute("UPDATE hotword_sets SET status=?,updated_at=? WHERE id=?",
                             (status, _now(), hotword_set_id))
            return cur.rowcount > 0

    def delete_hotword_set(self, hotword_set_id: str) -> bool:
        """Deactivate a set; immutable version snapshots remain queryable."""
        return self.set_hotword_set_status(hotword_set_id, "inactive")

    def purge_hotword_set(self, hotword_set_id: str) -> bool:
        """Permanently remove a set and versions for explicit retention cleanup."""
        with self.transaction() as db:
            cur = db.execute("DELETE FROM hotword_sets WHERE id=?", (hotword_set_id,))
            return cur.rowcount > 0
