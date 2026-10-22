"""Local backup/restore and retention maintenance for SQLite plus FileStore."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from contextlib import closing
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePath
from typing import Any

from .database import SQLiteStore
from .files import FileStore, UnsafeStoragePath

BACKUP_FORMAT = "asr-local-backup-v1"
DATABASE_FILENAME = "database.sqlite3"
MANIFEST_FILENAME = "manifest.json"
_TERMINAL_STATUSES = ("succeeded", "completed", "complete", "failed", "cancelled", "canceled")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: datetime | None = None) -> datetime:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return _utc(parsed)


def _safe_relative_path(root: Path, raw: str) -> Path:
    relative = Path(raw)
    if not raw or relative.is_absolute() or PurePath(raw).is_absolute() or ".." in relative.parts:
        raise UnsafeStoragePath("artifact path must be relative and cannot traverse parent directories")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise UnsafeStoragePath("artifact path resolves outside its root") from exc
    if candidate == root.resolve():
        raise UnsafeStoragePath("artifact path must identify a file")
    return candidate


def _references(connection: sqlite3.Connection) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = defaultdict(lambda: {"transcription_jobs": 0, "speaker_samples": 0})
    for row in connection.execute("SELECT file_path, COUNT(*) FROM transcription_jobs WHERE file_path IS NOT NULL GROUP BY file_path"):
        result[str(row[0])]["transcription_jobs"] = int(row[1])
    for row in connection.execute("SELECT audio_path, COUNT(*) FROM speaker_samples WHERE audio_path IS NOT NULL GROUP BY audio_path"):
        result[str(row[0])]["speaker_samples"] = int(row[1])
    return dict(result)


def _expire_snapshot_audio(connection: sqlite3.Connection, cutoff: datetime) -> int:
    rows = connection.execute(
        "SELECT id, status, finished_at FROM transcription_jobs WHERE file_path IS NOT NULL"
    ).fetchall()
    expired = [str(row[0]) for row in rows
               if str(row[1]).lower() in _TERMINAL_STATUSES
               and (finished := _parse_time(row[2])) is not None and finished < cutoff]
    if expired:
        connection.executemany("UPDATE transcription_jobs SET file_path=NULL WHERE id=?", ((job_id,) for job_id in expired))
        connection.commit()
    return len(expired)


def _verify_backup_dir(backup_dir: Path) -> dict[str, Any]:
    root = backup_dir.expanduser().resolve()
    manifest_path = root / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format") != BACKUP_FORMAT:
        raise ValueError("unsupported or malformed backup manifest")
    database_info = manifest.get("database")
    if not isinstance(database_info, dict) or database_info.get("file") != DATABASE_FILENAME:
        raise ValueError("manifest database entry is invalid")
    database_path = _safe_relative_path(root, DATABASE_FILENAME)
    if database_path.is_symlink():
        raise ValueError("backup database must not be a symbolic link")
    if not database_path.is_file():
        raise FileNotFoundError(f"backup database is missing: {database_path}")
    if database_path.stat().st_size != database_info.get("size_bytes") or _sha256(database_path) != database_info.get("sha256"):
        raise ValueError("backup database size or SHA-256 does not match the manifest")
    database_uri = database_path.as_uri() + "?mode=ro"
    with closing(sqlite3.connect(database_uri, uri=True)) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"backup database integrity check failed: {integrity}")
        references = _references(connection)
    listed = manifest.get("files")
    if not isinstance(listed, list):
        raise ValueError("manifest files must be an array")
    seen: set[str] = set()
    for item in listed:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("manifest file entry is invalid")
        rel = item["path"]
        if rel in seen:
            raise ValueError(f"duplicate file path in manifest: {rel}")
        seen.add(rel)
        artifact = _safe_relative_path(root / "files", rel)
        if artifact.is_symlink():
            raise ValueError(f"backup artifact must not be a symbolic link: {rel}")
        if not artifact.is_file():
            raise FileNotFoundError(f"backup artifact is missing: {rel}")
        if artifact.stat().st_size != item.get("size_bytes") or _sha256(artifact) != item.get("sha256"):
            raise ValueError(f"backup artifact does not match manifest: {rel}")
        if item.get("references") != references.get(rel):
            raise ValueError(f"backup artifact reference counts do not match database: {rel}")
    missing = set(references) - seen
    if missing:
        raise ValueError(f"database references files absent from backup manifest: {sorted(missing)!r}")
    if len(seen) != len(references):
        raise ValueError("manifest contains an unreferenced file")
    files_root = root / "files"
    actual: set[str] = set()
    if files_root.exists():
        for artifact in files_root.rglob("*"):
            if artifact.is_symlink():
                raise ValueError(f"backup file tree contains a symbolic link: {artifact.relative_to(files_root)}")
            if artifact.is_file():
                actual.add(artifact.relative_to(files_root).as_posix())
    if actual != seen:
        raise ValueError(f"backup file tree and manifest differ: unlisted={sorted(actual-seen)!r}, missing={sorted(seen-actual)!r}")
    return {"valid": True, "database_sha256": database_info["sha256"],
            "file_count": len(seen), "referenced_paths": len(references),
            "integrity_check": integrity}


def verify_backup(backup_dir: str | Path) -> dict[str, Any]:
    """Verify manifest hashes, SQLite integrity, and DB-to-file references."""
    return _verify_backup_dir(Path(backup_dir))


def create_backup(database_path: str | Path, files_root: str | Path,
                  destination: str | Path, *, retention_days: int = 30,
                  now: datetime | None = None) -> dict[str, Any]:
    """Create an atomic backup directory using SQLite's online backup API.

    Expired terminal-task file references are cleared only in the backed-up
    database. The live application database is never modified by backup.
    """
    if retention_days < 1:
        raise ValueError("retention_days must be at least one")
    source_db = Path(database_path).expanduser().resolve()
    source_files = Path(files_root).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    if not source_db.is_file():
        raise FileNotFoundError(f"database does not exist: {source_db}")
    if not source_files.is_dir():
        raise FileNotFoundError(f"file store does not exist: {source_files}")
    if target.exists():
        raise FileExistsError(f"backup destination already exists: {target}")
    if target == source_files or target.is_relative_to(source_files):
        raise ValueError("backup destination must be outside the live file store")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    cutoff = _utc(now) - timedelta(days=retention_days)
    try:
        snapshot_db = stage / DATABASE_FILENAME
        source = sqlite3.connect(str(source_db), timeout=30.0)
        snapshot = sqlite3.connect(str(snapshot_db))
        try:
            source.backup(snapshot)
            if snapshot.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("online SQLite backup failed integrity_check")
            expired_count = _expire_snapshot_audio(snapshot, cutoff)
            refs = _references(snapshot)
        finally:
            snapshot.close()
            source.close()
        files_out = stage / "files"
        files_out.mkdir()
        file_entries = []
        for relative, counts in sorted(refs.items()):
            source_file = _safe_relative_path(source_files, relative)
            if not source_file.is_file():
                raise FileNotFoundError(f"database references missing file: {relative}")
            destination_file = _safe_relative_path(files_out, relative)
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, destination_file)
            file_entries.append({"path": relative, "size_bytes": destination_file.stat().st_size,
                                 "sha256": _sha256(destination_file), "references": counts})
        database_info = {"file": DATABASE_FILENAME, "size_bytes": snapshot_db.stat().st_size,
                         "sha256": _sha256(snapshot_db), "integrity_check": "ok"}
        manifest = {
            "format": BACKUP_FORMAT,
            "created_at": _utc(now).isoformat(timespec="seconds"),
            "task_audio_retention_days": retention_days,
            "task_audio_cutoff": cutoff.isoformat(timespec="seconds"),
            "expired_terminal_task_references_cleared": expired_count,
            "database": database_info,
            "files": file_entries,
        }
        (stage / MANIFEST_FILENAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        verified = _verify_backup_dir(stage)
        os.replace(stage, target)
        return {**verified, "backup_dir": str(target), "expired_terminal_task_references_cleared": expired_count}
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def restore_backup(backup_dir: str | Path, destination: str | Path) -> Path:
    """Restore a verified database/file pair into a new directory atomically."""
    source = Path(backup_dir).expanduser().resolve()
    verify_backup(source)
    target = Path(destination).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"restore destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.restore-", dir=target.parent))
    try:
        shutil.copy2(source / DATABASE_FILENAME, stage / DATABASE_FILENAME)
        shutil.copy2(source / MANIFEST_FILENAME, stage / MANIFEST_FILENAME)
        shutil.copytree(source / "files", stage / "files")
        _verify_backup_dir(stage)
        os.replace(stage, target)
        return target
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _expire_one(database: SQLiteStore, files: FileStore, job_id: str,
                cutoff: datetime) -> tuple[str, str | None]:
    with database.transaction() as connection:
        row = connection.execute(
            "SELECT status, finished_at, file_path FROM transcription_jobs WHERE id=?", (job_id,)
        ).fetchone()
        if row is None or not row[2] or str(row[0]).lower() not in _TERMINAL_STATUSES:
            return "skipped", None
        finished = _parse_time(row[1])
        if finished is None or finished >= cutoff:
            return "skipped", None
        relative = str(row[2])
        others = connection.execute(
            """SELECT
                 (SELECT COUNT(*) FROM transcription_jobs WHERE file_path=? AND id<>?) +
                 (SELECT COUNT(*) FROM speaker_samples WHERE audio_path=?)""",
            (relative, job_id, relative),
        ).fetchone()[0]
        if others:
            connection.execute("UPDATE transcription_jobs SET file_path=NULL, updated_at=? WHERE id=?",
                               (_utc().isoformat(timespec="milliseconds"), job_id))
            return "retained_shared", relative
        # Keep SQLite's write lock until unlink and pointer update both finish.
        # If unlink raises, the transaction rolls back and the reference remains.
        files.delete(relative)
        connection.execute("UPDATE transcription_jobs SET file_path=NULL, updated_at=? WHERE id=?",
                           (_utc().isoformat(timespec="milliseconds"), job_id))
        return "deleted", relative


def cleanup_expired_task_audio(database: SQLiteStore, files: FileStore, *,
                               retention_days: int = 30, dry_run: bool = True,
                               now: datetime | None = None) -> dict[str, Any]:
    """Clear expired terminal task file references and delete only unshared files."""
    if retention_days < 1:
        raise ValueError("retention_days must be at least one")
    cutoff = _utc(now) - timedelta(days=retention_days)
    with database._lock:
        rows = database._connection.execute(
            "SELECT id, status, finished_at, file_path FROM transcription_jobs WHERE file_path IS NOT NULL"
        ).fetchall()
    candidates = []
    for row in rows:
        finished = _parse_time(row["finished_at"])
        if str(row["status"]).lower() in _TERMINAL_STATUSES and finished is not None and finished < cutoff:
            candidates.append({"job_id": str(row["id"]), "file_path": str(row["file_path"])})
    if dry_run:
        candidate_ids = {item["job_id"] for item in candidates}
        planned_delete, shared = [], []
        with database._lock:
            for relative in dict.fromkeys(item["file_path"] for item in candidates):
                refs = database._connection.execute(
                    """SELECT
                         (SELECT COUNT(*) FROM transcription_jobs WHERE file_path=? AND id NOT IN (%s)) +
                         (SELECT COUNT(*) FROM speaker_samples WHERE audio_path=?)""" %
                    (",".join("?" for _ in candidate_ids) or "NULL"),
                    (relative, *candidate_ids, relative),
                ).fetchone()[0]
                (shared if refs else planned_delete).append(relative)
        return {"dry_run": True, "cutoff": cutoff.isoformat(), "candidate_jobs": candidates,
                "planned_delete_files": planned_delete, "shared_files_retained": shared,
                "deleted_files": [], "cleared_jobs": []}

    deleted: list[str] = []
    shared: list[str] = []
    cleared: list[str] = []
    failures: list[dict[str, str]] = []
    for item in candidates:
        try:
            outcome, relative = _expire_one(database, files, item["job_id"], cutoff)
            if outcome == "deleted" and relative:
                deleted.append(relative)
            if outcome == "retained_shared" and relative:
                shared.append(relative)
            if outcome in {"deleted", "retained_shared"}:
                cleared.append(item["job_id"])
        except Exception as exc:
            failures.append({"job_id": item["job_id"], "file_path": item["file_path"],
                             "error": f"{type(exc).__name__}: {exc}"})
    return {"dry_run": False, "cutoff": cutoff.isoformat(), "candidate_jobs": candidates,
            "deleted_files": list(dict.fromkeys(deleted)),
            "shared_files_retained": list(dict.fromkeys(shared)),
            "cleared_jobs": cleared, "failures": failures}
