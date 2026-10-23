from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from backend.app.storage import FileStore, SQLiteStore, UnsafeStoragePath
from backend.app.storage.maintenance import (
    cleanup_expired_task_audio,
    create_backup,
    restore_backup,
    verify_backup,
)
from backend.app.storage import maintenance


def open_store(tmp_path):
    database = SQLiteStore(tmp_path / "platform.sqlite3")
    database.initialize()
    files = FileStore(tmp_path / "files")
    return database, files


def add_job(database, job_id, path, status, finished_at=None):
    database.create_job(job_id=job_id, file_path=path, checksum="checksum", status="queued")
    database.update_job(job_id, status=status, finished_at=finished_at)
    return database.get_job(job_id)


def test_backup_is_manifested_online_pair_and_expires_audio_only_in_snapshot(tmp_path):
    database, files = open_store(tmp_path)
    now = datetime(2026, 10, 23, tzinfo=timezone.utc)
    old_path, fresh_path, queued_path, sample_path = (
        "transcriptions/old.wav", "transcriptions/fresh.wav",
        "transcriptions/queued.wav", "speakers/spk/sample.wav",
    )
    for path, content in ((old_path, b"old audio"), (fresh_path, b"fresh audio"),
                          (queued_path, b"queued audio"), (sample_path, b"speaker audio")):
        files.write(path, content)
    database.create_speaker("Alice", speaker_id="spk")
    database.create_sample("spk", sample_path, sample_id="sample-1", duration_seconds=3,
                            quality={}, model_version="embed-v1", embedding=[1.0, 0.0])
    database.set_speaker_profile("spk", [1.0, 0.0], sample_count=1, model_version="embed-v1")
    add_job(database, "old-job", old_path, "succeeded", (now - timedelta(days=31)).isoformat())
    add_job(database, "fresh-job", fresh_path, "succeeded", (now - timedelta(days=3)).isoformat())
    add_job(database, "queued-job", queued_path, "running", None)
    database.upsert_segment("old-job", 0, 0, 1, "retained result")

    destination = tmp_path / "backup"
    result = create_backup(database.path, files.root, destination, now=now)

    assert result["valid"] is True
    assert result["expired_terminal_task_references_cleared"] == 1
    assert (destination / "files" / fresh_path).read_bytes() == b"fresh audio"
    assert (destination / "files" / queued_path).read_bytes() == b"queued audio"
    assert (destination / "files" / sample_path).read_bytes() == b"speaker audio"
    assert not (destination / "files" / old_path).exists()
    assert database.get_job("old-job")["file_path"] == old_path  # source DB untouched

    restored = restore_backup(destination, tmp_path / "restored")
    assert verify_backup(restored)["valid"] is True
    recovered = SQLiteStore(restored / "database.sqlite3")
    try:
        recovered.initialize()
        assert recovered.get_job("old-job")["file_path"] is None
        assert recovered.get_job("old-job")["status"] == "succeeded"
        assert recovered.list_segments("old-job")[0]["text"] == "retained result"
        assert recovered.get_speaker_profile("spk")["embedding"] == [1.0, 0.0]
    finally:
        recovered.close()
    database.close()


def test_cleanup_defaults_to_dry_run_then_keeps_shared_and_active_references(tmp_path):
    database, files = open_store(tmp_path)
    now = datetime(2026, 10, 23, tzinfo=timezone.utc)
    cutoff_old = (now - timedelta(days=31)).isoformat()
    task_path = "transcriptions/task.wav"
    shared_path = "speakers/alice/sample.wav"
    active_path = "transcriptions/active.wav"
    files.write(task_path, b"old task")
    files.write(shared_path, b"shared sample")
    files.write(active_path, b"active task")
    database.create_speaker("Alice", speaker_id="alice")
    database.create_sample("alice", shared_path, sample_id="s1", model_version="v1", embedding=[1.0])
    add_job(database, "expired", task_path, "succeeded", cutoff_old)
    add_job(database, "shared-expired", shared_path, "failed", cutoff_old)
    add_job(database, "active-shared", shared_path, "running", None)
    add_job(database, "running", active_path, "running", None)
    database.upsert_segment("expired", 0, 0, 1, "keep this result")

    plan = cleanup_expired_task_audio(database, files, dry_run=True, now=now)
    assert {item["job_id"] for item in plan["candidate_jobs"]} == {"expired", "shared-expired"}
    assert plan["planned_delete_files"] == [task_path]
    assert plan["shared_files_retained"] == [shared_path]
    assert database.get_job("expired")["file_path"] == task_path
    assert files.exists(task_path)

    result = cleanup_expired_task_audio(database, files, dry_run=False, now=now)
    assert set(result["cleared_jobs"]) == {"expired", "shared-expired"}
    assert result["deleted_files"] == [task_path]
    assert result["shared_files_retained"] == [shared_path]
    assert database.get_job("expired")["file_path"] is None
    assert database.get_job("shared-expired")["file_path"] is None
    assert database.get_job("running")["file_path"] == active_path
    assert not files.exists(task_path)
    assert files.read(shared_path) == b"shared sample"
    assert database.list_segments("expired")[0]["text"] == "keep this result"
    database.close()


def test_cleanup_surfaces_unsafe_path_after_clearing_durable_reference(tmp_path):
    database, files = open_store(tmp_path)
    finished = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
    add_job(database, "unsafe", "../outside.wav", "succeeded", finished)

    result = cleanup_expired_task_audio(database, files, dry_run=False)

    assert result["failures"]
    assert "UnsafeStoragePath" in result["failures"][0]["error"]
    assert database.get_job("unsafe")["file_path"] is None
    database.close()


def test_manifest_hash_and_reference_check_detects_tampering(tmp_path):
    database, files = open_store(tmp_path)
    path = "transcriptions/input.wav"
    files.write(path, b"contents")
    add_job(database, "one", path, "succeeded", datetime.now(timezone.utc).isoformat())
    backup = tmp_path / "backup"
    create_backup(database.path, files.root, backup)

    (backup / "files" / path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match manifest"):
        verify_backup(backup)
    database.close()


def test_backup_rejects_path_traversal_and_restore_never_overwrites(tmp_path):
    database, files = open_store(tmp_path)
    add_job(database, "unsafe", "../outside.wav", "succeeded", datetime.now(timezone.utc).isoformat())
    with pytest.raises(UnsafeStoragePath):
        create_backup(database.path, files.root, tmp_path / "bad-backup")
    database.close()

    # Existing restore targets are never replaced.
    db2, files2 = open_store(tmp_path / "second")
    second_backup = tmp_path / "second-backup"
    create_backup(db2.path, files2.root, second_backup)
    target = tmp_path / "already-there"
    target.mkdir()
    (target / "marker").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        restore_backup(second_backup, target)
    assert (target / "marker").read_text(encoding="utf-8") == "keep"
    db2.close()


def test_cleanup_refuses_internal_symlink_to_registered_sample(tmp_path):
    database, files = open_store(tmp_path)
    now = datetime.now(timezone.utc)
    sample = "speakers/alice/sample.wav"
    alias = "transcriptions/expired.wav"
    files.write(sample, b"registered sample")
    Path(files.root / alias).parent.mkdir(parents=True, exist_ok=True)
    (files.root / alias).symlink_to(files.root / sample)
    database.create_speaker("Alice", speaker_id="alice")
    database.create_sample("alice", sample, sample_id="sample", model_version="v1", embedding=[1.0])
    add_job(database, "expired-link", alias, "succeeded", (now - timedelta(days=31)).isoformat())

    result = cleanup_expired_task_audio(database, files, dry_run=False, now=now)

    assert result["failures"]
    assert database.get_job("expired-link")["file_path"] is None
    assert (files.root / alias).is_symlink()
    assert files.read(sample) == b"registered sample"
    database.close()


def test_verify_rejects_symlink_anywhere_in_backup_tree(tmp_path):
    database, files = open_store(tmp_path)
    backup = tmp_path / "backup"
    create_backup(database.path, files.root, backup)
    (backup / "database-link.sqlite3").symlink_to(backup / "database.sqlite3")

    with pytest.raises(ValueError, match="symbolic links"):
        verify_backup(backup)

    database.close()


def test_restore_rejects_dangling_symlink_and_nested_destination(tmp_path):
    database, files = open_store(tmp_path)
    backup = tmp_path / "backup"
    create_backup(database.path, files.root, backup)
    dangling = tmp_path / "dangling-restore-target"
    dangling.symlink_to(tmp_path / "not-created")

    with pytest.raises(ValueError, match="symbolic link"):
        restore_backup(backup, dangling)
    with pytest.raises(ValueError, match="contain one another"):
        restore_backup(backup, backup / "files" / "recursive")
    backup_alias = tmp_path / "backup-alias"
    backup_alias.symlink_to(backup, target_is_directory=True)
    with pytest.raises(ValueError, match="backup directory.*symbolic link"):
        restore_backup(backup_alias, tmp_path / "restore-through-alias")
    database.close()


def test_cleanup_commits_reference_clear_before_reporting_unlink_failure(tmp_path, monkeypatch):
    database, files = open_store(tmp_path)
    now = datetime.now(timezone.utc)
    path = "transcriptions/expired.wav"
    files.write(path, b"audio")
    add_job(database, "expired", path, "succeeded", (now - timedelta(days=31)).isoformat())

    def fail_unlink(_files, _relative):
        raise OSError("simulated unlink failure")

    monkeypatch.setattr(maintenance, "_unlink_confined", fail_unlink)
    result = cleanup_expired_task_audio(database, files, dry_run=False, now=now)

    assert database.get_job("expired")["file_path"] is None
    assert files.read(path) == b"audio"
    assert "simulated unlink failure" in result["failures"][0]["error"]
    database.close()
