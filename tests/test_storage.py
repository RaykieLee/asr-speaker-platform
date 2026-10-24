"""Storage contract tests: migrations, persistence, relationships and file safety."""

from concurrent.futures import ThreadPoolExecutor
import stat
from pathlib import Path

import pytest

from backend.app.storage import FileStore, SQLiteStore, StorageService, UnsafeStoragePath


def test_database_and_wal_files_are_private(tmp_path):
    parent = tmp_path / "private-data"
    store = SQLiteStore(parent / "storage.sqlite3")
    store.initialize()
    store.create_speaker("private")
    assert stat.S_IMODE(parent.stat().st_mode) == 0o700
    assert stat.S_IMODE((parent / "storage.sqlite3").stat().st_mode) == 0o600
    for suffix in ("-wal", "-shm"):
        sidecar = parent / f"storage.sqlite3{suffix}"
        if sidecar.exists():
            assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600
    store.close()


def test_migration_is_idempotent_and_persisted_rows_survive_reopen(tmp_path):
    database = tmp_path / "data" / "asr.sqlite3"
    store = SQLiteStore(database)
    store.initialize()
    store.initialize()
    speaker = store.create_speaker("Ari", metadata={"language": "zh"})
    sample = store.create_sample(speaker["id"], "speakers/a.wav", duration_seconds=2.5,
                                 quality={"snr": 18}, model_version="embed-v1",
                                 embedding=[0.25, -0.5, 1.0])
    store.set_speaker_profile(speaker["id"], [0.1, 0.2], sample_count=1,
                              model_version="embed-v1")
    job = store.create_job(file_path="jobs/a.wav", options={"language": "zh"})
    store.update_job(job["id"], status="running", progress=0.4,
                     started_at="2026-10-23T00:00:00+00:00")
    segment = store.upsert_segment(job["id"], 1, 0.2, 1.4, "hello",
                                   registered_speaker_id=speaker["id"], overlap=True)
    hotwords = store.create_hotword_set("names", ["Ari", "李"])
    store.close()

    reopened = SQLiteStore(database)
    reopened.initialize()
    assert reopened.get_speaker(speaker["id"])["metadata"] == {"language": "zh"}
    assert reopened.get_sample(sample["id"])["embedding"] == pytest.approx([0.25, -0.5, 1.0])
    assert reopened.get_speaker_profile(speaker["id"])["embedding"] == pytest.approx([0.1, 0.2])
    assert reopened.get_job(job["id"])["status"] == "running"
    assert reopened.get_job(job["id"])["options"] == {"language": "zh"}
    assert reopened.list_segments(job["id"])[0]["overlap"] is True
    assert reopened.list_segments(job["id"])[0]["text"] == segment["text"]
    assert reopened.get_hotword_version(hotwords["set_id"], 1)["entries"] == ["Ari", "李"]
    reopened.close()


def test_speaker_and_sample_deletes_apply_relationship_policy(tmp_path):
    store = SQLiteStore(tmp_path / "db.sqlite3")
    store.initialize()
    person = store.create_speaker("Ari")
    sample = store.create_sample(person["id"], "s/one.wav", embedding=[1.0])
    store.set_speaker_profile(person["id"], [1.0], sample_count=1)
    assert store.delete_sample(sample["id"])
    assert store.get_sample(sample["id"]) is None
    assert store.get_speaker_profile(person["id"]) is None
    sample = store.create_sample(person["id"], "s/two.wav")
    job = store.create_job()
    store.upsert_segment(job["id"], 0, 0, 1, "hi", registered_speaker_id=person["id"])
    store.set_mapping_override(job["id"], "speaker_0", registered_speaker_id=person["id"])
    assert store.delete_speaker(person["id"])
    assert store.get_speaker(person["id"]) is None
    assert store.get_sample(sample["id"]) is None
    assert store.list_segments(job["id"])[0]["registered_speaker_id"] is None
    assert store.get_mapping_override(job["id"], "speaker_0")["registered_speaker_id"] is None
    assert store.delete_speaker(person["id"]) is False
    store.close()


def test_threaded_repository_writes_are_serialized_without_lost_rows(tmp_path):
    store = SQLiteStore(tmp_path / "threaded.sqlite3")
    store.initialize()
    count = 120

    def write(index):
        return store.create_speaker(f"speaker-{index}")["id"]

    with ThreadPoolExecutor(max_workers=12) as pool:
        ids = list(pool.map(write, range(count)))
    assert len(set(ids)) == count
    assert len(store.list_speakers()) == count
    store.close()


def test_hotword_snapshots_use_optimistic_version_check(tmp_path):
    store = SQLiteStore(tmp_path / "db.sqlite3")
    store.initialize()
    current = store.create_hotword_set("terms", ["first"])
    updated = store.append_hotword_version(current["set_id"], ["second"], expected_version=1)
    assert updated["version"] == 2
    assert store.get_hotword_version(current["set_id"], 1)["entries"] == ["first"]
    with pytest.raises(ValueError, match="stale"):
        store.append_hotword_version(current["set_id"], ["lost update"], expected_version=1)
    job = store.create_job(hotword_set_id=current["set_id"], hotword_version=2)
    assert store.delete_hotword_set(current["set_id"])
    assert store.get_hotword_set(current["set_id"])["status"] == "inactive"
    assert store.get_hotword_version(current["set_id"], 1)["entries"] == ["first"]
    assert store.get_hotword_version(current["set_id"], 2)["entries"] == ["second"]
    assert store.get_job(job["id"])["hotword_set_id"] == current["set_id"]
    store.close()


def test_storage_service_removes_sample_audio_and_preserves_shared_files(tmp_path):
    store = SQLiteStore(tmp_path / "db.sqlite3")
    store.initialize()
    files = FileStore(tmp_path / "objects")
    service = StorageService(store, files)
    first = store.create_speaker("first")
    second = store.create_speaker("second")
    files.write("samples/shared.wav", b"shared")
    files.write("samples/only.wav", b"only")
    shared_one = store.create_sample(first["id"], "samples/shared.wav")
    only = store.create_sample(first["id"], "samples/only.wav")
    store.create_sample(second["id"], "samples/shared.wav")

    assert service.delete_sample(shared_one["id"])
    assert files.exists("samples/shared.wav")
    assert service.delete_sample(only["id"])
    assert not files.exists("samples/only.wav")
    store.close()


def test_storage_service_deleting_speaker_cleans_all_sample_files(tmp_path):
    store = SQLiteStore(tmp_path / "db.sqlite3")
    store.initialize()
    files = FileStore(tmp_path / "objects")
    service = StorageService(store, files)
    person = store.create_speaker("registered")
    files.write("samples/a.wav", b"a")
    files.write("samples/b.wav", b"b")
    store.create_sample(person["id"], "samples/a.wav")
    store.create_sample(person["id"], "samples/b.wav")
    job = store.create_job(file_path="jobs/upload.wav")
    files.write("jobs/upload.wav", b"job audio")

    assert service.delete_speaker(person["id"])
    assert not files.exists("samples/a.wav")
    assert not files.exists("samples/b.wav")
    # Uploaded transcription audio has a different owner and is retained.
    assert files.exists("jobs/upload.wav")
    assert store.get_job(job["id"]) is not None
    store.close()


def test_file_store_atomic_operations_and_path_traversal_rejection(tmp_path):
    root = tmp_path / "objects"
    outside = tmp_path / "outside.txt"
    outside.write_text("safe")
    files = FileStore(root)
    target = files.write("jobs/2026/audio.wav", b"audio payload")
    assert target.read_bytes() == b"audio payload"
    assert files.read("jobs/2026/audio.wav") == b"audio payload"
    assert files.delete("jobs/2026/audio.wav") is True
    assert files.delete("jobs/2026/audio.wav") is False
    for path in ("../outside.txt", str(outside), "nested/../../outside.txt"):
        with pytest.raises(UnsafeStoragePath):
            files.write(path, b"no")
        with pytest.raises(UnsafeStoragePath):
            files.delete(path)
    assert outside.read_text() == "safe"


def test_file_store_creates_private_root_nested_directories_and_atomic_files(tmp_path, monkeypatch):
    from backend.app.storage import files as files_module

    root = tmp_path / "new" / "objects"
    files = FileStore(root)
    created = files.write("jobs/2026/audio.wav", b"audio payload")

    assert root.stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "new").stat().st_mode & 0o777 == 0o700
    assert (root / "jobs").stat().st_mode & 0o777 == 0o700
    assert (root / "jobs" / "2026").stat().st_mode & 0o777 == 0o700
    assert created.stat().st_mode & 0o777 == 0o600

    observed = {}
    real_replace = files_module.os.replace

    def inspect_temporary(source, destination):
        observed["mode"] = Path(source).stat().st_mode & 0o777
        return real_replace(source, destination)

    monkeypatch.setattr(files_module.os, "replace", inspect_temporary)
    files.write("jobs/2026/second.wav", b"second payload")
    assert observed["mode"] == 0o600


def test_file_store_does_not_change_existing_shared_directories_or_symlink_target(tmp_path):
    shared_root = tmp_path / "shared"
    shared_root.mkdir(mode=0o755)
    shared_root.chmod(0o755)
    existing = shared_root / "already-shared"
    existing.mkdir(mode=0o755)
    existing.chmod(0o755)

    files = FileStore(shared_root)
    files.write("already-shared/new-branch/payload.bin", b"private file")
    assert shared_root.stat().st_mode & 0o777 == 0o755
    assert existing.stat().st_mode & 0o777 == 0o755
    assert (existing / "new-branch").stat().st_mode & 0o777 == 0o700
    assert (existing / "new-branch" / "payload.bin").stat().st_mode & 0o777 == 0o600

    alias = tmp_path / "shared-alias"
    alias.symlink_to(shared_root, target_is_directory=True)
    FileStore(alias)
    assert alias.is_symlink()
    assert shared_root.stat().st_mode & 0o777 == 0o755


def test_file_store_rejects_symlink_escape(tmp_path):
    root = tmp_path / "objects"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    files = FileStore(root)
    with pytest.raises(UnsafeStoragePath):
        files.write("link/payload.bin", b"no")
    assert not (outside / "payload.bin").exists()
