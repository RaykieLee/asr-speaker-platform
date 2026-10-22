"""Storage contract tests: migrations, persistence, relationships and file safety."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from backend.app.storage import FileStore, SQLiteStore, UnsafeStoragePath


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
