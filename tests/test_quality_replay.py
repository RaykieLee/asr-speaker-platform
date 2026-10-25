import json
import os
import sqlite3

from scripts import run_diarization_quality as replay


def test_cli_writes_hashed_artifacts_and_exact_parameters(tmp_path, monkeypatch):
    result = {"segments": [{"start_seconds": 0, "end_seconds": 1, "text": "你好"}]}
    metrics = {"metrics": {"cer": {"value": 0.0}}}
    monkeypatch.setattr(replay, "run", lambda args: (result, metrics, {
        "schema_version": 1,
        "parameters": {"cluster_threshold": args.cluster_threshold, "num_speakers": args.num_speakers},
        "outputs": {},
    }))
    out = tmp_path / "artifacts"
    out.mkdir(mode=0o755)
    (out / "result.json").write_text("stale", encoding="utf-8")
    (out / "metrics.json").write_text("stale", encoding="utf-8")
    (out / "run-manifest.json").write_text("stale", encoding="utf-8")
    out.chmod(0o755)
    for artifact in out.iterdir():
        artifact.chmod(0o644)
    status = replay.main([
        "--audio", str(tmp_path / "input.wav"), "--annotation", str(tmp_path / "annotation.json"),
        "--registry", str(tmp_path / "registry.sqlite3"), "--asr-model-dir", str(tmp_path / "models"),
        "--segmentation-model", str(tmp_path / "seg.onnx"), "--speaker-model", str(tmp_path / "speaker.onnx"),
        "--speaker-model-version", "speaker-v1", "--cluster-threshold", "0.73", "--num-speakers", "2",
        "--output-dir", str(out),
    ])
    assert status == 0
    manifest = json.loads((out / "run-manifest.json").read_text())
    assert manifest["parameters"] == {"cluster_threshold": 0.73, "num_speakers": 2}
    assert manifest["outputs"]["result"]["sha256"] == replay.sha256(out / "result.json")
    assert manifest["outputs"]["metrics"]["sha256"] == replay.sha256(out / "metrics.json")
    assert os.stat(out).st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in out.iterdir())


def test_model_directory_requires_unambiguous_artifacts(tmp_path):
    for name in ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"):
        (tmp_path / name).touch()
    extra = tmp_path / "encoder-alt.onnx"
    extra.touch()
    try:
        replay._model_files(tmp_path)
    except replay.ReplayError as exc:
        assert "exactly one encoder" in str(exc)
    else:
        raise AssertionError("ambiguous model directory must fail")


def test_recognizer_uses_platform_modeling_unit_and_vocab(tmp_path, monkeypatch):
    from backend.app.engines.sherpa_offline import SherpaOfflineRecognizer

    model_dir = tmp_path / "offline-model"
    vocab = model_dir / "bpe.vocab"
    vocab.parent.mkdir()
    vocab.touch()
    artifacts = tuple(model_dir / name for name in
                      ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt")) + (vocab,)
    captured = {}

    def fake_load(cls, model, files, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(SherpaOfflineRecognizer, "load", classmethod(fake_load))
    replay._load_recognizer(model_dir, "offline-v1", artifacts)
    assert captured == {"modeling_unit": "cjkchar+bpe", "bpe_vocab": vocab}


def test_registration_wal_is_rejected_even_when_empty(tmp_path):
    database = tmp_path / "registry.sqlite3"
    database.touch()
    (tmp_path / "registry.sqlite3-wal").touch()
    try:
        replay._reject_wal(database)
    except replay.ReplayError as exc:
        assert "WAL file" in str(exc)
    else:
        raise AssertionError("existing registration WAL must fail closed")


def test_readonly_immutable_profile_connection_creates_no_sidecars(tmp_path):
    database = tmp_path / "registry.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE speakers (id TEXT, created_at TEXT)")
        connection.execute("CREATE TABLE speaker_profiles (speaker_id TEXT, embedding BLOB)")
    profiles = replay.ReadOnlyProfiles(database)
    try:
        assert profiles.list_speakers() == []
    finally:
        profiles.close()
    assert not (tmp_path / "registry.sqlite3-wal").exists()
    assert not (tmp_path / "registry.sqlite3-shm").exists()


def test_registry_change_during_replay_is_rejected(tmp_path):
    database = tmp_path / "registry.sqlite3"
    database.write_bytes(b"initial snapshot")
    initial_hash = replay.sha256(database)
    database.write_bytes(b"changed snapshot")
    try:
        replay._assert_unchanged(database, initial_hash)
    except replay.ReplayError as exc:
        assert "changed during replay" in str(exc)
    else:
        raise AssertionError("registry mutation during replay must fail closed")
