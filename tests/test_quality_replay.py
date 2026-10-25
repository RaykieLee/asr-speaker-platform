import json

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
