import json
from pathlib import Path

import pytest

from backend.app.engines import (
    EngineConfigError,
    HotwordsUnsupportedError,
    MissingModelError,
    ModelArtifact,
    capabilities_for,
)
from models.download import DownloadError, download_model
from backend.app.engines.sherpa_online import OnlineTransducerFiles, SherpaOnlineRecognizer


def test_online_dynamic_hotwords_use_sherpa_separator():
    class FakeRecognizer:
        def create_stream(self, hotwords=None):
            return hotwords

    adapter = SherpaOnlineRecognizer(FakeRecognizer())
    assert adapter.create_stream(["你 好", "项 目"]) == "你 好/项 目"
    with pytest.raises(EngineConfigError):
        adapter.create_stream(["甲/乙"])


def test_online_bpe_model_requires_vocabulary_before_loading(tmp_path):
    paths = [tmp_path / name for name in ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt")]
    for path in paths:
        path.write_bytes(b"placeholder")
    artifact = ModelArtifact("online", "transducer", "v1", tuple(paths))
    files = OnlineTransducerFiles(*paths)
    with pytest.raises(EngineConfigError, match="bpe_vocab"):
        SherpaOnlineRecognizer.load(artifact, files, modeling_unit="cjkchar+bpe")


def test_transducer_hotword_capabilities_are_mode_specific():
    capabilities_for("transducer").require(hotwords=True, mode="online")
    capabilities_for("transducer").require(hotwords=True, mode="offline")
    with pytest.raises(HotwordsUnsupportedError, match="sense_voice"):
        capabilities_for("sense_voice").require(hotwords=True, mode="offline")


def test_unknown_architecture_and_mode_fail_explicitly():
    with pytest.raises(EngineConfigError, match="未知模型架构"):
        capabilities_for("imaginary")
    with pytest.raises(EngineConfigError, match="未知识别模式"):
        capabilities_for("transducer").require(mode="sideways")


def test_missing_model_has_version_and_remediation(tmp_path: Path):
    artifact = ModelArtifact("asr-zh", "transducer", "v1.2", (tmp_path / "missing.onnx",))
    with pytest.raises(MissingModelError, match="v1.2") as error:
        artifact.require_files()
    assert "models/README.md" in str(error.value)


def test_download_refuses_unpinned_candidate(tmp_path: Path):
    model = {"id": "candidate", "version_status": "待固定", "files": []}
    with pytest.raises(DownloadError, match="尚无已固定"):
        download_model(model, tmp_path)
    assert not list(tmp_path.iterdir())


def test_download_rejects_non_string_checksum_without_traceback(tmp_path: Path):
    model = {
        "id": "invalid-checksum",
        "files": [{"name": "model.onnx", "url": "https://example.invalid/model", "sha256": 123}],
    }
    with pytest.raises(DownloadError, match="SHA-256 格式无效"):
        download_model(model, tmp_path)
    assert not list(tmp_path.iterdir())


def test_download_verifies_before_renaming(tmp_path: Path, monkeypatch):
    import models.download as downloader

    payload = b"small test model"

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, size=-1):
            nonlocal payload
            chunk, payload = payload[:size], payload[size:]
            return chunk

    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda *a, **k: Response())
    model = {
        "id": "test-model",
        "files": [{"name": "model.onnx", "url": "https://example.invalid/model", "sha256": "a" * 64}],
    }
    with pytest.raises(DownloadError, match="不匹配"):
        download_model(model, tmp_path)
    assert not (tmp_path / "test-model" / "model.onnx").exists()
    assert not (tmp_path / "test-model" / "model.onnx.part").exists()


def test_manifest_explicitly_marks_unpinned_values():
    manifest = json.loads((Path(__file__).parents[1] / "models/manifest.json").read_text())
    assert manifest["runtime"]["version_status"] == "待固定"
    assert all(not model["enabled"] for model in manifest["models"])
    assert all(model["sha256"] is None for model in manifest["models"])
