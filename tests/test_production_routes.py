from fastapi.testclient import TestClient

from backend.app.main import app


def test_production_app_exposes_hotword_and_file_routes_without_models(tmp_path, monkeypatch):
    monkeypatch.setenv("ASR_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ASR_ONLINE_MODEL_DIR", raising=False)
    monkeypatch.delenv("ASR_OFFLINE_MODEL_DIR", raising=False)
    with TestClient(app) as client:
        hotwords = client.get("/api/v1/hotword-sets")
        assert hotwords.status_code == 503
        transcription = client.get("/api/v1/transcriptions/missing")
        assert transcription.status_code == 503
