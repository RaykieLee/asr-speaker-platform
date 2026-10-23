from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.app.main import app


def _disable_models(monkeypatch, tmp_path):
    monkeypatch.setenv("ASR_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ASR_ONLINE_MODEL_DIR", raising=False)
    monkeypatch.delenv("ASR_OFFLINE_MODEL_DIR", raising=False)
    monkeypatch.delenv("ASR_SPEAKER_MODEL_PATH", raising=False)
    monkeypatch.delenv("ASR_DIARIZATION_MODEL_PATH", raising=False)
    monkeypatch.delenv("ASR_API_TOKEN", raising=False)


def test_liveness_is_lightweight_and_readiness_fails_without_models(monkeypatch, tmp_path):
    _disable_models(monkeypatch, tmp_path)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

        ready = client.get("/ready")
        assert ready.status_code == 503
        assert ready.json() == {
            "status": "not_ready",
            "checks": {
                "initialization_complete": True,
                "database": True,
                "file_storage": True,
                "online_asr": False,
                "offline_asr": False,
                "speaker_embedding": False,
                "file_diarization": False,
            },
        }
        assert "text" not in ready.text and "audio" not in ready.text

        status = client.get("/api/v1/status")
        assert status.status_code == 200
        payload = status.json()
        assert payload["status"] == "not_ready"
        assert payload["file_queue"] == {
            "available": False, "queued": None, "running": None,
            "active": None, "capacity": None, "available_slots": None,
        }
        assert "file_path" not in status.text and "transcript" not in status.text


def test_status_reports_only_file_queue_counts_and_ready_when_fake_services_exist(monkeypatch, tmp_path):
    _disable_models(monkeypatch, tmp_path)
    with TestClient(app) as client:
        state = app.state
        database = state.database
        database.create_job(job_id="queued-private", file_path="private.wav", checksum="queued",
                            status="queued", options={"text": "PRIVATE TRANSCRIPT"})
        database.create_job(job_id="running-private", file_path="private.wav", checksum="running",
                            status="running", options={"text": "PRIVATE TRANSCRIPT"})
        service = SimpleNamespace(
            database=database, max_pending_jobs=10,
            asr=SimpleNamespace(diarizer=object()), close=lambda: None,
        )
        monkeypatch.setattr(state, "initialization_complete", False, raising=False)
        monkeypatch.setattr(state, "live_asr_service", object(), raising=False)
        monkeypatch.setattr(state, "transcription_service", service, raising=False)
        monkeypatch.setattr(state, "speaker_registration_service",
                            SimpleNamespace(extractor=object()), raising=False)

        incomplete = client.get("/ready")
        assert incomplete.status_code == 503
        assert incomplete.json()["checks"]["initialization_complete"] is False
        monkeypatch.setattr(state, "initialization_complete", True)
        assert client.get("/ready").status_code == 200
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        payload = response.json()
        assert payload["file_queue"] == {
            "available": True, "queued": 1, "running": 1,
            "active": 2, "capacity": 10, "available_slots": 8,
        }
        assert payload["status"] == "ready"
        assert "PRIVATE TRANSCRIPT" not in response.text
        assert "private.wav" not in response.text
