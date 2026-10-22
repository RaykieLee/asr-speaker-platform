from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.app.api.auth import authorized_websocket
from backend.app.main import app


def test_http_api_key_protects_platform_routes_but_health_is_public(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("ASR_API_TOKEN", "local-secret")
    monkeypatch.setenv("ASR_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ASR_ONLINE_MODEL_DIR", raising=False)
    monkeypatch.delenv("ASR_OFFLINE_MODEL_DIR", raising=False)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/v1/speakers").status_code == 401
        assert client.get("/api/v1/speakers", headers={"Authorization": "Bearer wrong"}).status_code == 401
        response = client.get("/api/v1/status", headers={"Authorization": "Bearer local-secret"})
        assert response.status_code == 200
        assert response.json()["max_streams"] == 16
    assert "local-secret" not in caplog.text


def test_browser_websocket_key_must_match_configured_secret(monkeypatch):
    monkeypatch.setenv("ASR_API_TOKEN", "local-secret")
    socket = SimpleNamespace(headers={})
    assert not authorized_websocket(socket, {"api_key": "wrong"})
    assert authorized_websocket(socket, {"api_key": "local-secret"})
    socket.headers = {"authorization": "Bearer local-secret"}
    assert authorized_websocket(socket, {})
