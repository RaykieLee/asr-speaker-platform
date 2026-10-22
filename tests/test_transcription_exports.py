from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.transcriptions import router
from backend.app.storage import SQLiteStore


class TranscriptionServiceStub:
    def __init__(self, database):
        self.database = database

    def get_job(self, job_id):
        job = self.database.get_job(job_id)
        if job is not None:
            job["segments"] = self.database.list_segments(job_id)
        return job


def setup_client(tmp_path):
    database = SQLiteStore(tmp_path / "exports.sqlite3")
    database.initialize()
    alice = database.create_speaker("Alice")
    job = database.create_job(model_version="offline-test")
    database.update_job(job["id"], status="succeeded", result_metadata={
        "duration": 3.75,
        "speaker_diarization": "completed",
        "speaker_mappings": [{
            "local_speaker_id": "speaker_0",
            "registered_speaker_id": alice["id"],
            "speaker_name": "Alice model prediction",
            "match_status": "matched",
        }],
    })
    database.upsert_segment(job["id"], 1, 0.2, 1.4, "Hello there",
                            local_speaker_id="speaker_0",
                            registered_speaker_id=alice["id"], metadata={"speaker_name": "Alice"})
    database.upsert_segment(job["id"], 2, 2.0, 3.75, "Goodbye",
                            local_speaker_id="speaker_0",
                            registered_speaker_id=alice["id"])
    app = FastAPI()
    app.state.transcription_service = TranscriptionServiceStub(database)
    app.include_router(router)
    return database, alice, job, TestClient(app)


def test_correction_preserves_model_prediction_and_appends_audit(tmp_path):
    database, alice, job, client = setup_client(tmp_path)
    try:
        response = client.patch(
            f"/api/v1/transcriptions/{job['id']}/speakers/speaker_0",
            json={"registered_speaker_id": alice["id"], "display_name": "Corrected Alice",
                  "operator": "reviewer-1"},
        )
        assert response.status_code == 200
        assert response.json()["original_prediction"]["speaker_name"] == "Alice model prediction"

        response = client.patch(
            f"/api/v1/transcriptions/{job['id']}/speakers/speaker_0",
            json={"display_name": "Reviewed Alice", "operator": "reviewer-2"},
        )
        assert response.status_code == 200
        override = database.get_mapping_override(job["id"], "speaker_0")
        assert override["display_name"] == "Reviewed Alice"
        assert [item["operator"] for item in override["metadata"]["audit_log"][1:]] == [
            "reviewer-1", "reviewer-2",
        ]
        fetched = client.get(f"/api/v1/transcriptions/{job['id']}").json()
        assert fetched["effective_segments"][0]["speaker"]["display_name"] == "Reviewed Alice"
        assert fetched["segments"][0]["registered_speaker_id"] == alice["id"]
        # Corrections live alongside the raw model prediction; they do not rewrite it.
        raw = database.list_segments(job["id"])[0]
        assert raw["registered_speaker_id"] == alice["id"]
        assert database.get_job(job["id"])["result_metadata"]["speaker_mappings"][0]["speaker_name"] == "Alice model prediction"
    finally:
        client.close()
        database.close()


def test_exports_share_corrected_names_and_timeline(tmp_path):
    database, alice, job, client = setup_client(tmp_path)
    try:
        corrected = client.patch(
            f"/api/v1/transcriptions/{job['id']}/speakers/speaker_0",
            json={"registered_speaker_id": alice["id"], "display_name": "张三"},
        )
        assert corrected.status_code == 200

        exported_json = client.get(f"/api/v1/transcriptions/{job['id']}/export?format=json")
        exported_txt = client.get(f"/api/v1/transcriptions/{job['id']}/export?format=txt")
        exported_srt = client.get(f"/api/v1/transcriptions/{job['id']}/export?format=srt")
        assert [exported_json.status_code, exported_txt.status_code, exported_srt.status_code] == [200] * 3
        payload = exported_json.json()
        assert payload["segments"][0]["original_prediction"]["speaker_name"] == "Alice model prediction"
        assert payload["segments"][0]["speaker"]["display_name"] == "张三"
        assert payload["segments"][0]["start_seconds"] == 0.2
        assert "[张三] Hello there" in exported_txt.text
        assert "00:00:00,200 --> 00:00:01,400" in exported_srt.text
        assert "[张三] Hello there" in exported_srt.text
        assert "00:00:02,000 --> 00:00:03,750" in exported_srt.text
    finally:
        client.close()
        database.close()


def test_correction_rejects_unknown_job_speaker_and_person(tmp_path):
    database, _, job, client = setup_client(tmp_path)
    try:
        base = f"/api/v1/transcriptions/{job['id']}/speakers"
        assert client.patch(f"{base}/speaker_9", json={"display_name": "Other"}).status_code == 404
        assert client.patch(f"{base}/speaker_0", json={"registered_speaker_id": "missing"}).status_code == 404
        assert client.patch(f"{base}/speaker_0", json={}).status_code == 422
        assert client.get(f"/api/v1/transcriptions/missing/export?format=srt").status_code == 404
        assert client.get(f"/api/v1/transcriptions/{job['id']}/export?format=xml").status_code == 422
    finally:
        client.close()
        database.close()
