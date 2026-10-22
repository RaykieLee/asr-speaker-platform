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
            json={"registered_speaker_id": alice["id"], "display_name": "Corrected Alice"},
            headers={"X-Operator": "spoofed-actor"},
        )
        assert response.status_code == 200
        assert response.json()["original_prediction"]["speaker_name"] == "Alice model prediction"

        response = client.patch(
            f"/api/v1/transcriptions/{job['id']}/speakers/speaker_0",
            json={"display_name": "Reviewed Alice"},
        )
        assert response.status_code == 200
        override = database.get_mapping_override(job["id"], "speaker_0")
        assert override["display_name"] == "Reviewed Alice"
        audit = override["metadata"]["audit_log"][1:]
        assert [item["operator"] for item in audit] == [None, None]
        assert all(item["actor_verified"] is False for item in audit)
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
        assert "声道" not in exported_txt.text + exported_srt.text
    finally:
        client.close()
        database.close()


def test_stereo_txt_and_srt_keep_interleaved_channel_and_speaker_labels(tmp_path):
    database, alice, job, client = setup_client(tmp_path)
    bob = database.create_speaker("Bob")
    try:
        database.update_job(job["id"], result_metadata={
            "duration": 3.75,
            "speaker_diarization": "completed",
            "speaker_mappings": [
                {"local_speaker_id": "speaker_0", "registered_speaker_id": alice["id"],
                 "speaker_name": "Alice", "match_status": "matched", "channel_index": 0},
                {"local_speaker_id": "speaker_1", "registered_speaker_id": bob["id"],
                 "speaker_name": "Bob", "match_status": "matched", "channel_index": 1},
            ],
        })
        database.upsert_segment(job["id"], 1, 0.2, 1.4, "Hello there",
                                local_speaker_id="speaker_0", registered_speaker_id=alice["id"],
                                metadata={"channel_index": 0})
        database.upsert_segment(job["id"], 2, 2.0, 3.75, "Goodbye",
                                local_speaker_id="speaker_0", registered_speaker_id=alice["id"],
                                metadata={"channel_index": 0})
        database.upsert_segment(job["id"], 3, 1.0, 1.6, "Middle turn",
                                local_speaker_id="speaker_1", registered_speaker_id=bob["id"],
                                metadata={"channel_index": 1})

        txt = client.get(f"/api/v1/transcriptions/{job['id']}/export?format=txt")
        srt = client.get(f"/api/v1/transcriptions/{job['id']}/export?format=srt")
        assert txt.status_code == srt.status_code == 200
        txt_lines = txt.text.splitlines()
        assert "[00:00:00.200 --> 00:00:01.400] [声道 1 · Alice] Hello there" == txt_lines[0]
        assert "[00:00:01.000 --> 00:00:01.600] [声道 2 · Bob] Middle turn" == txt_lines[1]
        assert "[00:00:02.000 --> 00:00:03.750] [声道 1 · Alice] Goodbye" == txt_lines[2]
        assert srt.text.index("[声道 1 · Alice] Hello there") < srt.text.index("[声道 2 · Bob] Middle turn")
        assert srt.text.index("[声道 2 · Bob] Middle turn") < srt.text.index("[声道 1 · Alice] Goodbye")
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
        assert client.patch(f"{base}/speaker_0", json={"display_name": "Alice\r\nInjected"}).status_code == 422
        assert client.patch(f"{base}/speaker_0", json={"display_name": "Alice\u2028Injected"}).status_code == 422
        assert client.patch(
            f"{base}/speaker_0", json={"display_name": "Alice", "operator": "spoofed"}
        ).status_code == 422
        assert client.get(f"/api/v1/transcriptions/missing/export?format=srt").status_code == 404
        assert client.get(f"/api/v1/transcriptions/{job['id']}/export?format=xml").status_code == 422
    finally:
        client.close()
        database.close()


def test_legacy_control_characters_cannot_add_export_lines(tmp_path):
    database, _, job, client = setup_client(tmp_path)
    try:
        database.set_mapping_override(job["id"], "speaker_0", display_name="Alice\r\nInjected")
        txt = client.get(f"/api/v1/transcriptions/{job['id']}/export?format=txt").text
        srt = client.get(f"/api/v1/transcriptions/{job['id']}/export?format=srt").text
        assert "[Alice Injected] Hello there" in txt
        assert "[Alice Injected] Hello there" in srt
        assert "\r" not in txt + srt
        assert "\nInjected]" not in txt + srt
    finally:
        client.close()
        database.close()
