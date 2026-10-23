from __future__ import annotations

from copy import deepcopy

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.transcriptions import router


PRIVATE_PATH = "/private/machine/jobs/secret-upload.wav"
PRIVATE_VALUE = "internal-secret-value"


class FakeDatabase:
    def get_mapping_override(self, job_id, local_id):
        return {
            "job_id": job_id,
            "local_speaker_id": local_id,
            "registered_speaker_id": "person-1",
            "display_name": "Reviewed Name",
            "operator": "private-operator",
            "metadata": {"audit_log": [{"operator": "private-operator"}]},
        }

    def get_speaker(self, speaker_id):
        return {"id": speaker_id, "name": "Registered Name"} if speaker_id else None


class FakeService:
    max_upload_bytes = 10_000

    def __init__(self, job, *, create_error=None, lookup_error=None, retry_error=None):
        self.job = deepcopy(job)
        self.database = FakeDatabase()
        self.create_error = create_error
        self.lookup_error = lookup_error
        self.retry_error = retry_error

    async def create_job(self, *_args, **_kwargs):
        if self.create_error:
            raise self.create_error
        return deepcopy(self.job)

    def get_job(self, _job_id):
        if self.lookup_error:
            raise self.lookup_error
        return deepcopy(self.job)

    def retry_job(self, _job_id):
        if self.retry_error:
            raise self.retry_error
        retried = deepcopy(self.job)
        retried["status"] = "queued"
        retried["error_message"] = None
        return retried


def make_client(service):
    app = FastAPI()
    app.include_router(router)
    app.state.transcription_service = service
    return TestClient(app)


def raw_job(status="succeeded"):
    return {
        "id": "job-public-id",
        "status": status,
        "progress": 1.0,
        "created_at": "2026-10-24T00:00:00Z",
        "updated_at": "2026-10-24T00:00:01Z",
        "started_at": "2026-10-24T00:00:00Z",
        "finished_at": "2026-10-24T00:00:01Z",
        "file_path": PRIVATE_PATH,
        "checksum": PRIVATE_VALUE,
        "options": {"token": PRIVATE_VALUE, "debug_path": PRIVATE_PATH,
                    "hotwords": [["hello"]], "channel_mode": "separate"},
        "hotword_set_id": "hotwords-1",
        "hotword_version": 4,
        "model_version": "offline-v1",
        "error_code": "invalid_audio" if status == "failed" else None,
        "error_message": f"engine failure at {PRIVATE_PATH}: {PRIVATE_VALUE}" if status == "failed" else None,
        "result_metadata": {
            "duration": 2.0,
            "text": "A transcript needed by the web UI",
            "hotword_set_id": "hotwords-1",
            "hotword_set_version": 4,
            "channel_mode": "separate",
            "speaker_diarization": "completed",
            "speaker_mappings": [{
                "local_speaker_id": "speaker_0",
                "registered_speaker_id": "person-1",
                "speaker_name": "Alice",
                "match_status": "matched",
                "channel_index": 0,
                "embedding": [0.1, 0.2],
                "debug_path": PRIVATE_PATH,
            }],
            "debug": PRIVATE_VALUE,
        },
        "segments": [{
            "segment_id": 1,
            "start_seconds": 0.25,
            "end_seconds": 1.5,
            "text": "A transcript needed by the web UI",
            "local_speaker_id": "speaker_0",
            "registered_speaker_id": "person-1",
            "overlap": False,
            "revision": 1,
            "metadata": {"channel_index": 0, "speaker_name": "Alice",
                         "speaker_match_status": "matched", "embedding": [0.1, 0.2],
                         "debug_path": PRIVATE_PATH},
            "debug": PRIVATE_VALUE,
        }],
    }


def assert_no_private_fields(payload):
    serialized = str(payload)
    for sentinel in (PRIVATE_PATH, PRIVATE_VALUE, "embedding", "debug_path", "private-operator", "audit_log"):
        assert sentinel not in serialized
    assert "file_path" not in payload
    assert "checksum" not in payload
    assert "options" not in payload


def test_create_get_and_retry_return_only_public_task_fields():
    service = FakeService(raw_job("queued"))
    client = make_client(service)
    try:
        created = client.post("/api/v1/transcriptions", files={
            "file": ("meeting.wav", b"fake audio", "audio/wav"),
        }).json()
        assert created["job_id"] == "job-public-id"
        assert created["status"] == "queued"
        assert created["status_url"] == "/api/v1/transcriptions/job-public-id"
        assert created["hotword_set_id"] == "hotwords-1"
        assert created["hotword_set_version"] == 4
        assert created["model_version"] == "offline-v1"
        assert_no_private_fields(created)

        service.job = raw_job()
        fetched = client.get(created["status_url"]).json()
        assert fetched["job_id"] == created["job_id"]
        assert fetched["id"] == created["job_id"]
        assert fetched["status_url"] == created["status_url"]
        assert fetched["segments"][0]["metadata"] == {
            "channel_index": 0, "speaker_name": "Alice", "speaker_match_status": "matched",
        }
        assert fetched["effective_segments"][0]["speaker"]["display_name"] == "Reviewed Name"
        assert fetched["effective_segments"][0]["original_prediction"]["speaker_name"] == "Alice"
        assert fetched["result_metadata"]["text"] == "A transcript needed by the web UI"
        assert fetched["result_metadata"]["hotword_set_version"] == 4
        assert fetched["hotword_set_version"] == fetched["hotword_version"] == 4
        assert fetched["model_version"] == "offline-v1"
        assert_no_private_fields(fetched)

        retry_service = FakeService(raw_job("failed"))
        retry_client = make_client(retry_service)
        try:
            retried = retry_client.post("/api/v1/transcriptions/job-public-id/retry").json()
            assert retried["job_id"] == "job-public-id"
            assert retried["status"] == "queued"
            assert retried["status_url"] == "/api/v1/transcriptions/job-public-id"
            assert retried["hotword_set_version"] == 4
            assert_no_private_fields(retried)
        finally:
            retry_client.close()
    finally:
        client.close()


def test_internal_exceptions_use_safe_error_envelope_for_create_get_and_retry():
    create_client = make_client(FakeService(
        raw_job(), create_error=RuntimeError(f"traceback leaked {PRIVATE_PATH} {PRIVATE_VALUE}"),
    ))
    lookup_client = make_client(FakeService(
        raw_job(), lookup_error=RuntimeError(f"traceback leaked {PRIVATE_PATH} {PRIVATE_VALUE}"),
    ))
    retry_client = make_client(FakeService(
        raw_job("failed"), retry_error=RuntimeError(f"traceback leaked {PRIVATE_PATH} {PRIVATE_VALUE}"),
    ))
    try:
        create = create_client.post("/api/v1/transcriptions", files={
            "file": ("meeting.wav", b"fake audio", "audio/wav"),
        })
        get = lookup_client.get("/api/v1/transcriptions/job-public-id")
        retry = retry_client.post("/api/v1/transcriptions/job-public-id/retry")
        for response, code in ((create, "job_creation_failed"),
                               (get, "job_lookup_failed"),
                               (retry, "job_retry_failed")):
            assert response.status_code == 500
            body = response.json()
            assert set(body) == {"type", "code", "message", "request_id"}
            assert body["type"] == "error"
            assert body["code"] == code
            assert body["request_id"]
            assert PRIVATE_PATH not in response.text
            assert PRIVATE_VALUE not in response.text
    finally:
        create_client.close()
        lookup_client.close()
        retry_client.close()


def test_failed_task_hides_internal_exception_but_keeps_user_safe_status():
    client = make_client(FakeService(raw_job("failed")))
    try:
        response = client.get("/api/v1/transcriptions/job-public-id")
        assert response.status_code == 200
        job = response.json()
        assert job["status"] == "failed"
        assert job["error_code"] == "invalid_audio"
        assert job["error_message"] == "音频文件无法识别，请检查格式与内容"
        assert PRIVATE_PATH not in response.text
        assert PRIVATE_VALUE not in response.text
    finally:
        client.close()


def test_validation_errors_share_the_transcription_error_envelope():
    client = make_client(FakeService(raw_job()))
    try:
        correction = client.patch(
            "/api/v1/transcriptions/job-public-id/speakers/speaker_0",
            json={"display_name": "Alice", "operator": PRIVATE_VALUE},
        )
        export = client.get("/api/v1/transcriptions/job-public-id/export?format=xml")
        for response in (correction, export):
            assert response.status_code == 422
            body = response.json()
            assert set(body) == {"type", "code", "message", "request_id"}
            assert body["type"] == "error"
            assert body["request_id"]
            assert PRIVATE_VALUE not in response.text
    finally:
        client.close()
