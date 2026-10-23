import array
import asyncio
import io
import math
import wave

import pytest
from fastapi.testclient import TestClient

from backend.app.audio import decode_wav
from backend.app.speakers import (
    EmbeddingModelVersionError, MissingSpeakerModelError, SampleQualityError,
    SpeakerModel, SpeakerRegistrationService, SherpaSpeakerEmbeddingAdapter,
    assess_audio_quality, l2_normalize,
)
from backend.app.speakers.service import SampleDurationTooLongError, SampleUploadTooLargeError
from backend.app.storage import FileStore, SQLiteStore
from backend.app.main import app as production_app


class TestExtractor:
    model_version = "test-model-v1"

    def extract(self, audio):
        # Deterministic test seam only. Production service requires a configured
        # SherpaSpeakerEmbeddingAdapter and never invents fallback vectors.
        return [float(len(audio.samples)), 3.0, 4.0]


def make_wav(values, rate=16000):
    data = array.array("h", values)
    if __import__("sys").byteorder != "little":
        data.byteswap()
    stream = io.BytesIO()
    with wave.open(stream := io.BytesIO(), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(data.tobytes())
    return stream.getvalue()


def tone(duration=1.25, amplitude=9000, rate=16000):
    return make_wav([int(amplitude * math.sin(2 * math.pi * 220 * i / rate))
                     for i in range(int(duration * rate))], rate)


def setup_service(tmp_path, extractor=None):
    database = SQLiteStore(tmp_path / "db.sqlite3")
    database.initialize()
    files = FileStore(tmp_path / "objects")
    return database, files, SpeakerRegistrationService(database, files, extractor)


def test_production_app_registers_and_persists_speaker_routes(tmp_path, monkeypatch):
    monkeypatch.setenv("ASR_DATA_DIR", str(tmp_path))
    with TestClient(production_app) as client:
        created = client.post("/api/v1/speakers", json={"name": "测试人员"})
        assert created.status_code == 200
        speaker_id = created.json()["id"]
        assert client.get(f"/api/v1/speakers/{speaker_id}").status_code == 200
        unavailable = client.post(
            f"/api/v1/speakers/{speaker_id}/samples",
            files={"file": ("sample.wav", tone(), "audio/wav")},
        )
        assert unavailable.status_code == 503
    with TestClient(production_app) as client:
        assert client.get(f"/api/v1/speakers/{speaker_id}").json()["name"] == "测试人员"


def test_quality_rejects_silence_short_and_clipped_samples():
    silent = decode_wav(make_wav([0] * 32000))
    with pytest.raises(SampleQualityError, match="静音"):
        assess_audio_quality(silent)
    short = decode_wav(tone(0.5))
    with pytest.raises(SampleQualityError, match="过短"):
        assess_audio_quality(short)
    clipped = decode_wav(make_wav([32767] * 32000))
    with pytest.raises(SampleQualityError, match="削波"):
        assess_audio_quality(clipped)
    quality = assess_audio_quality(decode_wav(tone()))
    assert quality["speech_seconds"] >= 1.0
    assert quality["clipped_ratio"] == 0


def test_no_configured_model_fails_without_persisting_sample(tmp_path):
    database, files, service = setup_service(tmp_path)
    speaker = service.create_speaker("测试人员")
    with pytest.raises(MissingSpeakerModelError, match="未配置"):
        service.add_sample(speaker["id"], "voice.wav", tone())
    assert database.list_samples(speaker["id"]) == []
    assert list((tmp_path / "objects").rglob("*")) == []


def test_upload_profile_restart_and_sample_delete_reaggregates(tmp_path):
    database, files, service = setup_service(tmp_path, TestExtractor())
    speaker = service.create_speaker("  林晓  ")
    first = service.add_sample(speaker["id"], "a.wav", tone())
    second = service.add_sample(speaker["id"], "b.wav", tone())
    profile = database.get_speaker_profile(speaker["id"])
    assert profile["sample_count"] == 2
    assert sum(v * v for v in profile["embedding"]) == pytest.approx(1.0)
    assert first["sample"]["model_version"] == "test-model-v1"
    assert service.delete_sample(speaker["id"], first["sample"]["id"])
    assert database.get_speaker_profile(speaker["id"])["sample_count"] == 1
    assert not files.exists(first["sample"]["audio_path"])
    database.close()
    reopened = SQLiteStore(tmp_path / "db.sqlite3")
    reopened.initialize()
    reloaded_profile = reopened.get_speaker_profile(speaker["id"])
    assert reloaded_profile["sample_count"] == 1
    assert sum(v * v for v in reloaded_profile["embedding"]) == pytest.approx(1.0)
    reopened.close()


def test_person_crud_and_delete_cleans_audio(tmp_path):
    database, files, service = setup_service(tmp_path, TestExtractor())
    person = service.create_speaker("原名")
    assert service.rename_speaker(person["id"], "新名字")["name"] == "新名字"
    sample = service.add_sample(person["id"], "voice.wav", tone())["sample"]
    assert service.delete_speaker(person["id"])
    assert database.get_speaker(person["id"]) is None
    assert not files.exists(sample["audio_path"])
    database.close()


def test_model_version_mismatch_refuses_mixed_profile(tmp_path):
    database, files, service = setup_service(tmp_path, TestExtractor())
    person = service.create_speaker("甲")
    service.add_sample(person["id"], "one.wav", tone())

    class OtherModel(TestExtractor):
        model_version = "test-model-v2"

    switched = SpeakerRegistrationService(database, files, OtherModel())
    with pytest.raises(EmbeddingModelVersionError, match="版本"):
        switched.add_sample(person["id"], "two.wav", tone())
    assert len(database.list_samples(person["id"])) == 1
    database.close()


def test_adapter_reports_missing_model_instead_of_faking_embedding(tmp_path):
    with pytest.raises(MissingSpeakerModelError, match="缺失"):
        SherpaSpeakerEmbeddingAdapter(SpeakerModel(tmp_path / "missing.onnx", "release-1"))


def test_l2_normalization_rejects_zero_or_nonfinite_vectors():
    assert l2_normalize([3, 4]) == pytest.approx([0.6, 0.8])
    with pytest.raises(Exception, match="零向量"):
        l2_normalize([0, 0])
    with pytest.raises(Exception, match="非有限"):
        l2_normalize([float("nan")])


def test_speaker_http_crud_contract(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.app.api.speakers import router

    database, files, service = setup_service(tmp_path, TestExtractor())
    app = FastAPI()
    app.state.speaker_registration_service = service
    app.include_router(router)
    with TestClient(app) as client:
        created = client.post("/api/v1/speakers", json={"name": "API 人员"})
        assert created.status_code == 200
        speaker_id = created.json()["id"]
        assert client.get("/api/v1/speakers").json()[0]["sample_count"] == 0
        assert client.patch(f"/api/v1/speakers/{speaker_id}", json={"name": "更新名字"}).json()["name"] == "更新名字"
        assert client.delete(f"/api/v1/speakers/{speaker_id}").json() == {"deleted": True}
        assert client.get(f"/api/v1/speakers/{speaker_id}").status_code == 404
    database.close()


def test_speaker_http_sample_upload_and_quality_error(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.app.api.speakers import router

    database, files, service = setup_service(tmp_path, TestExtractor())
    person = service.create_speaker("上传测试")
    app = FastAPI()
    app.state.speaker_registration_service = service
    app.include_router(router)
    with TestClient(app) as client:
        good = client.post(f"/api/v1/speakers/{person['id']}/samples",
                           files={"file": ("voice.wav", tone(), "audio/wav")})
        assert good.status_code == 201, good.text
        assert good.json()["sample"]["embedding"]
        silent = client.post(f"/api/v1/speakers/{person['id']}/samples",
                             files={"file": ("silent.wav", make_wav([0] * 32000), "audio/wav")})
        assert silent.status_code == 422
        sample_id = good.json()["sample"]["id"]
        assert client.delete(f"/api/v1/speakers/{person['id']}/samples/{sample_id}").json() == {"deleted": True}
    database.close()


def test_speaker_sample_limits_reject_before_decode_and_embedding_and_return_413(tmp_path, monkeypatch):
    from fastapi import FastAPI

    class CountingExtractor(TestExtractor):
        calls = 0

        def extract(self, audio):
            self.calls += 1
            return super().extract(audio)

    database = SQLiteStore(tmp_path / "db.sqlite3")
    database.initialize()
    files = FileStore(tmp_path / "objects")
    extractor = CountingExtractor()
    service = SpeakerRegistrationService(
        database, files, extractor, max_sample_bytes=100_000,
        max_sample_duration_seconds=1.0,
    )
    def fail_decode(data, filename):
        raise AssertionError("duration cap must run before decoding")

    monkeypatch.setattr(service, "_decode_upload", fail_decode)
    speaker = service.create_speaker("限制测试")

    with pytest.raises(SampleUploadTooLargeError):
        service.add_sample(speaker["id"], "large.wav", b"x" * 100_001)
    with pytest.raises(SampleDurationTooLongError):
        service.add_sample(speaker["id"], "long.wav", tone(1.25))
    assert extractor.calls == 0
    assert database.list_samples(speaker["id"]) == []

    app = FastAPI()
    app.state.speaker_registration_service = service
    from backend.app.api.speakers import router
    app.include_router(router)
    with TestClient(app) as client:
        too_large = client.post(
            f"/api/v1/speakers/{speaker['id']}/samples",
            files={"file": ("large.wav", b"x" * 100_001, "audio/wav")},
        )
        too_long = client.post(
            f"/api/v1/speakers/{speaker['id']}/samples",
            files={"file": ("long.wav", tone(1.25), "audio/wav")},
        )
        assert too_large.status_code == 413
        assert too_large.json()["detail"]["code"] == "sample_too_large"
        assert too_long.status_code == 413
        assert too_long.json()["detail"]["code"] == "sample_too_long"
    assert extractor.calls == 0
    database.close()


@pytest.mark.parametrize("content_length", [None, "1"])
def test_speaker_request_receive_cap_precedes_multipart_spooling(
    tmp_path, monkeypatch, content_length
):
    from fastapi import FastAPI, HTTPException
    from starlette.formparsers import SpooledTemporaryFile as original_spooled
    import starlette.formparsers as formparsers
    from starlette.requests import Request
    from backend.app.api.speakers import add_sample

    database, _, service = setup_service(tmp_path / "unused", TestExtractor())
    service.max_sample_bytes = 100
    app = FastAPI()
    app.state.speaker_registration_service = service

    created_files = []

    def track_spooled_file(*args, **kwargs):
        file_obj = original_spooled(*args, **kwargs)
        created_files.append(file_obj)
        return file_obj

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", track_spooled_file)
    body = (
        b"--boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.wav\"\r\n"
        b"Content-Type: audio/wav\r\n\r\n"
        + b"x" * (600 * 1024)
        + b"\r\n--boundary--\r\n"
    )
    chunks = [body[index:index + 16 * 1024] for index in range(0, len(body), 16 * 1024)]
    headers = [(b"content-type", b"multipart/form-data; boundary=boundary")]
    if content_length is not None:
        headers.append((b"content-length", content_length.encode("ascii")))
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/api/v1/speakers/s/samples",
        "raw_path": b"/api/v1/speakers/s/samples", "query_string": b"",
        "root_path": "", "headers": headers, "client": ("test", 1),
        "server": ("test", 80), "app": app,
    }

    async def receive():
        if chunks:
            chunk = chunks.pop(0)
            return {"type": "http.request", "body": chunk, "more_body": bool(chunks)}
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(scope, receive)
    with pytest.raises(HTTPException) as error:
        asyncio.run(add_sample("s", request))
    assert error.value.status_code == 413
    assert created_files
    assert all(file_obj.closed for file_obj in created_files)
    database.close()


def test_speaker_upload_form_is_closed_after_success(tmp_path):
    from fastapi import FastAPI
    from starlette.datastructures import FormData, UploadFile
    from starlette.requests import Request
    from backend.app.api.speakers import add_sample

    _, _, service = setup_service(tmp_path, TestExtractor())
    person = service.create_speaker("关闭资源测试")
    backing_file = io.BytesIO(tone())
    upload = UploadFile(filename="voice.wav", file=backing_file)
    form = FormData([("file", upload)])
    app = FastAPI()
    app.state.speaker_registration_service = service
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/", "raw_path": b"/",
        "query_string": b"", "root_path": "", "headers": [],
        "client": ("test", 1), "server": ("test", 80), "app": app,
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(scope, receive)
    request._form = form
    result = asyncio.run(add_sample(person["id"], request))
    assert result["sample"]["speaker_id"] == person["id"]
    assert backing_file.closed


def test_speaker_sample_byte_limit_requires_integer(tmp_path, monkeypatch):
    from backend.app.main import _positive_integer_setting
    from backend.app.engines.base import EngineConfigError

    setting = "ASR_TEST_POSITIVE_INT_SETTING"
    monkeypatch.delenv(setting, raising=False)
    assert _positive_integer_setting(setting, 64) == 64
    for raw in ("1.5", "0", "-1", " 1"):
        with pytest.raises(EngineConfigError, match="正整数"):
            monkeypatch.setenv(setting, raw)
            _positive_integer_setting(setting, 64)
    database, files, _ = setup_service(tmp_path, TestExtractor())
    with pytest.raises(ValueError, match="positive"):
        SpeakerRegistrationService(database, files, TestExtractor(), max_sample_bytes=1.5)
    database.close()
