import array
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
