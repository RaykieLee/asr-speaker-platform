from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.hotwords import router
from backend.app.engines import HotwordsUnsupportedError
from backend.app.hotwords import (
    HotwordCompilationError,
    HotwordCompiler,
    HotwordService,
    HotwordVocabularyUnavailableError,
    StaleHotwordVersionError,
)
from backend.app.storage import SQLiteStore


def test_production_package_import_path():
    from backend.app.api.hotwords import router as production_router

    assert production_router.prefix == "/api/v1/hotword-sets"


def _service(tmp_path, *, architecture="transducer", decoding_method="modified_beam_search", modeling_unit="cjkchar", vocabulary=None):
    store = SQLiteStore(tmp_path / "hotwords.sqlite3")
    store.initialize()
    vocab = vocabulary if vocabulary is not None else set("张三张三先生李梅OpenAI公司产品更新版".replace(" ", ""))
    compiler = HotwordCompiler(
        lambda text: list(text),
        architecture=architecture,
        decoding_method=decoding_method,
        modeling_unit=modeling_unit,
        tokenizer_id="test-character-tokenizer",
        vocabulary=vocab,
    )
    return store, HotwordService(store, compiler)


def test_create_compiles_entries_aliases_and_deduplicates_alias_within_entry(tmp_path):
    store, service = _service(tmp_path)
    created = service.create(
        name="  客户项目  ",
        entries=[{"word": "张三", "aliases": ["张三", "张三先生"], "language": "zh-CN"}, "李梅"],
        weight=2.5,
    )

    assert created["name"] == "客户项目"
    assert created["current_version"] == created["version"] == 1
    assert created["entries"][0] == {
        "word": "张三", "aliases": ["张三先生"], "language": "zh-CN", "enabled": True,
    }
    compiled = created["compiled"]
    assert compiled["decoding_method"] == "modified_beam_search"
    assert compiled["weight"] == 2.5
    assert [item["text"] for item in compiled["entries"]] == ["张三", "张三先生", "李梅"]
    assert compiled["hotwords_file_content"] == "张三\n张三先生\n李梅\n"
    assert compiled["entries"][0]["tokens"] == ["张", "三"]
    store.close()


def test_model_capability_vocab_and_unknown_token_fail_before_writing(tmp_path):
    store, service = _service(tmp_path, architecture="sense_voice")
    with pytest.raises(HotwordsUnsupportedError):
        service.create(name="unsupported", entries=["张三"])
    assert service.list_sets() == []
    store.close()

    store, service = _service(tmp_path / "greedy", decoding_method="greedy_search")
    with pytest.raises(HotwordsUnsupportedError, match="modified_beam_search"):
        service.create(name="wrong-decoder", entries=["张三"])
    assert service.list_sets() == []
    store.close()

    store, service = _service(tmp_path / "missing-vocab", modeling_unit="bpe", vocabulary=None)
    # The factory normally supplies model vocabulary; explicitly omit it here.
    service.compiler.vocabulary = None
    with pytest.raises(HotwordVocabularyUnavailableError, match="BPE"):
        service.create(name="bpe", entries=["OpenAI"])
    assert service.list_sets() == []
    store.close()


def test_bpe_compilation_uses_the_model_tokenizer_and_vocabulary(tmp_path):
    store = SQLiteStore(tmp_path / "bpe.sqlite3")
    store.initialize()
    compiler = HotwordCompiler(
        lambda text: ["Open", "AI"] if text == "OpenAI" else [],
        modeling_unit="bpe",
        tokenizer_id="bpe-test-vocab-v7",
        vocabulary={"Open", "AI"},
    )
    service = HotwordService(store, compiler)
    created = service.create(name="model terms", entries=["OpenAI"])
    assert created["compiled"]["modeling_unit"] == "bpe"
    assert created["compiled"]["tokenizer_id"] == "bpe-test-vocab-v7"
    assert created["compiled"]["entries"][0]["tokens"] == ["Open", "AI"]
    store.close()

    store, service = _service(tmp_path / "oov", vocabulary={"张", "三"})
    with pytest.raises(HotwordCompilationError, match="词表未覆盖"):
        service.create(name="oov", entries=["产品"])
    assert service.list_sets() == []
    store.close()


def test_versioned_resolver_keeps_old_snapshot_and_deactivation_blocks_new_jobs(tmp_path):
    store, service = _service(tmp_path)
    created = service.create(name="names", entries=["张三"])
    set_id = created["set_id"]
    updated = service.update(
        set_id, expected_version=1, entries=["李梅"], name="people", weight=3.0
    )

    assert updated["version"] == 2
    assert updated["name"] == "people"
    assert service.get(set_id, version=1)["entries"] == created["entries"]
    snapshot = service.resolve(set_id, version=1, mode="online")
    assert snapshot.version == 1
    assert snapshot.compiled["mode"] == "online"
    with pytest.raises(TypeError):
        snapshot.entries[0]["word"] = "mutated"
    assert snapshot.as_dict()["entries"][0]["word"] == "张三"
    with pytest.raises(StaleHotwordVersionError):
        service.update(set_id, expected_version=1, entries=["产品"])

    assert service.deactivate(set_id)
    with pytest.raises(LookupError, match="已停用"):
        service.resolve(set_id)
    assert service.resolve(set_id, version=1).version == 1
    store.close()


def test_optimistic_updates_allow_only_one_writer_at_a_version(tmp_path):
    store, service = _service(tmp_path)
    created = service.create(name="shared", entries=["张三"])
    set_id = created["set_id"]

    def update(word):
        try:
            return service.update(set_id, expected_version=1, entries=[word])
        except StaleHotwordVersionError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(update, ["李梅", "产品"]))
    assert sum(isinstance(result, dict) for result in results) == 1
    assert sum(isinstance(result, StaleHotwordVersionError) for result in results) == 1
    assert service.get(set_id)["current_version"] == 2
    store.close()


def test_http_routes_cover_crud_and_optimistic_conflict(tmp_path):
    store, service = _service(tmp_path)
    app = FastAPI()
    app.state.hotword_service = service
    app.include_router(router)

    with TestClient(app) as client:
        response = client.post("/api/v1/hotword-sets", json={
            "name": "names", "entries": [{"word": "张三", "aliases": ["张先生"]}],
        })
        assert response.status_code == 201
        created = response.json()
        set_id = created["set_id"]
        assert created["version"] == 1

        response = client.put(f"/api/v1/hotword-sets/{set_id}", json={
            "expected_version": 1,
            "entries": ["李梅"],
            "name": "renamed",
            "weight": 4,
        })
        assert response.status_code == 200
        assert response.json()["name"] == "renamed"
        assert response.json()["version"] == 2

        response = client.put(f"/api/v1/hotword-sets/{set_id}", json={
            "expected_version": 1, "entries": ["张三"],
        })
        assert response.status_code == 409
        assert response.json()["code"] == "HOTWORD_VERSION_CONFLICT"

        assert client.get(f"/api/v1/hotword-sets/{set_id}?version=1").json()["entries"] == created["entries"]
        assert client.get("/api/v1/hotword-sets").json()[0]["current_version"] == 2
        assert client.delete(f"/api/v1/hotword-sets/{set_id}").json()["status"] == "inactive"
        assert client.get("/api/v1/hotword-sets").json() == []
        assert client.get("/api/v1/hotword-sets?status=all").json()[0]["status"] == "inactive"
    store.close()


def test_api_reports_missing_bpe_vocab_and_unsupported_model(tmp_path):
    store, service = _service(tmp_path, modeling_unit="bpe")
    service.compiler.vocabulary = None
    app = FastAPI()
    app.state.hotword_service = service
    app.include_router(router)
    with TestClient(app) as client:
        response = client.post("/api/v1/hotword-sets", json={"name": "bpe", "entries": ["OpenAI"]})
        assert response.status_code == 422
        assert response.json()["code"] == "HOTWORD_VOCAB_UNAVAILABLE"
    store.close()

    store, service = _service(tmp_path / "unsupported-api", architecture="sense_voice")
    app = FastAPI()
    app.state.hotword_service = service
    app.include_router(router)
    with TestClient(app) as client:
        response = client.post("/api/v1/hotword-sets", json={"name": "sense", "entries": ["张三"]})
        assert response.status_code == 422
        assert response.json()["code"] == "HOTWORDS_UNSUPPORTED"
    store.close()
