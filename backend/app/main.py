"""FastAPI application entry point."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from backend.app.api.hotwords import router as hotwords_router
from backend.app.api.speakers import router as speakers_router
from backend.app.api.stream import router as stream_router
from backend.app.api.transcriptions import router as transcriptions_router
from backend.app.engines.base import EngineConfigError, ModelArtifact
from backend.app.engines.sherpa_offline import OfflineTransducerFiles, SherpaOfflineRecognizer
from backend.app.engines.sherpa_online import OnlineTransducerFiles, SherpaOnlineRecognizer
from backend.app.hotwords.compiler import HotwordCompiler
from backend.app.hotwords.model_tokenizer import load_vocabulary, mixed_cjk_bpe_tokenizer
from backend.app.hotwords.service import HotwordService
from backend.app.jobs.service import TranscriptionService
from backend.app.pipelines.file_asr import FileASRService
from backend.app.pipelines.live_asr import LiveASRService
from backend.app.speakers.service import (
    SherpaSpeakerEmbeddingAdapter,
    SpeakerModel,
    SpeakerRegistrationService,
)
from backend.app.storage import FileStore, SQLiteStore, StorageService


def _pick_model_file(directory: Path, stem: str) -> Path:
    candidates = sorted(directory.glob(f"{stem}*.int8.onnx"))
    if not candidates:
        candidates = sorted(directory.glob(f"{stem}*.onnx"))
    if len(candidates) != 1:
        raise EngineConfigError(f"{directory} 应包含唯一的 {stem} ONNX 文件")
    return candidates[0]


def _asr_paths(directory: Path, *, online: bool):
    paths = (
        _pick_model_file(directory, "encoder"),
        _pick_model_file(directory, "decoder"),
        _pick_model_file(directory, "joiner"),
        directory / "tokens.txt",
    )
    files = OnlineTransducerFiles(*paths) if online else OfflineTransducerFiles(*paths)
    files.require_files()
    vocab = next((path for path in (directory / "bpe.vocab", directory / "bbpe.vocab") if path.is_file()), None)
    bpe_model = next((path for path in (directory / "bpe.model", directory / "bbpe.model") if path.is_file()), None)
    if vocab is None or bpe_model is None:
        raise EngineConfigError(f"{directory} 缺少与模型匹配的 BPE 模型及导出词表")
    return files, vocab, bpe_model


def _hotword_service(database: SQLiteStore, directory: Path, *, mode: str, bpe_model: Path) -> HotwordService:
    compiler = HotwordCompiler(
        mixed_cjk_bpe_tokenizer(bpe_model),
        architecture="transducer", decoding_method="modified_beam_search",
        mode=mode, tokenizer_id=directory.name,
        modeling_unit="cjkchar+bpe", vocabulary=load_vocabulary(directory / "tokens.txt"),
    )
    return HotwordService(database, compiler)


def _resolved_stream_words(service: HotwordService, set_id: str) -> tuple[list[str], int]:
    snapshot = service.resolve(set_id, mode="online")
    compiled = snapshot.as_dict()["compiled"]
    words = [" ".join(entry["tokens"]) for entry in compiled["entries"]]
    return words, snapshot.version


@asynccontextmanager
async def lifespan(application: FastAPI):
    data_dir = Path(os.environ.get("ASR_DATA_DIR", "data")).expanduser()
    database = SQLiteStore(data_dir / "platform.sqlite3")
    database.initialize()
    files = FileStore(data_dir / "files")
    try:
        application.state.live_asr_service = None
        application.state.hotword_service = None
        application.state.transcription_service = None
        extractor = None
        model_path = os.environ.get("ASR_SPEAKER_MODEL_PATH")
        if model_path:
            extractor = SherpaSpeakerEmbeddingAdapter(SpeakerModel(
                path=Path(model_path),
                version=os.environ.get("ASR_SPEAKER_MODEL_VERSION", ""),
            ))
        application.state.database = database
        application.state.files = files
        application.state.speaker_registration_service = SpeakerRegistrationService(
            database, files, extractor, storage=StorageService(database, files)
        )

        online_dir = os.environ.get("ASR_ONLINE_MODEL_DIR")
        if online_dir:
            directory = Path(online_dir)
            model_files, vocab, bpe_model = _asr_paths(directory, online=True)
            model = ModelArtifact("online-transducer", "transducer", directory.name,
                                  tuple(model_files.__dict__.values()))
            recognizer = SherpaOnlineRecognizer.load(
                model, model_files, modeling_unit="cjkchar+bpe", bpe_vocab=vocab,
                num_threads=int(os.environ.get("ASR_ONLINE_THREADS", "2")),
            )
            online_hotwords = _hotword_service(database, directory, mode="online", bpe_model=bpe_model)
            application.state.hotword_service = online_hotwords
            application.state.live_asr_service = LiveASRService(
                recognizer, hotword_resolver=lambda set_id: _resolved_stream_words(online_hotwords, set_id)
            )

        offline_dir = os.environ.get("ASR_OFFLINE_MODEL_DIR")
        if offline_dir:
            directory = Path(offline_dir)
            model_files, vocab, bpe_model = _asr_paths(directory, online=False)
            model = ModelArtifact("offline-transducer", "transducer", directory.name,
                                  tuple(model_files.__dict__.values()))
            recognizer = SherpaOfflineRecognizer.load(
                model, model_files, modeling_unit="cjkchar+bpe", bpe_vocab=vocab,
                num_threads=int(os.environ.get("ASR_OFFLINE_THREADS", "2")),
            )
            offline_hotwords = _hotword_service(database, directory, mode="offline", bpe_model=bpe_model)
            if not online_dir:
                application.state.hotword_service = offline_hotwords
            application.state.transcription_service = TranscriptionService(
                database, files, FileASRService(recognizer),
                hotword_resolver=lambda set_id: offline_hotwords.resolve(set_id, mode="offline"),
                max_workers=int(os.environ.get("ASR_FILE_WORKERS", "1")),
            )
        yield
    finally:
        service = getattr(application.state, "transcription_service", None)
        if service is not None:
            service.close()
        database.close()

app = FastAPI(
    title="ASR 与说话人识别平台",
    version="0.1.0",
    description="本地部署的语音识别与说话人识别服务。",
    lifespan=lifespan,
)
app.include_router(stream_router)
app.include_router(speakers_router)
app.include_router(hotwords_router)
app.include_router(transcriptions_router)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Return a lightweight process health response."""
    return {"status": "ok"}
