"""FastAPI application entry point."""

import asyncio
import os
import stat
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.app.api.auth import api_key_middleware
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
from backend.app.speakers.live_diarization import profile_cosine_matcher
from backend.app.speakers.diarization import SherpaOfflineDiarizer
from backend.app.speakers.matching import SpeakerMatcher
from backend.app.speakers.vad import EnergyVAD
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


def _speaker_profiles(database: SQLiteStore) -> list[dict]:
    profiles = []
    for speaker in database.list_speakers(status="active"):
        profile = database.get_speaker_profile(speaker["id"])
        if profile and profile.get("embedding"):
            profiles.append({"id": speaker["id"], "name": speaker["name"],
                             "embedding": profile["embedding"],
                             "model_version": profile["model_version"]})
    return profiles


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.initialization_complete = False
    data_dir = Path(os.environ.get("ASR_DATA_DIR", "data")).expanduser()
    database = SQLiteStore(data_dir / "platform.sqlite3")
    database.initialize()
    files = FileStore(data_dir / "files")
    try:
        application.state.active_streams = 0
        application.state.stream_lock = asyncio.Lock()
        application.state.max_streams = int(os.environ.get("ASR_MAX_STREAMS", "16"))
        if application.state.max_streams < 1:
            raise EngineConfigError("ASR_MAX_STREAMS 必须为正整数")
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
                recognizer, hotword_resolver=lambda set_id: _resolved_stream_words(online_hotwords, set_id),
                vad=EnergyVAD() if extractor is not None else None,
                speaker_extractor=extractor,
                speaker_matcher=profile_cosine_matcher(
                    min_similarity=float(os.environ.get("ASR_LIVE_SPEAKER_MIN_SIMILARITY", "0.55")),
                    min_margin=float(os.environ.get("ASR_LIVE_SPEAKER_MIN_MARGIN", "0.05")),
                ) if extractor is not None else None,
                speaker_profiles=lambda: _speaker_profiles(database),
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
            diarization_path = os.environ.get("ASR_DIARIZATION_MODEL_PATH")
            if diarization_path and not model_path:
                raise EngineConfigError("配置离线说话人分段时还需要 ASR_SPEAKER_MODEL_PATH")
            diarizer = SherpaOfflineDiarizer.load(
                diarization_path, model_path,
                num_threads=int(os.environ.get("ASR_DIARIZATION_THREADS", "2")),
                num_speakers=int(os.environ.get("ASR_DIARIZATION_NUM_SPEAKERS", "-1")),
            ) if diarization_path and model_path else None
            speaker_matcher = SpeakerMatcher(
                database, extractor,
                min_similarity=float(os.environ.get("ASR_FILE_SPEAKER_MIN_SIMILARITY", "0.55")),
                min_margin=float(os.environ.get("ASR_FILE_SPEAKER_MIN_MARGIN", "0.05")),
                min_segment_seconds=float(os.environ.get("ASR_FILE_SPEAKER_MIN_SEGMENT_SECONDS", "1.0")),
            ) if diarizer is not None and extractor is not None else None
            application.state.transcription_service = TranscriptionService(
                database, files, FileASRService(recognizer, diarizer=diarizer,
                                               speaker_matcher=speaker_matcher),
                hotword_resolver=lambda set_id: offline_hotwords.resolve(set_id, mode="offline"),
                max_workers=int(os.environ.get("ASR_FILE_WORKERS", "1")),
                max_upload_bytes=int(os.environ.get("ASR_MAX_UPLOAD_BYTES", str(512 * 1024 * 1024))),
                max_pending_jobs=int(os.environ.get("ASR_MAX_PENDING_JOBS", "32")),
            )
        application.state.initialization_complete = True
        yield
    finally:
        application.state.initialization_complete = False
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
app.middleware("http")(api_key_middleware)
app.include_router(stream_router)
app.include_router(speakers_router)
app.include_router(hotwords_router)
app.include_router(transcriptions_router)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Return a lightweight process health response."""
    return {"status": "ok"}


def _readiness_snapshot(state) -> dict:
    transcription = getattr(state, "transcription_service", None)
    registration = getattr(state, "speaker_registration_service", None)
    database = getattr(state, "database", None)
    queue_available = database is not None
    if queue_available:
        try:
            database.count_jobs(status="queued")
            database.count_jobs(status="running")
        except Exception:
            queue_available = False

    checks = {
        "initialization_complete": bool(getattr(state, "initialization_complete", False)),
        "database": queue_available,
        "file_storage": _file_storage_ready(getattr(state, "files", None)),
        "online_asr": getattr(state, "live_asr_service", None) is not None,
        "offline_asr": transcription is not None,
        "speaker_embedding": registration is not None and getattr(registration, "extractor", None) is not None,
        "file_diarization": transcription is not None and
        getattr(getattr(transcription, "asr", None), "diarizer", None) is not None,
    }
    return {
        "status": "ready" if all(checks.values()) else "not_ready",
        "checks": checks,
    }


def _file_storage_ready(files) -> bool:
    """Probe the configured backing directory without exposing its path."""
    root = getattr(files, "root", None)
    if not isinstance(root, Path):
        return False
    probe_path = None
    probe_fd = None
    marker = os.urandom(16)
    ready = False
    try:
        mode = root.stat().st_mode
        has_read_write_search_bits = bool(mode & 0o444 and mode & 0o222 and mode & 0o111)
        if (stat.S_ISDIR(mode) and has_read_write_search_bits
                and os.access(root, os.R_OK | os.W_OK | os.X_OK)):
            probe_fd, probe_path = tempfile.mkstemp(prefix=".asr-ready-", dir=root)
            descriptor = probe_fd
            probe_fd = None
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(marker)
                stream.flush()
            ready = Path(probe_path).read_bytes() == marker
    except OSError:
        ready = False
    finally:
        if probe_fd is not None:
            try:
                os.close(probe_fd)
            except OSError:
                ready = False
        if probe_path is not None:
            try:
                os.unlink(probe_path)
            except FileNotFoundError:
                pass
            except OSError:
                ready = False
    return ready


def _file_queue_snapshot(transcription) -> dict:
    if transcription is None:
        return {
            "available": False, "queued": None, "running": None,
            "active": None, "capacity": None, "available_slots": None,
        }
    capacity = getattr(transcription, "max_pending_jobs", None)
    if not isinstance(capacity, int) or capacity < 1:
        return {
            "available": False, "queued": None, "running": None,
            "active": None, "capacity": None, "available_slots": None,
        }
    try:
        queued = transcription.database.count_jobs(status="queued")
        running = transcription.database.count_jobs(status="running")
    except Exception:
        return {
            "available": False, "queued": None, "running": None,
            "active": None, "capacity": capacity, "available_slots": None,
        }
    active = queued + running
    return {
        "available": True, "queued": queued, "running": running,
        "active": active, "capacity": capacity,
        "available_slots": max(0, capacity - active),
    }


@app.get("/ready", tags=["health"])
def readiness(request: Request):
    """Report whether the complete ASR service stack can accept work."""
    snapshot = _readiness_snapshot(request.app.state)
    return JSONResponse(status_code=200 if snapshot["status"] == "ready" else 503,
                        content=snapshot)


@app.get("/api/v1/status", tags=["health"])
def service_status(request: Request) -> dict:
    state = request.app.state
    transcription = getattr(state, "transcription_service", None)
    registration = getattr(state, "speaker_registration_service", None)
    return {
        **_readiness_snapshot(state),
        "online_asr": getattr(state, "live_asr_service", None) is not None,
        "offline_asr": transcription is not None,
        "speaker_embedding": registration is not None and getattr(registration, "extractor", None) is not None,
        "file_diarization": transcription is not None and
        getattr(getattr(transcription, "asr", None), "diarizer", None) is not None,
        "active_streams": getattr(state, "active_streams", 0),
        "max_streams": getattr(state, "max_streams", None),
        "file_queue": _file_queue_snapshot(transcription),
    }
