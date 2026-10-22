"""FastAPI application entry point."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from backend.app.api.speakers import router as speakers_router
from backend.app.api.stream import router as stream_router
from backend.app.speakers.service import (
    SherpaSpeakerEmbeddingAdapter,
    SpeakerModel,
    SpeakerRegistrationService,
)
from backend.app.storage import FileStore, SQLiteStore, StorageService


@asynccontextmanager
async def lifespan(application: FastAPI):
    data_dir = Path(os.environ.get("ASR_DATA_DIR", "data")).expanduser()
    database = SQLiteStore(data_dir / "platform.sqlite3")
    database.initialize()
    files = FileStore(data_dir / "files")
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
    try:
        yield
    finally:
        database.close()

app = FastAPI(
    title="ASR 与说话人识别平台",
    version="0.1.0",
    description="本地部署的语音识别与说话人识别服务。",
    lifespan=lifespan,
)
app.include_router(stream_router)
app.include_router(speakers_router)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Return a lightweight process health response."""
    return {"status": "ok"}
