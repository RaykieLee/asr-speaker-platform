"""FastAPI application entry point."""

from fastapi import FastAPI

from backend.app.api.stream import router as stream_router

app = FastAPI(
    title="ASR 与说话人识别平台",
    version="0.1.0",
    description="本地部署的语音识别与说话人识别服务。",
)
app.include_router(stream_router)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Return a lightweight process health response."""
    return {"status": "ok"}
