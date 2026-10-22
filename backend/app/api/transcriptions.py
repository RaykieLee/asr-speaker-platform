"""HTTP routes for uploading audio and tracking durable transcription jobs."""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from backend.app.jobs.service import HotwordSnapshotError, JobQueueFull, TranscriptionService, UploadTooLarge

router = APIRouter()


def _service(request: Request) -> TranscriptionService | None:
    return getattr(request.app.state, "transcription_service", None)


def _error(status: int, code: str, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={
        "type": "error", "code": code, "message": message, "request_id": request_id,
    })


async def _read_upload(upload: Any, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(min(1024 * 1024, limit + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > limit:
            raise UploadTooLarge(f"上传文件超过 {limit} 字节限制")
    return b"".join(chunks)


@router.post("/api/v1/transcriptions", status_code=202)
async def create_transcription(request: Request) -> Any:
    request_id = uuid.uuid4().hex
    service = _service(request)
    if service is None:
        return _error(503, "service_unavailable", "文件识别服务未配置", request_id)
    try:
        form = await request.form()
    except Exception as exc:
        return _error(400, "invalid_request", f"无法解析上传表单: {exc}", request_id)
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read") or not getattr(upload, "filename", None):
        return _error(400, "invalid_request", "multipart 表单必须包含 file 文件", request_id)
    try:
        content = await _read_upload(upload, service.max_upload_bytes)
        job = await service.create_job(
            content, upload.filename,
            language=str(form.get("language", "auto")),
            channel_mode=str(form.get("channel_mode", "mixdown")),
            hotword_set_id=(str(form["hotword_set_id"]) if form.get("hotword_set_id") else None),
        )
    except UploadTooLarge as exc:
        return _error(413, "upload_too_large", str(exc), request_id)
    except JobQueueFull as exc:
        return _error(429, "queue_full", str(exc), request_id)
    except HotwordSnapshotError as exc:
        return _error(422, "hotword_snapshot_failed", str(exc), request_id)
    except ValueError as exc:
        return _error(422, "invalid_request", str(exc), request_id)
    except Exception as exc:
        return _error(500, "job_creation_failed", str(exc), request_id)
    return {
        "job_id": job["id"], "status": job["status"],
        "status_url": f"/api/v1/transcriptions/{job['id']}",
        "hotword_set_id": job["hotword_set_id"],
        "hotword_set_version": job["hotword_version"],
        "model_version": job["model_version"],
    }


@router.get("/api/v1/transcriptions/{job_id}")
async def get_transcription(job_id: str, request: Request) -> Any:
    request_id = uuid.uuid4().hex
    service = _service(request)
    if service is None:
        return _error(503, "service_unavailable", "文件识别服务未配置", request_id)
    job = service.get_job(job_id)
    if job is None:
        return _error(404, "job_not_found", "转写任务不存在", request_id)
    return job


@router.post("/api/v1/transcriptions/{job_id}/retry", status_code=202)
async def retry_transcription(job_id: str, request: Request) -> Any:
    request_id = uuid.uuid4().hex
    service = _service(request)
    if service is None:
        return _error(503, "service_unavailable", "文件识别服务未配置", request_id)
    job = service.get_job(job_id)
    if job is None:
        return _error(404, "job_not_found", "转写任务不存在", request_id)
    if job["status"] != "failed":
        return _error(409, "job_not_retryable", "只有失败任务可以重试", request_id)
    retry = service.retry_job(job_id)
    return {"job_id": job_id, "status": retry["status"] if retry else "queued"}
