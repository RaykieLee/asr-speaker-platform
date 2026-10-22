"""HTTP routes for uploading audio and tracking durable transcription jobs."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal
from unicodedata import category

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.exports import build_export_payload, render_json, render_srt, render_txt
from backend.app.jobs.service import HotwordSnapshotError, JobQueueFull, TranscriptionService, UploadTooLarge

router = APIRouter()


class SpeakerCorrection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registered_speaker_id: str | None = None
    display_name: str | None = Field(default=None, min_length=1, max_length=100)

    @field_validator("display_name")
    @classmethod
    def reject_control_characters(cls, value: str | None) -> str | None:
        if value is not None and any(category(character) in {"Cc", "Zl", "Zp"}
                                     for character in value):
            raise ValueError("display_name 不能包含控制符或换行符")
        return value


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
    if job.get("status") == "succeeded":
        export_payload = build_export_payload(job, service.database)
        # Keep the original segment prediction intact and expose a parallel
        # correction-aware view for clients that render the task result.
        job["speaker_mappings"] = export_payload["speaker_mappings"]
        job["effective_segments"] = export_payload["segments"]
    return job


@router.patch("/api/v1/transcriptions/{job_id}/speakers/{local_id}")
async def correct_transcription_speaker(job_id: str, local_id: str,
                                        correction: SpeakerCorrection,
                                        request: Request) -> Any:
    request_id = uuid.uuid4().hex
    service = _service(request)
    if service is None:
        return _error(503, "service_unavailable", "文件识别服务未配置", request_id)
    if correction.registered_speaker_id is None and correction.display_name is None:
        return _error(422, "invalid_correction", "必须提供 registered_speaker_id 或 display_name", request_id)
    if correction.registered_speaker_id is not None and not correction.registered_speaker_id.strip():
        return _error(422, "invalid_correction", "registered_speaker_id 不能为空", request_id)
    if correction.display_name is not None and not correction.display_name.strip():
        return _error(422, "invalid_correction", "display_name 不能为空", request_id)

    job = service.get_job(job_id)
    if job is None:
        return _error(404, "job_not_found", "转写任务不存在", request_id)
    if job.get("status") != "succeeded":
        return _error(409, "job_not_complete", "只有已完成的转写任务可以修正说话人", request_id)
    metadata = job.get("result_metadata") or {}
    known_ids = {str(segment["local_speaker_id"]) for segment in job.get("segments", [])
                 if segment.get("local_speaker_id")}
    known_ids.update(str(item["local_speaker_id"]) for item in metadata.get("speaker_mappings", [])
                     if item.get("local_speaker_id"))
    if local_id not in known_ids:
        return _error(404, "local_speaker_not_found", "该任务中不存在此本地说话人", request_id)

    database = service.database
    registered_id = correction.registered_speaker_id
    if registered_id and database.get_speaker(registered_id) is None:
        return _error(404, "speaker_not_found", "注册人员不存在", request_id)
    previous = database.get_mapping_override(job_id, local_id)
    audit = list((previous or {}).get("metadata", {}).get("audit_log", []))
    prediction = next((item for item in metadata.get("speaker_mappings", [])
                       if item.get("local_speaker_id") == local_id), {})
    if not audit:
        audit.append({
            "at": (previous or {}).get("created_at") or job.get("finished_at"),
            "action": "model_prediction",
            "registered_speaker_id": prediction.get("registered_speaker_id"),
            "display_name": prediction.get("speaker_name"),
            "match_status": prediction.get("match_status"),
        })
    actor = getattr(request.state, "actor", None)
    actor_verified = bool(getattr(request.state, "actor_verified", False))
    audit.append({
        "at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "action": "manual_correction",
        "operator": actor,
        "actor_verified": actor_verified,
        "registered_speaker_id": registered_id,
        "display_name": correction.display_name.strip() if correction.display_name else None,
    })
    saved = database.set_mapping_override(
        job_id, local_id, registered_speaker_id=registered_id,
        display_name=correction.display_name.strip() if correction.display_name else None,
        operator=actor,
        metadata={"audit_log": audit},
    )
    return {
        "job_id": job_id,
        "local_speaker_id": local_id,
        "original_prediction": prediction,
        "correction": saved,
    }


@router.get("/api/v1/transcriptions/{job_id}/export")
async def export_transcription(job_id: str, request: Request,
                               format: Literal["json", "txt", "srt"] = "json") -> Response:
    request_id = uuid.uuid4().hex
    service = _service(request)
    if service is None:
        return _error(503, "service_unavailable", "文件识别服务未配置", request_id)
    job = service.get_job(job_id)
    if job is None:
        return _error(404, "job_not_found", "转写任务不存在", request_id)
    if job.get("status") != "succeeded":
        return _error(409, "job_not_complete", "只有已完成的转写任务可以导出", request_id)

    payload = build_export_payload(job, service.database)
    renderers = {"json": (render_json, "application/json; charset=utf-8"),
                 "txt": (render_txt, "text/plain; charset=utf-8"),
                 "srt": (render_srt, "application/x-subrip; charset=utf-8")}
    renderer, media_type = renderers[format]
    return Response(
        renderer(payload), media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{job_id}.{format}"'},
    )


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
    try:
        retry = service.retry_job(job_id)
    except JobQueueFull as exc:
        return _error(429, "queue_full", str(exc), request_id)
    return {"job_id": job_id, "status": retry["status"] if retry else "queued"}
