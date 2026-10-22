"""HTTP API for registered speaker identities and enrollment samples."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from backend.app.speakers.service import (
    EmbeddingModelVersionError, MissingSpeakerModelError, SampleQualityError,
    SpeakerNotFoundError, SpeakerRegistrationService,
)

router = APIRouter(prefix="/api/v1/speakers", tags=["speakers"])


def _service(request: Request) -> SpeakerRegistrationService:
    service = getattr(request.app.state, "speaker_registration_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="声纹注册服务未配置")
    return service


def _raise(exc: Exception) -> None:
    if isinstance(exc, SpeakerNotFoundError):
        raise HTTPException(status_code=404, detail="人员不存在") from exc
    if isinstance(exc, MissingSpeakerModelError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if isinstance(exc, EmbeddingModelVersionError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, SampleQualityError):
        raise HTTPException(status_code=422, detail={"message": str(exc), "quality": exc.quality}) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise exc


@router.post("")
def create_speaker(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return _service(request).create_speaker(str(payload.get("name", "")))
    except Exception as exc:
        _raise(exc)


@router.get("")
def list_speakers(request: Request) -> list[dict[str, Any]]:
    return _service(request).list_speakers()


@router.get("/{speaker_id}")
def get_speaker(speaker_id: str, request: Request) -> dict[str, Any]:
    try:
        return _service(request).get_speaker(speaker_id)
    except Exception as exc:
        _raise(exc)


@router.patch("/{speaker_id}")
def rename_speaker(speaker_id: str, request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return _service(request).rename_speaker(speaker_id, str(payload.get("name", "")))
    except Exception as exc:
        _raise(exc)


@router.post("/{speaker_id}/samples", status_code=201)
async def add_sample(speaker_id: str, request: Request) -> dict[str, Any]:
    try:
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise ValueError("请使用 multipart/form-data 上传 file 字段")
        return _service(request).add_sample(
            speaker_id, getattr(upload, "filename", None) or "sample.wav", await upload.read()
        )
    except Exception as exc:
        _raise(exc)


@router.delete("/{speaker_id}/samples/{sample_id}")
def delete_sample(speaker_id: str, sample_id: str, request: Request) -> dict[str, bool]:
    service = _service(request)
    if service.database.get_speaker(speaker_id) is None:
        raise HTTPException(status_code=404, detail="人员不存在")
    if not service.delete_sample(speaker_id, sample_id):
        raise HTTPException(status_code=404, detail="样本不存在")
    return {"deleted": True}


@router.delete("/{speaker_id}")
def delete_speaker(speaker_id: str, request: Request) -> dict[str, bool]:
    if not _service(request).delete_speaker(speaker_id):
        raise HTTPException(status_code=404, detail="人员不存在")
    return {"deleted": True}
