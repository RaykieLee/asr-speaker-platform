"""HTTP routes for versioned native sherpa-onnx hotword sets.

The application integrator supplies ``app.state.hotword_service``. This keeps
the API testable and avoids coupling API registration to model loading or DB
startup order.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from backend.app.engines import EngineConfigError, HotwordsUnsupportedError
from backend.app.hotwords.compiler import HotwordCompilationError, HotwordVocabularyUnavailableError
from backend.app.hotwords.service import (
    HotwordNotFoundError,
    HotwordService,
    StaleHotwordVersionError,
)

router = APIRouter(prefix="/api/v1/hotword-sets", tags=["hotwords"])


class HotwordSetCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    entries: list[Any] = Field(min_length=1)
    weight: float = Field(default=1.0, gt=0, le=20)
    tenant_id: str | None = None


class HotwordSetUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_version: int = Field(ge=1)
    entries: list[Any] = Field(min_length=1)
    weight: float | None = Field(default=None, gt=0, le=20)
    name: str | None = Field(default=None, min_length=1, max_length=100)


def get_hotword_service(request: Request) -> HotwordService:
    service = getattr(request.app.state, "hotword_service", None)
    if isinstance(service, HotwordService):
        return service
    # Accept an explicitly installed protocol-compatible service in apps and
    # tests without requiring a concrete class identity check.
    if service is not None and all(hasattr(service, attr) for attr in ("create", "get", "list_sets", "update", "deactivate")):
        return service
    raise HTTPException(
        status_code=503,
        detail={
            "code": "HOTWORD_SERVICE_UNAVAILABLE",
            "message": "热词服务未配置；请注入使用模型词表的服务实例。",
            "request_id": getattr(request.state, "request_id", None) or uuid.uuid4().hex,
        },
    )


def _error(request: Request, status: int, code: str, message: str, **extra: Any) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
    return JSONResponse(
        status_code=status,
        content={"code": code, "message": message, "request_id": str(request_id), **extra},
    )


def _translate_error(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HotwordNotFoundError):
        return _error(request, 404, "HOTWORD_SET_NOT_FOUND", str(exc))
    if isinstance(exc, StaleHotwordVersionError):
        return _error(request, 409, "HOTWORD_VERSION_CONFLICT", str(exc))
    if isinstance(exc, (HotwordsUnsupportedError, EngineConfigError)):
        return _error(request, 422, "HOTWORDS_UNSUPPORTED", str(exc))
    if isinstance(exc, HotwordVocabularyUnavailableError):
        return _error(request, 422, "HOTWORD_VOCAB_UNAVAILABLE", str(exc))
    if isinstance(exc, HotwordCompilationError):
        return _error(request, 422, "HOTWORD_TOKEN_NOT_COVERED", str(exc))
    if isinstance(exc, ValueError):
        return _error(request, 422, "INVALID_HOTWORD_SET", str(exc))
    return _error(request, 500, "HOTWORD_INTERNAL_ERROR", "热词操作失败。")


@router.post("", status_code=201)
def create_hotword_set(body: HotwordSetCreate, request: Request,
                       service: HotwordService = Depends(get_hotword_service)) -> Any:
    try:
        current = service.create(
            name=body.name,
            entries=body.entries,
            weight=body.weight,
            tenant_id=body.tenant_id,
        )
        return current
    except Exception as exc:
        return _translate_error(request, exc)


@router.get("")
def list_hotword_sets(
    request: Request,
    status: Literal["active", "inactive", "all"] = "active",
    tenant_id: str | None = None,
    service: HotwordService = Depends(get_hotword_service),
) -> Any:
    try:
        return service.list_sets(status=None if status == "all" else status, tenant_id=tenant_id)
    except Exception as exc:
        return _translate_error(request, exc)


@router.get("/{set_id}")
def get_hotword_set(
    set_id: str,
    request: Request,
    version: int | None = Query(default=None, ge=1),
    service: HotwordService = Depends(get_hotword_service),
) -> Any:
    try:
        result = service.get(set_id, version=version)
        if result is None:
            return _error(request, 404, "HOTWORD_SET_NOT_FOUND", f"热词集或版本不存在: {set_id}")
        return result
    except Exception as exc:
        return _translate_error(request, exc)


@router.put("/{set_id}")
def update_hotword_set(
    set_id: str,
    body: HotwordSetUpdate,
    request: Request,
    service: HotwordService = Depends(get_hotword_service),
) -> Any:
    try:
        current = service.get(set_id)
        if current is None:
            return _error(request, 404, "HOTWORD_SET_NOT_FOUND", f"热词集不存在: {set_id}")
        if current["status"] != "active":
            return _error(request, 409, "HOTWORD_SET_INACTIVE", "已停用的热词集不能继续编辑。")
        return service.update(
            set_id,
            expected_version=body.expected_version,
            entries=body.entries,
            weight=body.weight if body.weight is not None else float(current["weight"]),
            name=body.name,
        )
    except Exception as exc:
        return _translate_error(request, exc)


@router.delete("/{set_id}")
def deactivate_hotword_set(
    set_id: str,
    request: Request,
    service: HotwordService = Depends(get_hotword_service),
) -> Any:
    try:
        if not service.deactivate(set_id):
            return _error(request, 404, "HOTWORD_SET_NOT_FOUND", f"热词集不存在: {set_id}")
        return {"set_id": set_id, "status": "inactive"}
    except Exception as exc:
        return _translate_error(request, exc)
