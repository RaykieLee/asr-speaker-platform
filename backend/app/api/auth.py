"""Single administrator API-key checks for local deployments."""

from __future__ import annotations

import hmac
import os
import uuid

from fastapi import Request, WebSocket
from fastapi.responses import JSONResponse


def _configured_key() -> str | None:
    value = os.environ.get("ASR_API_TOKEN", "").strip()
    return value or None


def _valid(candidate: str | None) -> bool:
    expected = _configured_key()
    if expected is None:
        return True
    return bool(candidate) and hmac.compare_digest(candidate, expected)


def _bearer(value: str | None) -> str | None:
    if not value:
        return None
    scheme, separator, token = value.partition(" ")
    if not separator or scheme.lower() != "bearer":
        return None
    return token.strip()


def authorized_websocket(websocket: WebSocket, start_message: dict) -> bool:
    """Browsers may place the key in the first JSON message, not the URL."""
    header_key = _bearer(websocket.headers.get("authorization"))
    message_key = start_message.get("api_key")
    return _valid(header_key) or _valid(message_key if isinstance(message_key, str) else None)


async def api_key_middleware(request: Request, call_next):
    key = _configured_key()
    token = _bearer(request.headers.get("authorization"))
    if request.url.path.startswith("/api/v1/") and not _valid(token):
        return JSONResponse(status_code=401, content={
            "type": "error", "code": "unauthorized", "message": "API Key 无效或未提供",
            "request_id": uuid.uuid4().hex,
        })
    if key is not None and _valid(token):
        request.state.actor = "admin"
        request.state.actor_verified = True
    return await call_next(request)
