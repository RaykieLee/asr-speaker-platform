"""WebSocket endpoint for the version 1 live ASR protocol."""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.audio.core import PCMFormatError
from app.engines.base import EngineConfigError
from app.pipelines.live_asr import LiveASRService, LiveProtocolError, parse_start_message

router = APIRouter()


def _service(websocket: WebSocket) -> LiveASRService:
    service = getattr(websocket.app.state, "live_asr_service", None)
    if service is None:
        raise RuntimeError("在线识别模型未配置或未安装")
    return service


async def _error(websocket: WebSocket, code: str, message: str, request_id: str) -> None:
    await websocket.send_json({"type": "error", "code": code, "message": message,
                               "request_id": request_id})


@router.websocket("/api/v1/stream")
async def stream_asr(websocket: WebSocket) -> None:
    """Accept start JSON, PCM s16le frames and stop JSON on one socket."""
    request_id = uuid.uuid4().hex
    await websocket.accept()
    try:
        service = _service(websocket)
    except RuntimeError as exc:
        await _error(websocket, "model_unavailable", str(exc), request_id)
        await websocket.close(code=1013)
        return

    try:
        first = await websocket.receive()
        if first["type"] == "websocket.disconnect":
            return
        if first.get("text") is None:
            raise LiveProtocolError("首条消息必须是 JSON start")
        try:
            start_data = json.loads(first["text"])
        except json.JSONDecodeError as exc:
            raise LiveProtocolError("start 消息不是有效 JSON") from exc
        options = parse_start_message(start_data)
        session = service.create_session(options)
    except EngineConfigError as exc:
        await _error(websocket, "model_unavailable", str(exc), request_id)
        await websocket.close(code=1013)
        return
    except (LiveProtocolError, ValueError, RuntimeError) as exc:
        await _error(websocket, "protocol_error", str(exc), request_id)
        await websocket.close(code=1003)
        return

    await websocket.send_json({
        "type": "started", "protocol_version": 1, "session_id": session.session_id,
        "hotword_set_id": options.hotword_set_id,
        "hotword_set_version": options.hotword_set_version,
    })
    audio_queue: asyncio.Queue[bytes | None] = service.create_queue()
    send_lock = asyncio.Lock()

    async def send_json(data: dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_json(data)

    async def worker() -> None:
        while True:
            chunk = await audio_queue.get()
            try:
                if chunk is None:
                    events = await asyncio.to_thread(session.finish)
                    for event in events:
                        await send_json(event)
                    await send_json({"type": "completed", "session_id": session.session_id,
                                     "duration": session.duration})
                    return
                events = await asyncio.to_thread(session.feed, chunk)
                for event in events:
                    await send_json(event)
            finally:
                audio_queue.task_done()

    worker_task = asyncio.create_task(worker())
    stopped = False
    receive_task: asyncio.Task[Any] | None = None
    try:
        while not stopped:
            if worker_task.done():
                await worker_task
            receive_task = asyncio.create_task(websocket.receive())
            done, _ = await asyncio.wait((receive_task, worker_task), return_when=asyncio.FIRST_COMPLETED)
            if worker_task in done:
                receive_task.cancel()
                await worker_task
                break
            message = await receive_task
            receive_task = None
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                try:
                    audio_queue.put_nowait(message["bytes"])
                except asyncio.QueueFull:
                    await _error(websocket, "queue_full", "音频处理队列已满，请降低发送速度", request_id)
                    await websocket.close(code=1013)
                    stopped = True
                continue
            text = message.get("text")
            if text is None:
                await _error(websocket, "protocol_error", "消息必须是 JSON 控制消息或二进制 PCM", request_id)
                await websocket.close(code=1003)
                stopped = True
                continue
            try:
                control = json.loads(text)
            except json.JSONDecodeError:
                control = None
            if isinstance(control, dict) and control.get("type") == "stop":
                await audio_queue.put(None)
                await worker_task
                stopped = True
            else:
                await _error(websocket, "protocol_error", "启动后只允许发送二进制 PCM 或 JSON stop", request_id)
                await websocket.close(code=1003)
                stopped = True
    except WebSocketDisconnect:
        pass
    except (PCMFormatError, LiveProtocolError) as exc:
        await _error(websocket, "invalid_audio", str(exc), request_id)
        await websocket.close(code=1003)
    except Exception as exc:
        await _error(websocket, "recognition_error", str(exc), request_id)
        await websocket.close(code=1011)
    finally:
        if receive_task is not None and not receive_task.done():
            receive_task.cancel()
        if not worker_task.done():
            worker_task.cancel()
            try:
                await worker_task
            except (asyncio.CancelledError, Exception):
                pass
