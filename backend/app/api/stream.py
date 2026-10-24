"""WebSocket endpoint for the version 1 live ASR protocol."""
from __future__ import annotations

import asyncio
from concurrent.futures import Future
import json
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.app.api.auth import authorized_websocket
from backend.app.audio.core import PCMFormatError
from backend.app.engines.base import EngineConfigError
from backend.app.pipelines.live_asr import LiveASRService, LiveProtocolError, parse_start_message
from backend.app.pipelines.live_final import LiveFinalQueueFull

router = APIRouter()
MAX_PCM_FRAME_BYTES = 256 * 1024


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
        if not isinstance(start_data, dict):
            raise LiveProtocolError("start 消息必须是 JSON 对象")
        if not authorized_websocket(websocket, start_data):
            await _error(websocket, "unauthorized", "API Key 无效或未提供", request_id)
            await websocket.close(code=1008)
            return
        options = parse_start_message(start_data)
        session = await service.create_session(options)
    except EngineConfigError as exc:
        await _error(websocket, "model_unavailable", str(exc), request_id)
        await websocket.close(code=1013)
        return
    except (LiveProtocolError, ValueError, RuntimeError) as exc:
        await _error(websocket, "protocol_error", str(exc), request_id)
        await websocket.close(code=1003)
        return
    except Exception as exc:
        await _error(websocket, "recognition_error", str(exc), request_id)
        await websocket.close(code=1011)
        return

    state = websocket.app.state
    if getattr(state, "stream_lock", None) is None:
        state.stream_lock = asyncio.Lock()
        state.active_streams = 0
        state.max_streams = 16
    async with state.stream_lock:
        if state.active_streams >= state.max_streams:
            await _error(websocket, "capacity_exceeded", "实时连接已达到上限", request_id)
            await websocket.close(code=1013)
            return
        state.active_streams += 1

    try:
        await websocket.send_json({
            "type": "started", "protocol_version": 1, "session_id": session.session_id,
            "hotword_set_id": session.options.hotword_set_id,
            "hotword_set_version": session.options.hotword_set_version,
            "offline_finalization": getattr(state, "live_final_service", None) is not None,
        })
        audio_queue: asyncio.Queue[bytes | None] = service.create_queue()
    except BaseException as exc:
        async with state.stream_lock:
            state.active_streams -= 1
        if isinstance(exc, WebSocketDisconnect):
            return
        raise
    send_lock = asyncio.Lock()
    final_tasks: set[asyncio.Task[Any]] = set()

    async def send_json(data: dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_json(data)

    async def finalize_audio(segment: Any) -> None:
        final_service = getattr(state, "live_final_service", None)
        if final_service is None:
            return
        try:
            future: Future[str] = await asyncio.to_thread(final_service.submit, segment)
        except Exception as exc:
            code = "offline_final_queue_full" if isinstance(exc, LiveFinalQueueFull) else "offline_final_unavailable"
            message = "离线定稿队列已满" if code == "offline_final_queue_full" else "离线定稿暂不可用"
            await send_json({"type": "error", "code": code, "message": message,
                             "request_id": request_id, "session_id": session.session_id,
                             "segment_id": segment.segment_id})
            return
        try:
            text = await asyncio.wrap_future(future)
            await send_json({"type": "segment", "session_id": session.session_id,
                             "segment_id": segment.segment_id, "revision": segment.revision + 1,
                             "status": "final", "start": segment.start, "end": segment.end,
                             "text": text, **segment.speaker})
        except Exception:
            await send_json({"type": "error", "code": "offline_final_failed",
                             "message": "离线定稿失败，保留在线识别结果",
                             "request_id": request_id, "session_id": session.session_id,
                             "segment_id": segment.segment_id})

    def schedule_final_audio() -> None:
        for segment in session.drain_final_audio():
            task = asyncio.create_task(finalize_audio(segment))
            final_tasks.add(task)
            task.add_done_callback(final_tasks.discard)

    async def worker() -> None:
        while True:
            chunk = await audio_queue.get()
            try:
                if chunk is None:
                    events = await asyncio.to_thread(session.finish)
                    for event in events:
                        await send_json(event)
                    schedule_final_audio()
                    if final_tasks:
                        await asyncio.gather(*tuple(final_tasks), return_exceptions=True)
                    await send_json({"type": "completed", "session_id": session.session_id,
                                     "duration": session.duration, "dropped_frames": 0})
                    return
                events = await asyncio.to_thread(session.feed, chunk)
                for event in events:
                    await send_json(event)
                schedule_final_audio()
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
                if len(message["bytes"]) > MAX_PCM_FRAME_BYTES:
                    await _error(websocket, "audio_frame_too_large",
                                 "单个音频帧超过 256 KiB 限制", request_id)
                    await websocket.close(code=1009)
                    stopped = True
                    continue
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
        async with state.stream_lock:
            state.active_streams -= 1
        final_service = getattr(state, "live_final_service", None)
        cancel_session = getattr(final_service, "cancel_session", None)
        if callable(cancel_session):
            cancel_session(session.session_id)
        if receive_task is not None and not receive_task.done():
            receive_task.cancel()
        if not worker_task.done():
            worker_task.cancel()
            try:
                await worker_task
            except (asyncio.CancelledError, Exception):
                pass
        for task in tuple(final_tasks):
            task.cancel()
        if final_tasks:
            await asyncio.gather(*tuple(final_tasks), return_exceptions=True)
