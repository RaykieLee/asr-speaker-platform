"""Isolated, bounded subprocess for real-time offline final recognition."""
from __future__ import annotations

import array
import hashlib
import multiprocessing as mp
import queue
import threading
import time
import uuid
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable

from backend.app.pipelines.live_asr import FinalAudio


class LiveFinalUnavailable(RuntimeError):
    """The isolated offline final recognizer is unavailable."""


class LiveFinalQueueFull(RuntimeError):
    """The bounded offline final queue has no free slot."""


def _session_key(session_id: str) -> bytes:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32].encode("ascii")


def _is_session_cancelled(shared_keys: Any, shared_flags: Any, shared_lock: Any,
                          session_id: str) -> bool:
    key = _session_key(session_id)
    with shared_lock:
        for slot in range(len(shared_flags)):
            start = slot * 32
            if bytes(shared_keys[start:start + 32]) == key:
                return bool(shared_flags[slot])
    return False


def _child_main(requests: Any, responses: Any, shared_keys: Any,
                shared_flags: Any, shared_lock: Any, config: dict[str, Any]) -> None:
    try:
        from backend.app.engines.base import ModelArtifact
        from backend.app.engines.sherpa_offline import OfflineTransducerFiles, SherpaOfflineRecognizer

        files = OfflineTransducerFiles(**{
            key: Path(config[key]) for key in ("encoder", "decoder", "joiner", "tokens")
        })
        recognizer = SherpaOfflineRecognizer.load(
            ModelArtifact("live-final-offline", "transducer", config["version"], tuple(files.__dict__.values())),
            files, num_threads=config["num_threads"], modeling_unit="cjkchar+bpe",
            bpe_vocab=config["bpe_vocab"],
        )
        responses.put(("__ready__", True, ""))
    except BaseException:
        # Startup errors are intentionally generic; model paths and exception details
        # are not sent across the process boundary or to WebSocket clients.
        responses.put(("__ready__", False, "offline recognizer initialization failed"))
        return

    heartbeat_stop = threading.Event()

    def heartbeat() -> None:
        while not heartbeat_stop.wait(1.0):
            try:
                responses.put_nowait(("__heartbeat__", True, time.monotonic()))
            except queue.Full:
                continue

    heartbeat_thread = threading.Thread(target=heartbeat, name="live-final-heartbeat", daemon=True)
    heartbeat_thread.start()

    while True:
        item = requests.get()
        if item is None:
            heartbeat_stop.set()
            return
        request_id, session_id, pcm, hotwords = item
        if _is_session_cancelled(shared_keys, shared_flags, shared_lock, session_id):
            responses.put((request_id, False, "session cancelled"))
            continue
        try:
            if isinstance(pcm, array.array):
                samples = pcm
            else:
                samples = array.array("f")
                samples.frombytes(pcm)
            text = recognizer.transcribe(samples, 16_000, list(hotwords) or None)
            responses.put((request_id, True, text))
        except BaseException:
            responses.put((request_id, False, "offline recognition failed"))


class LiveFinalProcess:
    """One dedicated child process with a bounded request queue.

    ``submit`` is non-blocking. The response reader resolves concurrent futures;
    callers can await them without tying up the WebSocket event loop.
    """

    def __init__(self, config: dict[str, Any], *, queue_size: int = 16,
                 startup_timeout: float = 60.0,
                 _worker_target: Callable[..., None] | None = None):
        if queue_size < 1 or queue_size > 64:
            raise ValueError("queue_size must be between 1 and 64")
        context = mp.get_context("spawn")
        self._requests = context.Queue(maxsize=queue_size)
        self._responses = context.Queue(maxsize=queue_size + 1)
        self._slot_count = queue_size + 1
        self._shared_keys = context.Array("B", self._slot_count * 32, lock=False)
        self._shared_flags = context.Array("b", self._slot_count, lock=False)
        self._shared_lock = context.Lock()
        self._process = context.Process(target=_worker_target or _child_main,
                                        args=(self._requests, self._responses, self._shared_keys,
                                              self._shared_flags, self._shared_lock, config),
                                        name="asr-live-final", daemon=True)
        self._futures: dict[str, Future[str]] = {}
        self._request_sessions: dict[str, str] = {}
        self._pending_by_session: dict[str, int] = {}
        self._session_slots: dict[str, int] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._last_heartbeat = 0.0
        self._process.start()
        try:
            ready_id, ready, message = self._responses.get(timeout=startup_timeout)
        except queue.Empty as exc:
            self.close()
            raise LiveFinalUnavailable("离线定稿进程启动超时") from exc
        if ready_id != "__ready__" or not ready:
            self.close()
            raise LiveFinalUnavailable(message or "离线定稿进程不可用")
        self._last_heartbeat = time.monotonic()
        self._reader = threading.Thread(target=self._read_responses,
                                        name="asr-live-final-responses", daemon=True)
        self._reader.start()

    def submit(self, segment: FinalAudio) -> Future[str]:
        if self._closed or not self._process.is_alive():
            raise LiveFinalUnavailable("离线定稿进程不可用")
        request_id = uuid.uuid4().hex
        future: Future[str] = Future()
        with self._lock:
            slot = self._reserve_session_slot(segment.session_id)
            self._futures[request_id] = future
            self._request_sessions[request_id] = segment.session_id
            self._pending_by_session[segment.session_id] = self._pending_by_session.get(segment.session_id, 0) + 1
            try:
                # Queue's feeder thread serializes the immutable sample array;
                # submit stays non-blocking on the WebSocket event loop.
                self._requests.put_nowait((request_id, segment.session_id,
                                           segment.samples, segment.hotwords))
            except (queue.Full, OSError, ValueError) as exc:
                self._futures.pop(request_id, None)
                self._request_sessions.pop(request_id, None)
                remaining = self._pending_by_session[segment.session_id] - 1
                if remaining:
                    self._pending_by_session[segment.session_id] = remaining
                else:
                    self._pending_by_session.pop(segment.session_id, None)
                    self._clear_session_slot(segment.session_id, slot)
                if isinstance(exc, queue.Full):
                    raise LiveFinalQueueFull("离线定稿队列已满") from exc
                raise LiveFinalUnavailable("离线定稿进程不可用") from exc
        return future

    def _reserve_session_slot(self, session_id: str) -> int:
        slot = self._session_slots.get(session_id)
        if slot is not None:
            with self._shared_lock:
                if self._shared_flags[slot]:
                    raise LiveFinalUnavailable("会话已取消，不能提交新的离线定稿")
            return slot
        key = _session_key(session_id)
        with self._shared_lock:
            for candidate in range(self._slot_count):
                start = candidate * 32
                current = bytes(self._shared_keys[start:start + 32])
                if current == key:
                    slot = candidate
                    break
                if not any(current) and slot is None:
                    slot = candidate
            if slot is None:
                raise LiveFinalQueueFull("实时会话取消状态槽已满")
            start = slot * 32
            self._shared_keys[start:start + 32] = key
            self._shared_flags[slot] = 0
        self._session_slots[session_id] = slot
        return slot

    def _clear_session_slot(self, session_id: str, slot: int) -> None:
        with self._shared_lock:
            start = slot * 32
            self._shared_keys[start:start + 32] = b"\x00" * 32
            self._shared_flags[slot] = 0
        self._session_slots.pop(session_id, None)

    def _complete_request(self, request_id: str) -> tuple[Future[str] | None, bool]:
        with self._lock:
            future = self._futures.pop(request_id, None)
            session_id = self._request_sessions.pop(request_id, None)
            if session_id is None:
                return future, False
            remaining = self._pending_by_session.get(session_id, 0) - 1
            if remaining > 0:
                self._pending_by_session[session_id] = remaining
                return future, False
            self._pending_by_session.pop(session_id, None)
            slot = self._session_slots.pop(session_id, None)
            if slot is not None:
                with self._shared_lock:
                    start = slot * 32
                    self._shared_keys[start:start + 32] = b"\x00" * 32
                    self._shared_flags[slot] = 0
            return future, True

    @property
    def available(self) -> bool:
        return (not self._closed and self._process.is_alive() and
                time.monotonic() - self._last_heartbeat < 5.0)

    def cancel_session(self, session_id: str) -> None:
        """Mark this session's outstanding requests for child-side skipping."""
        if self._closed:
            return
        with self._lock:
            slot = self._session_slots.get(session_id)
            if slot is not None:
                with self._shared_lock:
                    self._shared_flags[slot] = 1

    def _read_responses(self) -> None:
        while not self._closed:
            try:
                request_id, ok, value = self._responses.get(timeout=0.2)
            except queue.Empty:
                if not self._process.is_alive():
                    break
                continue
            if request_id == "__heartbeat__":
                self._last_heartbeat = float(value)
                continue
            future, _last_for_session = self._complete_request(request_id)
            if future is None or future.cancelled():
                continue
            if ok:
                future.set_result(str(value))
            else:
                future.set_exception(LiveFinalUnavailable(str(value)))
        with self._lock:
            pending = list(self._futures.items())
            self._futures.clear()
            self._request_sessions.clear()
            self._pending_by_session.clear()
            self._session_slots.clear()
            with self._shared_lock:
                for slot in range(self._slot_count):
                    start = slot * 32
                    self._shared_keys[start:start + 32] = b"\x00" * 32
                    self._shared_flags[slot] = 0
        for _request_id, future in pending:
            if not future.done():
                future.set_exception(LiveFinalUnavailable("离线定稿进程已退出"))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._requests.put_nowait(None)
        except (queue.Full, OSError, ValueError):
            pass
        self._process.join(timeout=3)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2)
        with self._lock:
            pending = list(self._futures.items())
            self._futures.clear()
            self._request_sessions.clear()
            self._pending_by_session.clear()
            self._session_slots.clear()
        for _request_id, future in pending:
            if not future.done():
                future.set_exception(LiveFinalUnavailable("离线定稿服务已关闭"))
        reader = getattr(self, "_reader", None)
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=1)
        for channel in (self._requests, self._responses):
            try:
                channel.close()
                channel.cancel_join_thread()
            except (OSError, ValueError):
                pass
