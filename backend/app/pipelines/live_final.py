"""Isolated, bounded subprocess for real-time offline final recognition."""
from __future__ import annotations

import array
import multiprocessing as mp
import queue
import threading
import uuid
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable

from backend.app.pipelines.live_asr import FinalAudio


class LiveFinalUnavailable(RuntimeError):
    """The isolated offline final recognizer is unavailable."""


class LiveFinalQueueFull(RuntimeError):
    """The bounded offline final queue has no free slot."""


def _child_main(requests: Any, responses: Any, cancellations: Any,
                config: dict[str, Any]) -> None:
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

    cancelled_sessions: set[str] = set()
    while True:
        while True:
            try:
                cancelled_sessions.add(cancellations.get_nowait())
            except queue.Empty:
                break
        item = requests.get()
        if item is None:
            return
        request_id, session_id, pcm, hotwords = item
        if session_id in cancelled_sessions:
            responses.put((request_id, False, "session cancelled"))
            continue
        try:
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
        self._cancellations = context.Queue(maxsize=queue_size + 1)
        self._process = context.Process(target=_worker_target or _child_main,
                                        args=(self._requests, self._responses,
                                              self._cancellations, config),
                                        name="asr-live-final", daemon=True)
        self._futures: dict[str, Future[str]] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._process.start()
        try:
            ready_id, ready, message = self._responses.get(timeout=startup_timeout)
        except queue.Empty as exc:
            self.close()
            raise LiveFinalUnavailable("离线定稿进程启动超时") from exc
        if ready_id != "__ready__" or not ready:
            self.close()
            raise LiveFinalUnavailable(message or "离线定稿进程不可用")
        self._reader = threading.Thread(target=self._read_responses,
                                        name="asr-live-final-responses", daemon=True)
        self._reader.start()

    def submit(self, segment: FinalAudio) -> Future[str]:
        if self._closed or not self._process.is_alive():
            raise LiveFinalUnavailable("离线定稿进程不可用")
        request_id = uuid.uuid4().hex
        future: Future[str] = Future()
        with self._lock:
            self._futures[request_id] = future
            try:
                self._requests.put_nowait((request_id, segment.session_id,
                                           segment.samples.tobytes(), segment.hotwords))
            except queue.Full as exc:
                self._futures.pop(request_id, None)
                raise LiveFinalQueueFull("离线定稿队列已满") from exc
        return future

    @property
    def available(self) -> bool:
        return not self._closed and self._process.is_alive()

    def cancel_session(self, session_id: str) -> None:
        """Best-effort cancel queued work after a client disconnects."""
        if self._closed:
            return
        try:
            self._cancellations.put_nowait(session_id)
        except (queue.Full, OSError, ValueError):
            pass

    def _read_responses(self) -> None:
        while not self._closed:
            try:
                request_id, ok, value = self._responses.get(timeout=0.2)
            except queue.Empty:
                if not self._process.is_alive():
                    break
                continue
            with self._lock:
                future = self._futures.pop(request_id, None)
            if future is None or future.cancelled():
                continue
            if ok:
                future.set_result(str(value))
            else:
                future.set_exception(LiveFinalUnavailable(str(value)))
        with self._lock:
            pending, self._futures = self._futures, {}
        for future in pending.values():
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
            pending, self._futures = self._futures, {}
        for future in pending.values():
            if not future.done():
                future.set_exception(LiveFinalUnavailable("离线定稿服务已关闭"))
        reader = getattr(self, "_reader", None)
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=1)
        for channel in (self._requests, self._responses, self._cancellations):
            try:
                channel.close()
                channel.cancel_join_thread()
            except (OSError, ValueError):
                pass
