"""CLI entry point for the standalone persistent file-task Worker.

Run with ``python -m backend.app.jobs.process``. The API and worker must share
the same ``ASR_DATA_DIR`` and model files. This process loads only the offline
recognition stack and never binds an HTTP port.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import threading

from .worker import JobWorker, WorkerAlreadyRunning


async def _run(once: bool, poll_interval: float) -> int:
    # These must be set before main is imported because model loading is selected
    # during FastAPI lifespan startup.
    os.environ["ASR_PROCESS_ROLE"] = "worker"
    os.environ["ASR_FILE_WORKER_MODE"] = "external"
    from backend.app.main import app

    async with app.router.lifespan_context(app):
        service = app.state.transcription_service
        if service is None or service.asr is None:
            raise RuntimeError("Worker 未创建文件识别服务，请配置 ASR_OFFLINE_MODEL_DIR")
        worker = JobWorker(app.state.database, app.state.files, service.asr)
        if once:
            try:
                return 0 if worker.run_once() else 1
            finally:
                worker.close()

        stop_event = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop_event.set())
        signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
        try:
            await asyncio.to_thread(worker.run_forever, stop_event=stop_event,
                                    poll_interval=poll_interval)
        finally:
            worker.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ASR persistent file-task Worker")
    parser.add_argument("--once", action="store_true", help="claim at most one queued job then exit")
    parser.add_argument("--poll-interval", type=float, default=0.5,
                        help="idle queue polling interval in seconds (default: 0.5)")
    args = parser.parse_args()
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be positive")
    try:
        return asyncio.run(_run(args.once, args.poll_interval))
    except WorkerAlreadyRunning as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
