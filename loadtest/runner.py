"""CLI and async load runner for the draft ASR HTTP/WebSocket API."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import math
import tempfile
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .core import (ResourceSampler, SAMPLE_RATE, SAMPLE_WIDTH, SessionResult,
                   _revision_replaces, _valid_revision, aggregate_final_speaker_events,
                   hardware_metadata, pace_pcm, sha256_file, write_report)


def load_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != SAMPLE_RATE or wav.getnchannels() != 1 or wav.getsampwidth() != SAMPLE_WIDTH:
            raise ValueError("WAV must be mono, 16 kHz, signed 16-bit PCM (s16le)")
        if wav.getcomptype() != "NONE":
            raise ValueError("WAV must be uncompressed PCM")
        pcm = wav.readframes(wav.getnframes())
    if not pcm:
        raise ValueError("WAV contains no PCM samples")
    return pcm


def write_repeated_wav(path: Path, pcm: bytes, repeats: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        for _ in range(repeats):
            wav.writeframesraw(pcm)
        wav.writeframes(b"")


def _server_elapsed(job: dict[str, Any], start_key: str, end_key: str) -> float | None:
    """Use durable server timestamps when polling misses a short job state."""
    start, end = job.get(start_key), job.get(end_key)
    if not start or not end:
        return None
    try:
        first = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        last = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
        elapsed = (last - first).total_seconds()
    except (TypeError, ValueError):
        return None
    return elapsed if elapsed >= 0 else None


def ws_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{parsed.netloc}/api/v1/stream"


def _event_type(event: dict[str, Any]) -> str:
    return str(event.get("type", event.get("status", ""))).lower()


def _is_type(event: dict[str, Any], *names: str) -> bool:
    kind = _event_type(event)
    status = str(event.get("status", "")).lower()
    return kind in names or status in names


def _remember_server_error(result: SessionResult, event: dict[str, Any]) -> None:
    nested = event.get("error") if isinstance(event.get("error"), dict) else {}
    code = event.get("code", event.get("error_code", nested.get("code")))
    message = event.get("message", event.get("detail", nested.get("message", "")))
    if code is not None:
        result.server_error_code = str(code)
    if message is not None:
        result.server_error_message = str(message)


def _remember_websocket_close(result: SessionResult, error: BaseException) -> None:
    """Extract close diagnostics across websockets versions and fake clients."""
    code = getattr(error, "code", None)
    reason = getattr(error, "reason", None)
    received = getattr(error, "rcvd", None)
    if received is not None:
        code = getattr(received, "code", code)
        reason = getattr(received, "reason", reason)
    try:
        if code is not None:
            result.websocket_close_code = int(code)
    except (TypeError, ValueError):
        pass
    if reason:
        result.websocket_close_reason = str(reason)


def _server_error_text(result: SessionResult) -> str:
    code = result.server_error_code or "unknown"
    return f"server_error:{code}"


async def _bounded_reader_drain(task: asyncio.Task[Any], *, timeout_seconds: float = 0.05) -> None:
    """Let already queued diagnostics reach the reader without stalling a run."""
    if task.done():
        return
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout_seconds)
    except TimeoutError:
        pass


async def run_live_session(
    *,
    pcm: bytes,
    endpoint: str,
    run_id: str,
    frame_ms: int,
    hotword_set_id: str | None,
    speaker_scope: str | None,
    token: str | None,
    timeout_seconds: float,
    concurrency_tier: int | None = None,
    repeat_count: int = 1,
    max_audio_seconds: float | None = None,
    start_delay_seconds: float = 0.0,
) -> SessionResult:
    try:
        import websockets  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("websockets is required; install loadtest/requirements.txt") from exc
    if start_delay_seconds > 0:
        await asyncio.sleep(start_delay_seconds)
    result = SessionResult(
        run_id=run_id,
        mode="realtime",
        started_at=datetime.now(timezone.utc).isoformat(),
        start_monotonic=time.monotonic(),
        audio_seconds=min(
            len(pcm) * repeat_count / (SAMPLE_RATE * SAMPLE_WIDTH),
            max_audio_seconds if max_audio_seconds is not None else float("inf"),
        ),
        concurrency_tier=concurrency_tier,
    )
    headers = {"Authorization": f"Bearer {token}"} if token else None
    try:
        header_arg = "additional_headers" if "additional_headers" in inspect.signature(websockets.connect).parameters else "extra_headers"
        ws_options: dict[str, Any] = {"open_timeout": timeout_seconds, "max_size": None}
        if headers:
            ws_options[header_arg] = headers
        async with websockets.connect(endpoint, **ws_options) as socket:
            start: dict[str, Any] = {
                "type": "start",
                "protocol_version": 1,
                "sample_rate": SAMPLE_RATE,
                "encoding": "pcm_s16le",
            }
            if hotword_set_id:
                start["hotword_set_id"] = hotword_set_id
            if speaker_scope:
                start["speaker_scope"] = speaker_scope
            await socket.send(json.dumps(start))
            # The draft protocol says start receives a confirmation. Accommodate
            # a first useful event before the acknowledgement in implementations.
            first_event: dict[str, Any] | None = None
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout_seconds
            while loop.time() < deadline:
                message = await asyncio.wait_for(socket.recv(), timeout=max(0.01, deadline - loop.time()))
                if isinstance(message, bytes):
                    continue
                event = json.loads(message)
                if not isinstance(event, dict):
                    continue
                if _is_type(event, "ready", "started", "session_started", "start_ack"):
                    result.session_id = str(event.get("session_id", ""))
                    break
                if _is_type(event, "error", "server_error"):
                    _remember_server_error(result, event)
                    raise RuntimeError(_server_error_text(result))
                first_event = event
                break
            else:
                raise TimeoutError("start acknowledgement timed out")

            final_segments: dict[str, tuple[int | None, int, float, str]] = {}
            speaker_final_events: list[dict[str, Any]] = []
            final_event_sequence = 0
            stop_sent_at: float | None = None

            def record(event: dict[str, Any]) -> None:
                nonlocal final_event_sequence
                if _is_type(event, "error", "server_error"):
                    _remember_server_error(result, event)
                    return
                received = time.monotonic() - result.start_monotonic
                if not result.session_id and event.get("session_id"):
                    result.session_id = str(event["session_id"])
                if _is_type(event, "partial") or str(event.get("status", "")).lower() == "partial":
                    if result.first_partial_seconds is None:
                        result.first_partial_seconds = received
                if _is_type(event, "segment") and str(event.get("status", "")).lower() == "partial":
                    if result.first_partial_seconds is None:
                        result.first_partial_seconds = received
                if _is_type(event, "segment") and str(event.get("status", "")).lower() in {"final", "corrected"}:
                    text = str(event.get("text", ""))
                    segment_id = str(event.get("segment_id", f"anonymous-{len(final_segments)}"))
                    sequence = final_event_sequence
                    final_event_sequence += 1
                    try:
                        segment_start = float(event.get("start", len(final_segments)))
                    except (TypeError, ValueError):
                        segment_start = float(len(final_segments))
                    revision = _valid_revision(event.get("revision"))
                    previous = final_segments.get(segment_id)
                    if previous is None or _revision_replaces(revision, previous[0], sequence, previous[1]):
                        final_segments[segment_id] = (revision, sequence, segment_start, text)
                    result.final_text = "".join(
                        part[3] for part in sorted(final_segments.values(), key=lambda value: value[2])
                    )
                    if result.first_final_segment_seconds is None:
                        result.first_final_segment_seconds = received
                    if result.final_text_seconds is None:
                        result.final_text_seconds = received
                    speaker = event.get("speaker_name") or event.get("registered_speaker_id") or event.get("local_speaker_id")
                    if speaker:
                        result.speaker = str(speaker)
                        if result.speaker_seconds is None:
                            result.speaker_seconds = received
                    # Store only timing, revision, and temporary speaker identifiers for
                    # aggregation. Never add segment text to this side-channel.
                    speaker_final_events.append({
                        "type": "segment", "status": str(event.get("status", "")).lower(),
                        "segment_id": event.get("segment_id"), "revision": event.get("revision"),
                        "start": event.get("start"), "end": event.get("end"),
                        "local_speaker_id": event.get("local_speaker_id"),
                        "registered_speaker_id": event.get("registered_speaker_id"),
                        "speaker_name": event.get("speaker_name"),
                        "_received_monotonic": time.monotonic(),
                    })
                if _is_type(event, "final") and str(event.get("type", "")).lower() != "segment":
                    text = str(event.get("text", ""))
                    result.final_text = (result.final_text + text).strip() or text
                    if result.first_final_segment_seconds is None:
                        result.first_final_segment_seconds = received
                    if result.final_text_seconds is None:
                        result.final_text_seconds = received
                    speaker = event.get("speaker_name") or event.get("registered_speaker_id") or event.get("local_speaker_id")
                    if speaker:
                        result.speaker = str(speaker)
                        if result.speaker_seconds is None:
                            result.speaker_seconds = received
                if "dropped_frames" in event:
                    result.server_dropped_frames = int(event["dropped_frames"])
                if _is_type(event, "completed"):
                    result.completed = True
                    if stop_sent_at is not None:
                        result.completion_after_stop_seconds = max(0.0, time.monotonic() - stop_sent_at)

            reader_error: list[BaseException] = []

            async def reader() -> None:
                try:
                    if first_event is not None:
                        record(first_event)
                    while True:
                        message = await socket.recv()
                        if isinstance(message, bytes):
                            continue
                        event = json.loads(message)
                        if isinstance(event, dict):
                            record(event)
                            if result.completed:
                                return
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    reader_error.append(exc)

            reader_task = asyncio.create_task(reader())
            audio_started_at = time.monotonic()
            try:
                async def send_audio(frame: bytes) -> None:
                    if result.server_error_code or reader_error:
                        # Give the reader a chance to consume the close frame that
                        # often follows a structured server error (for example 1013).
                        await asyncio.sleep(0.02)
                        if reader_error:
                            raise reader_error[0]
                        raise RuntimeError(_server_error_text(result))
                    try:
                        await socket.send(frame)
                    except Exception:
                        # A close can race the reader: send() may fail in the same
                        # turn that the server error was queued. Drain briefly before
                        # finally cancels the reader so structured diagnostics win.
                        await _bounded_reader_drain(reader_task)
                        raise

                def progress(frames_sent: int, backlog: float, late_frames: int) -> None:
                    result.frames_sent = frames_sent
                    result.max_client_backlog_seconds = backlog
                    result.client_late_frames = late_frames

                paced = await pace_pcm(
                    pcm, send_audio, frame_ms=frame_ms, on_progress=progress,
                    repeat_count=repeat_count, max_audio_seconds=max_audio_seconds,
                )
                result.frames_sent = int(paced["frames_sent"])
                result.max_client_backlog_seconds = float(paced["max_client_backlog_seconds"])
                result.client_late_frames = int(paced["client_late_frames"])
                if result.server_error_code:
                    raise RuntimeError(_server_error_text(result))
                if reader_error:
                    raise reader_error[0]
                stop_sent_at = time.monotonic()
                await socket.send(json.dumps({"type": "stop"}))
                try:
                    await asyncio.wait_for(reader_task, timeout=timeout_seconds)
                except TimeoutError:
                    raise TimeoutError("stream completion timed out")
                if reader_error:
                    raise reader_error[0]
            finally:
                if not reader_task.done():
                    reader_task.cancel()
                    await asyncio.gather(reader_task, return_exceptions=True)
            if result.completed:
                speaker_metrics = aggregate_final_speaker_events(
                    speaker_final_events, audio_started_at=audio_started_at
                )
                for key, value in speaker_metrics.items():
                    setattr(result, key, value)
    except Exception as exc:
        _remember_websocket_close(result, exc)
        if result.server_error_code:
            result.error = _server_error_text(result)
        else:
            result.error = f"{type(exc).__name__}: {exc}"
    return result


async def run_file_job(
    *,
    path: Path,
    audio_seconds: float,
    base_url: str,
    run_id: str,
    hotword_set_id: str | None,
    token: str | None,
    poll_seconds: float,
    timeout_seconds: float,
) -> SessionResult:
    try:
        import httpx  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("httpx is required; install loadtest/requirements.txt") from exc
    started = time.monotonic()
    result = SessionResult(
        run_id=run_id,
        mode="file",
        started_at=datetime.now(timezone.utc).isoformat(),
        start_monotonic=started,
        audio_seconds=audio_seconds,
    )
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        # A benchmark must reach the requested server directly. In particular,
        # macOS system proxies can otherwise intercept localhost uploads.
        async with httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers,
                                     timeout=timeout_seconds, trust_env=False) as client:
            data: dict[str, str] = {"language": "auto", "channel_mode": "mixdown"}
            if hotword_set_id:
                data["hotword_set_id"] = hotword_set_id
            with path.open("rb") as source:
                response = await client.post(
                    "/api/v1/transcriptions",
                    data=data,
                    files={"file": (path.name, source, "audio/wav")},
                )
            response.raise_for_status()
            body = response.json()
            result.job_id = str(body.get("job_id", body.get("id", "")))
            if not result.job_id:
                raise ValueError("create job response did not contain job_id")
            queued_at = time.monotonic()
            running_at: float | None = None
            deadline = queued_at + timeout_seconds
            while True:
                status_response = await client.get(f"/api/v1/transcriptions/{result.job_id}")
                status_response.raise_for_status()
                job = status_response.json()
                state = str(job.get("status", "")).lower()
                if state == "running" and running_at is None:
                    running_at = time.monotonic()
                    queue_value = job.get("queue_seconds")
                    result.queue_seconds = float(queue_value) if queue_value is not None else running_at - queued_at
                elif result.queue_seconds is None and job.get("queue_seconds") is not None:
                    result.queue_seconds = float(job["queue_seconds"])
                if state in {"succeeded", "completed", "complete"}:
                    ended = time.monotonic()
                    result.completed = True
                    server_queue = _server_elapsed(job, "created_at", "started_at")
                    if server_queue is not None:
                        result.queue_seconds = server_queue
                    if job.get("processing_seconds") is not None:
                        result.processing_seconds = float(job["processing_seconds"])
                    elif (server_processing := _server_elapsed(job, "started_at", "finished_at")) is not None:
                        result.processing_seconds = server_processing
                    elif running_at is not None:
                        result.processing_seconds = ended - running_at
                    result.rtf = (ended - started) / result.audio_seconds if result.audio_seconds else None
                    segments = job.get("effective_segments") or job.get("segments", [])
                    result.final_text = "".join(str(segment.get("text", "")) for segment in segments)
                    speakers = [(segment.get("speaker") or {}).get("display_name")
                                or segment.get("speaker_name")
                                or (segment.get("metadata") or {}).get("speaker_name")
                                or segment.get("registered_speaker_id")
                                or segment.get("local_speaker_id") for segment in segments]
                    result.speaker = ", ".join(dict.fromkeys(str(value) for value in speakers if value))
                    break
                if state in {"failed", "error", "cancelled", "canceled"}:
                    result.error = f"job_{state}: {job.get('error', job.get('error_code', 'unknown'))}"
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("file job polling timed out")
                await asyncio.sleep(poll_seconds)
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def _csv_ints(value: str) -> list[int]:
    try:
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from exc
    if not parsed or any(number <= 0 for number in parsed):
        raise argparse.ArgumentTypeError("concurrency levels must be positive integers")
    return parsed


async def _run_continuous_file_tier(
    *,
    slots: int,
    duration_seconds: float,
    max_jobs: int,
    deadline: float,
    path: Path,
    audio_seconds: float,
    base_url: str,
    run_tag: str,
    hotword_set_id: str | None,
    token: str | None,
    poll_seconds: float,
    timeout_seconds: float,
    concurrency_tier: int,
    job_runner=None,
    clock=time.monotonic,
) -> list[SessionResult]:
    """Keep a bounded number of file jobs active until the tier window closes.

    A slot submits its replacement only after its previous task reaches a final
    state. The window limits submission time; ``max_jobs`` and the number of
    slots also bound result growth and outstanding service work.
    """
    if slots < 1 or max_jobs < 1 or duration_seconds <= 0:
        raise ValueError("continuous file load requires positive slots, duration, and job cap")
    runner = job_runner or run_file_job
    lock = asyncio.Lock()
    next_index = 0
    results: list[SessionResult] = []

    async def worker(_slot_index: int) -> None:
        nonlocal next_index
        while True:
            async with lock:
                if clock() >= deadline or next_index >= max_jobs:
                    return
                next_index += 1
                task_index = next_index
            result = await runner(
                path=path, audio_seconds=audio_seconds, base_url=base_url,
                run_id=f"{run_tag}-file-{task_index}", hotword_set_id=hotword_set_id,
                token=token, poll_seconds=poll_seconds, timeout_seconds=timeout_seconds,
            )
            result.concurrency_tier = concurrency_tier
            result.audio_seconds = audio_seconds
            results.append(result)

    await asyncio.gather(*(worker(index) for index in range(slots)))
    return results


async def run(args: argparse.Namespace) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    source_pcm = load_wav(args.wav)
    source_audio_seconds = len(source_pcm) / (SAMPLE_RATE * SAMPLE_WIDTH)
    pcm = source_pcm * args.repeats if args.mode in {"realtime", "mixed"} else source_pcm
    audio_seconds = source_audio_seconds * args.repeats
    run_started = time.monotonic()
    resource_targets = list(args.resource_process or [])
    if args.server_pid is not None:
        resource_targets.insert(0, ("api", args.server_pid))
    sampler = ResourceSampler(args.resource_interval, args.server_pid, resource_targets)
    results: list[Any] = []
    temp_dir = tempfile.TemporaryDirectory(prefix="asr-loadtest-")
    upload_path = args.wav
    if args.repeats > 1 and args.mode in {"file", "mixed"}:
        upload_path = Path(temp_dir.name) / "repeated-input.wav"
        write_repeated_wav(upload_path, source_pcm, args.repeats)
    await sampler.start()
    file_jobs_by_tier: dict[str, int] = {}
    try:
        for level in args.concurrency:
            sampler.current_concurrency_tier = level
            run_tag = f"c{level}-{uuid.uuid4().hex[:8]}"
            file_slots = level if args.mode == "file" else (
                args.file_concurrency if args.mode == "mixed" else 0
            )
            if args.tier_duration_seconds is None:
                # Preserve the original one-batch behavior unless continuous
                # tiers are explicitly requested.
                tasks = []
                if args.mode in {"realtime", "mixed"}:
                    for index in range(level):
                        tasks.append(run_live_session(
                            pcm=pcm,
                            endpoint=args.ws_url or ws_url(args.base_url),
                            run_id=f"{run_tag}-live-{index + 1}",
                            frame_ms=args.frame_ms,
                            hotword_set_id=args.hotword_set_id,
                            speaker_scope=args.speaker_scope,
                            token=args.token,
                            timeout_seconds=args.timeout,
                            concurrency_tier=level,
                            start_delay_seconds=index * args.start_stagger_ms / 1000,
                        ))
                for index in range(file_slots):
                    tasks.append(run_file_job(
                        path=upload_path,
                        audio_seconds=audio_seconds,
                        base_url=args.base_url,
                        run_id=f"{run_tag}-file-{index + 1}",
                        hotword_set_id=args.hotword_set_id,
                        token=args.token,
                        poll_seconds=args.poll_interval,
                        timeout_seconds=args.timeout,
                    ))
                batch = await asyncio.gather(*tasks)
                sampler.current_concurrency_tier = None
                for item in batch:
                    if item.mode == "file":
                        item.audio_seconds = audio_seconds
                        file_jobs_by_tier[str(level)] = file_jobs_by_tier.get(str(level), 0) + 1
                    results.append(item)
                continue

            tier_duration = args.tier_duration_seconds
            tier_started = time.monotonic()
            deadline = tier_started + tier_duration
            tasks = []
            if args.mode in {"realtime", "mixed"}:
                for index in range(level):
                    start_delay = index * args.start_stagger_ms / 1000
                    if start_delay >= tier_duration:
                        raise ValueError(
                            "start stagger places a realtime session beyond its tier duration"
                        )
                    session_duration = tier_duration - start_delay
                    session_repeats = max(1, math.ceil(session_duration / audio_seconds))
                    tasks.append(asyncio.create_task(run_live_session(
                        pcm=pcm,
                        endpoint=args.ws_url or ws_url(args.base_url),
                        run_id=f"{run_tag}-live-{index + 1}",
                        frame_ms=args.frame_ms,
                        hotword_set_id=args.hotword_set_id,
                        speaker_scope=args.speaker_scope,
                        token=args.token,
                        timeout_seconds=args.timeout,
                        concurrency_tier=level,
                        repeat_count=session_repeats,
                        max_audio_seconds=session_duration,
                        start_delay_seconds=start_delay,
                    )))
            if file_slots:
                file_task = asyncio.create_task(_run_continuous_file_tier(
                    slots=file_slots,
                    duration_seconds=tier_duration,
                    max_jobs=args.max_file_jobs_per_tier,
                    deadline=deadline,
                    path=upload_path,
                    audio_seconds=audio_seconds,
                    base_url=args.base_url,
                    run_tag=run_tag,
                    hotword_set_id=args.hotword_set_id,
                    token=args.token,
                    poll_seconds=args.poll_interval,
                    timeout_seconds=args.timeout,
                    concurrency_tier=level,
                ))
                tasks.append(file_task)
            batch = await asyncio.gather(*tasks)
            sampler.current_concurrency_tier = None
            for item in batch:
                if isinstance(item, list):
                    results.extend(item)
                    file_jobs_by_tier[str(level)] = len(item)
                else:
                    results.append(item)
    finally:
        await sampler.stop()
        temp_dir.cleanup()
    config = {
        "base_url": args.base_url,
        "mode": args.mode,
        "concurrency_levels": args.concurrency,
        "frame_ms": args.frame_ms,
        "hotword_set_id": args.hotword_set_id,
        "speaker_scope": args.speaker_scope,
        "wav": {"name": args.wav.name, "sha256": sha256_file(args.wav)},
        "repeats": args.repeats,
        "tier_duration_seconds": args.tier_duration_seconds,
        "max_file_jobs_per_tier": args.max_file_jobs_per_tier,
        "file_jobs_by_concurrency_tier": file_jobs_by_tier,
        "start_stagger_ms": args.start_stagger_ms,
        "source_audio_seconds": source_audio_seconds,
        "audio_seconds": audio_seconds,
        "wall_elapsed_seconds": time.monotonic() - run_started,
        "server_pid": args.server_pid,
        "resource_targets": [{"role": role, "pid": pid} for role, pid in resource_targets],
        "resource_scope": "unique_descendants_of_configured_process_roots" if resource_targets else "loadtest_client_process",
        "thresholds": {
            "max_error_rate": args.max_error_rate,
            "max_first_partial_p95_seconds": args.max_first_partial_p95,
            "max_completion_after_stop_p95_seconds": args.max_completion_after_stop_p95,
            # Preserve the legacy report key; its meaning has always been
            # stop-to-completed latency, despite the ambiguous name.
            "max_final_p95_seconds": args.max_completion_after_stop_p95,
            "max_client_backlog_seconds": args.max_backlog,
            "max_server_dropped_frames": args.max_server_drops,
        },
        "thresholds_are_user_defined_and_not_capacity_evidence": True,
        "target_environment": {
            "os": args.target_os,
            "cpu_model": args.target_cpu_model,
            "cpu_cores": args.target_cpu_cores,
            "memory_bytes": args.target_memory_bytes,
            "gpu_model": args.target_gpu_model,
            "gpu_memory_bytes": args.target_gpu_memory_bytes,
            "model_version": args.model_version,
            "inference_threads": args.inference_threads,
            "worker_count": args.worker_count,
            "hotword_count": args.hotword_count,
            "audio_notes": args.audio_notes,
        },
    }
    return results, sampler.samples, config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repeatable ASR WebSocket and transcription-job load test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="ASR API base URL")
    parser.add_argument("--ws-url", help="override WebSocket URL; default derives /api/v1/stream from base URL")
    parser.add_argument("--wav", type=Path, required=True, help="mono 16 kHz s16le WAV input")
    parser.add_argument("--mode", choices=("realtime", "file", "mixed"), default="realtime")
    parser.add_argument("--concurrency", type=_csv_ints, default=[1, 2, 4], help="comma-separated staircase levels")
    parser.add_argument("--frame-ms", type=int, default=40, choices=(20, 40, 60, 80, 100))
    parser.add_argument("--repeats", type=int, default=1, help="repeat the authorized input clip to sustain longer runs")
    parser.add_argument("--file-concurrency", type=int, default=1, help="file jobs alongside each mixed-load level")
    parser.add_argument(
        "--tier-duration-seconds", type=float,
        help="run each file/mixed/realtime concurrency tier for this duration; continuously resubmit file jobs",
    )
    parser.add_argument(
        "--max-file-jobs-per-tier", type=int, default=1000,
        help="hard cap on file submissions in a continuous tier (prevents unbounded result growth)",
    )
    parser.add_argument(
        "--start-stagger-ms", type=float, default=0,
        help="delay each additional realtime session start by this many milliseconds",
    )
    parser.add_argument("--hotword-set-id", help="hotword set used for every run")
    parser.add_argument("--speaker-scope", help="registered speaker/library scope passed to stream start")
    parser.add_argument("--token", help="bearer token; prefer ASR_API_TOKEN environment variable")
    parser.add_argument("--output", type=Path, default=Path("loadtest-results"))
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--resource-interval", type=float, default=1.0)
    parser.add_argument("--server-pid", type=int, help="sample this server process if psutil is installed")
    parser.add_argument(
        "--resource-process", action="append", default=[], metavar="ROLE=PID",
        help="sample a process root and its descendants; repeat for API/worker roots (overlaps are counted once)",
    )
    parser.add_argument("--max-error-rate", type=float, default=0.01)
    parser.add_argument("--max-first-partial-p95", type=float, default=2.0)
    parser.add_argument(
        "--max-completion-after-stop-p95", "--max-final-p95",
        dest="max_completion_after_stop_p95", type=float, default=8.0,
        help="maximum P95 seconds from sending stop to completed (legacy alias: --max-final-p95)",
    )
    parser.add_argument("--max-backlog", type=float, default=0.5)
    parser.add_argument("--max-server-drops", type=int, default=0)
    parser.add_argument("--target-os", help="target ASR server OS/version; recorded as supplied")
    parser.add_argument("--target-cpu-model", help="target ASR server CPU model")
    parser.add_argument("--target-cpu-cores", type=int, help="target ASR server core count")
    parser.add_argument("--target-memory-bytes", type=int, help="target ASR server RAM in bytes")
    parser.add_argument("--target-gpu-model", help="target ASR server GPU model")
    parser.add_argument("--target-gpu-memory-bytes", type=int, help="target GPU memory in bytes")
    parser.add_argument("--model-version", help="model bundle/version identifier")
    parser.add_argument("--inference-threads", type=int, help="server inference thread count")
    parser.add_argument("--worker-count", type=int, help="server file worker count")
    parser.add_argument("--hotword-count", type=int, help="number of terms in the hotword set")
    parser.add_argument("--audio-notes", help="authorized sample language/noise/speaker-content notes")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    parsed_targets: list[tuple[str, int]] = []
    for raw_target in args.resource_process:
        role, separator, raw_pid = raw_target.partition("=")
        try:
            pid = int(raw_pid)
        except ValueError:
            pid = 0
        if not separator or not role.strip() or pid < 1:
            parser.error("--resource-process must be ROLE=positive-PID")
        parsed_targets.append((role.strip(), pid))
    args.resource_process = parsed_targets
    import os
    args.token = args.token or os.environ.get("ASR_API_TOKEN")
    if (
        args.file_concurrency < 1 or args.repeats < 1 or args.timeout <= 0
        or args.poll_interval <= 0 or args.resource_interval <= 0
        or args.max_file_jobs_per_tier < 1 or not math.isfinite(args.start_stagger_ms)
        or args.start_stagger_ms < 0
        or (args.tier_duration_seconds is not None and
            (not math.isfinite(args.tier_duration_seconds) or args.tier_duration_seconds <= 0))
        or not 0 <= args.max_error_rate <= 1
        or min(args.max_first_partial_p95, args.max_completion_after_stop_p95, args.max_backlog, args.max_server_drops) < 0
    ):
        parser.error("concurrency/repeats/intervals must be positive and thresholds nonnegative (error rate 0..1)")
    try:
        results, samples, config = asyncio.run(run(args))
        json_path, csv_path = write_report(
            args.output, config=config, results=results,
            resource_samples=samples, hardware=hardware_metadata(),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"loadtest: {exc}\n")
    gate_status = json.loads(json_path.read_text(encoding="utf-8"))["gates"]["status"]
    print(f"JSON: {json_path}\nCSV: {csv_path}\nGates: {gate_status}")
    return 1 if not results or gate_status == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
