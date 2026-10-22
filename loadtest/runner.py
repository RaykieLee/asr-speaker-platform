"""CLI and async load runner for the draft ASR HTTP/WebSocket API."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import tempfile
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .core import ResourceSampler, SAMPLE_RATE, SAMPLE_WIDTH, SessionResult, hardware_metadata, pace_pcm, sha256_file, write_report


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
) -> SessionResult:
    try:
        import websockets  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("websockets is required; install loadtest/requirements.txt") from exc
    result = SessionResult(
        run_id=run_id,
        mode="realtime",
        started_at=datetime.now(timezone.utc).isoformat(),
        start_monotonic=time.monotonic(),
        audio_seconds=len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH),
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
                if _is_type(event, "error"):
                    raise RuntimeError(f"server_error: {event.get('code', event.get('message', event))}")
                first_event = event
                break
            else:
                raise TimeoutError("start acknowledgement timed out")

            final_segments: dict[str, tuple[float, str]] = {}
            stop_sent_at: float | None = None

            def record(event: dict[str, Any]) -> None:
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
                    try:
                        segment_start = float(event.get("start", len(final_segments)))
                    except (TypeError, ValueError):
                        segment_start = float(len(final_segments))
                    final_segments[segment_id] = (segment_start, text)
                    result.final_text = "".join(part for _, part in sorted(final_segments.values(), key=lambda value: value[0]))
                    if result.final_text_seconds is None:
                        result.final_text_seconds = received
                    speaker = event.get("speaker_name") or event.get("registered_speaker_id") or event.get("local_speaker_id")
                    if speaker:
                        result.speaker = str(speaker)
                        if result.speaker_seconds is None:
                            result.speaker_seconds = received
                if _is_type(event, "final"):
                    text = str(event.get("text", ""))
                    result.final_text = (result.final_text + text).strip() or text
                    if result.final_text_seconds is None:
                        result.final_text_seconds = received
                    speaker = event.get("speaker_name") or event.get("registered_speaker_id") or event.get("local_speaker_id")
                    if speaker:
                        result.speaker = str(speaker)
                        if result.speaker_seconds is None:
                            result.speaker_seconds = received
                if "dropped_frames" in event:
                    result.server_dropped_frames = int(event["dropped_frames"])
                if _is_type(event, "error"):
                    raise RuntimeError(f"server_error: {event.get('code', event.get('message', event))}")
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
            paced = await pace_pcm(pcm, socket.send, frame_ms=frame_ms)
            result.frames_sent = int(paced["frames_sent"])
            result.max_client_backlog_seconds = float(paced["max_client_backlog_seconds"])
            result.client_late_frames = int(paced["client_late_frames"])
            stop_sent_at = time.monotonic()
            await socket.send(json.dumps({"type": "stop"}))
            try:
                await asyncio.wait_for(reader_task, timeout=timeout_seconds)
            except TimeoutError:
                reader_task.cancel()
                await asyncio.gather(reader_task, return_exceptions=True)
                raise TimeoutError("stream completion timed out")
            if reader_error:
                raise reader_error[0]
    except Exception as exc:
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
        async with httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_seconds) as client:
            data: dict[str, str] = {"language": "auto", "channel_mode": "mono"}
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
                    if job.get("processing_seconds") is not None:
                        result.processing_seconds = float(job["processing_seconds"])
                    elif running_at is not None:
                        result.processing_seconds = ended - running_at
                    result.rtf = (ended - started) / result.audio_seconds if result.audio_seconds else None
                    segments = job.get("segments", [])
                    result.final_text = "".join(str(segment.get("text", "")) for segment in segments)
                    speakers = [segment.get("speaker_name") or segment.get("registered_speaker_id") or segment.get("local_speaker_id") for segment in segments]
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


async def run(args: argparse.Namespace) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    source_pcm = load_wav(args.wav)
    source_audio_seconds = len(source_pcm) / (SAMPLE_RATE * SAMPLE_WIDTH)
    pcm = source_pcm * args.repeats if args.mode in {"realtime", "mixed"} else source_pcm
    audio_seconds = source_audio_seconds * args.repeats
    run_started = time.monotonic()
    sampler = ResourceSampler(args.resource_interval, args.server_pid)
    results: list[Any] = []
    temp_dir = tempfile.TemporaryDirectory(prefix="asr-loadtest-")
    upload_path = args.wav
    if args.repeats > 1 and args.mode in {"file", "mixed"}:
        upload_path = Path(temp_dir.name) / "repeated-input.wav"
        write_repeated_wav(upload_path, source_pcm, args.repeats)
    await sampler.start()
    try:
        for level in args.concurrency:
            run_tag = f"c{level}-{uuid.uuid4().hex[:8]}"
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
                    ))
            if args.mode == "file":
                file_count = level
            elif args.mode == "mixed":
                file_count = args.file_concurrency
            else:
                file_count = 0
            for index in range(file_count):
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
            for item in batch:
                if item.mode == "file":
                    item.audio_seconds = audio_seconds
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
        "source_audio_seconds": source_audio_seconds,
        "audio_seconds": audio_seconds,
        "wall_elapsed_seconds": time.monotonic() - run_started,
        "server_pid": args.server_pid,
        "thresholds": {
            "max_error_rate": args.max_error_rate,
            "max_first_partial_p95_seconds": args.max_first_partial_p95,
            "max_final_p95_seconds": args.max_final_p95,
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
    parser.add_argument("--hotword-set-id", help="hotword set used for every run")
    parser.add_argument("--speaker-scope", help="registered speaker/library scope passed to stream start")
    parser.add_argument("--token", help="bearer token; prefer ASR_API_TOKEN environment variable")
    parser.add_argument("--output", type=Path, default=Path("loadtest-results"))
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--resource-interval", type=float, default=1.0)
    parser.add_argument("--server-pid", type=int, help="sample this server process if psutil is installed")
    parser.add_argument("--max-error-rate", type=float, default=0.01)
    parser.add_argument("--max-first-partial-p95", type=float, default=2.0)
    parser.add_argument("--max-final-p95", type=float, default=8.0)
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
    import os
    args.token = args.token or os.environ.get("ASR_API_TOKEN")
    if (
        args.file_concurrency < 1 or args.repeats < 1 or args.timeout <= 0
        or args.poll_interval <= 0 or args.resource_interval <= 0
        or not 0 <= args.max_error_rate <= 1
        or min(args.max_first_partial_p95, args.max_final_p95, args.max_backlog, args.max_server_drops) < 0
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
