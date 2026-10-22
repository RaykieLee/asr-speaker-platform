"""Metrics, PCM pacing, resource sampling, and report generation."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2


def percentile(values: Iterable[float], p: float) -> float | None:
    """Return a linearly interpolated percentile (p in [0, 100])."""
    numbers = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not numbers:
        return None
    if not 0 <= p <= 100:
        raise ValueError("percentile must be between 0 and 100")
    rank = (len(numbers) - 1) * p / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return numbers[lower]
    return numbers[lower] + (numbers[upper] - numbers[lower]) * (rank - lower)


def pcm_frame_bytes(frame_ms: int, sample_rate: int = SAMPLE_RATE, sample_width: int = SAMPLE_WIDTH) -> int:
    if frame_ms <= 0 or sample_rate <= 0 or sample_width <= 0:
        raise ValueError("frame_ms, sample_rate, and sample_width must be positive")
    samples = sample_rate * frame_ms / 1000
    if not samples.is_integer():
        raise ValueError("frame duration must contain an integer number of samples")
    return int(samples) * sample_width


async def pace_pcm(
    pcm: bytes,
    send: Callable[[bytes], Awaitable[None]],
    *,
    frame_ms: int = 40,
    sample_rate: int = SAMPLE_RATE,
    sample_width: int = SAMPLE_WIDTH,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> dict[str, float | int]:
    """Send PCM frames against a wall-clock audio timeline and expose client backlog."""
    frame_bytes = pcm_frame_bytes(frame_ms, sample_rate, sample_width)
    frame_seconds = frame_ms / 1000
    started = clock()
    sent = 0
    max_lag = 0.0
    late_frames = 0
    for offset in range(0, len(pcm), frame_bytes):
        frame = pcm[offset : offset + frame_bytes]
        target = started + sent * frame_seconds
        delay = target - clock()
        if delay > 0:
            await sleep(delay)
        await send(frame)
        sent += 1
        # Compare against the amount of audio actually transmitted, including a
        # short tail frame, rather than assuming every final packet is full.
        audio_sent = min(len(pcm), offset + len(frame)) / (sample_rate * sample_width)
        lag = max(0.0, clock() - (started + audio_sent))
        max_lag = max(max_lag, lag)
        if lag >= frame_seconds:
            late_frames += 1
    elapsed = max(0.0, clock() - started)
    duration = len(pcm) / (sample_rate * sample_width)
    return {
        "frames_sent": sent,
        "audio_seconds": duration,
        "send_elapsed_seconds": elapsed,
        "max_client_backlog_seconds": max_lag,
        "client_late_frames": late_frames,
    }


@dataclass
class SessionResult:
    run_id: str
    mode: str
    session_id: str = ""
    started_at: str = ""
    start_monotonic: float = 0.0
    audio_seconds: float = 0.0
    frames_sent: int = 0
    server_dropped_frames: int | None = None
    max_client_backlog_seconds: float | None = None
    client_late_frames: int = 0
    first_partial_seconds: float | None = None
    final_text_seconds: float | None = None
    completion_after_stop_seconds: float | None = None
    speaker_seconds: float | None = None
    final_text: str = ""
    speaker: str = ""
    error: str = ""
    completed: bool = False
    job_id: str = ""
    queue_seconds: float | None = None
    processing_seconds: float | None = None
    rtf: float | None = None


class ResourceSampler:
    """Optional psutil sampler; samples the server PID when supplied."""

    def __init__(self, interval_seconds: float = 1.0, pid: int | None = None) -> None:
        self.interval_seconds = interval_seconds
        self.pid = pid
        self.samples: list[dict[str, Any]] = []
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        try:
            import psutil  # type: ignore[import-not-found]
        except ImportError:
            self._psutil = None
        else:
            self._psutil = psutil

    async def start(self) -> None:
        if self._psutil is not None:
            self._task = asyncio.create_task(self._collect())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task

    async def _collect(self) -> None:
        psutil = self._psutil
        assert psutil is not None
        target = None
        while not self._stop.is_set():
            sampled_at = datetime.now(timezone.utc).isoformat()
            try:
                if target is None:
                    target = psutil.Process(self.pid) if self.pid else psutil.Process(os.getpid())
                memory = target.memory_info()
                sample: dict[str, Any] = {
                    "sampled_at": sampled_at,
                    "scope": "server_pid" if self.pid else "loadtest_client_process",
                    "pid": target.pid,
                    "cpu_percent": target.cpu_percent(interval=None),
                    "rss_bytes": memory.rss,
                    "threads": target.num_threads(),
                    "open_files": len(target.open_files()),
                }
                try:
                    sample["pss_bytes"] = target.memory_full_info().pss
                except (AttributeError, psutil.AccessDenied, OSError):
                    sample["pss_bytes"] = None
                try:
                    io_counters = target.io_counters()
                    sample["read_bytes"] = io_counters.read_bytes
                    sample["write_bytes"] = io_counters.write_bytes
                except (AttributeError, psutil.AccessDenied, OSError):
                    sample["read_bytes"] = None
                    sample["write_bytes"] = None
                self.samples.append(sample)
                if not self.pid:
                    self.samples[-1]["system_cpu_percent"] = psutil.cpu_percent(interval=None)
                    self.samples[-1]["system_memory_percent"] = psutil.virtual_memory().percent
            except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                self.samples.append({"sampled_at": sampled_at, "scope": "server_pid", "error": str(exc)})
                return
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
            except TimeoutError:
                pass


def hardware_metadata() -> dict[str, Any]:
    """Collect identifying metadata only; capacity claims require an actual run."""
    result: dict[str, Any] = {
        "system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "hostname": platform.node(),
        "scope": "loadtest_client_machine",
        "gpu_model": None,
        "gpu_memory_bytes": None,
    }
    try:
        import psutil  # type: ignore[import-not-found]
        result["memory_total_bytes"] = psutil.virtual_memory().total
    except ImportError:
        result["memory_total_bytes"] = None
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_gates(summary: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    """Evaluate configured run gates without inferring deployment capacity."""
    error_rate = 1 - summary["success_rate"] if summary["success_rate"] is not None else None
    checks = {
        "error_rate": (error_rate, thresholds.get("max_error_rate")),
        "first_partial_p95_seconds": (
            summary["latency_seconds"]["first_partial"]["p95"],
            thresholds.get("max_first_partial_p95_seconds"),
        ),
        "completion_after_stop_p95_seconds": (
            summary["latency_seconds"]["completion_after_stop"]["p95"],
            thresholds.get("max_final_p95_seconds"),
        ),
        "max_client_backlog_seconds": (
            summary["max_client_backlog_seconds"],
            thresholds.get("max_client_backlog_seconds"),
        ),
        "server_dropped_frames": (
            summary["server_dropped_frames"], thresholds.get("max_server_dropped_frames"),
        ),
    }
    evaluated: dict[str, Any] = {}
    for name, (value, limit) in checks.items():
        if value is None or limit is None:
            evaluated[name] = {"status": "not_evaluable", "value": value, "limit": limit}
        else:
            evaluated[name] = {"status": "pass" if value <= limit else "fail", "value": value, "limit": limit}
    statuses = {item["status"] for item in evaluated.values()}
    overall = "fail" if "fail" in statuses else "not_evaluable" if "not_evaluable" in statuses else "pass"
    return {"status": overall, "checks": evaluated}


def summarize(results: list[SessionResult]) -> dict[str, Any]:
    elapsed_values = [item.final_text_seconds for item in results if item.final_text_seconds is not None]
    completion_values = [item.completion_after_stop_seconds for item in results if item.completion_after_stop_seconds is not None]
    partial_values = [item.first_partial_seconds for item in results if item.first_partial_seconds is not None]
    speaker_values = [item.speaker_seconds for item in results if item.speaker_seconds is not None]
    errors: dict[str, int] = {}
    for item in results:
        if item.error:
            key = item.error.split(":", 1)[0] or "unknown"
            errors[key] = errors.get(key, 0) + 1
    succeeded = sum(item.completed and not item.error for item in results)
    audio_seconds = sum(item.audio_seconds for item in results)
    file_results = [item for item in results if item.mode == "file"]
    successful_files = [item for item in file_results if item.completed and not item.error]
    processing_samples = [
        item for item in successful_files
        if item.processing_seconds is not None and item.processing_seconds > 0
    ]
    file_audio_seconds = sum(item.audio_seconds for item in successful_files)
    measured_file_audio_seconds = sum(item.audio_seconds for item in processing_samples)
    processing_seconds = sum(item.processing_seconds for item in processing_samples if item.processing_seconds is not None)
    live_results = [item for item in results if item.mode == "realtime"]
    drop_coverage = sum(item.server_dropped_frames is not None for item in live_results)
    return {
        "session_count": len(results),
        "success_count": succeeded,
        "error_count": sum(errors.values()),
        "error_counts": errors,
        "success_rate": succeeded / len(results) if results else None,
        "audio_seconds": audio_seconds,
        "file_audio_seconds": file_audio_seconds,
        "file_audio_seconds_with_processing_measurement": measured_file_audio_seconds,
        "processing_seconds": processing_seconds if processing_samples else None,
        "file_audio_hours_per_processing_hour": measured_file_audio_seconds / processing_seconds if processing_seconds > 0 else None,
        "file_processing_sample_count": len(processing_samples),
        "file_attempted_audio_seconds": sum(item.audio_seconds for item in file_results),
        "latency_seconds": {
            "first_partial": {f"p{p}": percentile(partial_values, p) for p in (50, 95, 99)},
            "final_text": {f"p{p}": percentile(elapsed_values, p) for p in (50, 95, 99)},
            "completion_after_stop": {f"p{p}": percentile(completion_values, p) for p in (50, 95, 99)},
            "speaker": {f"p{p}": percentile(speaker_values, p) for p in (50, 95, 99)},
        },
        "client_late_frames": sum(item.client_late_frames for item in results),
        "server_dropped_frames": sum(item.server_dropped_frames or 0 for item in live_results)
        if live_results and drop_coverage == len(live_results) else None,
        "server_drop_telemetry_coverage": {"reported": drop_coverage, "sessions": len(live_results)},
        "max_client_backlog_seconds": max(
            (item.max_client_backlog_seconds for item in results
             if item.mode == "realtime" and item.max_client_backlog_seconds is not None),
            default=None,
        ),
        "queue_seconds_p95": percentile((item.queue_seconds for item in results if item.queue_seconds is not None), 95),
        "rtf_p95": percentile((item.rtf for item in results if item.rtf is not None), 95),
    }


def write_report(
    output_dir: Path,
    *,
    config: dict[str, Any],
    results: list[SessionResult],
    resource_samples: list[dict[str, Any]],
    hardware: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)
    wall_elapsed = config.get("wall_elapsed_seconds")
    file_audio_seconds = sum(
        item.audio_seconds for item in results if item.mode == "file" and item.completed and not item.error
    )
    summary["file_audio_hours_per_wall_hour"] = (
        file_audio_seconds / float(wall_elapsed)
        if wall_elapsed is not None and float(wall_elapsed) > 0
        else None
    )
    thresholds = config.get("thresholds", {})
    document = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "hardware": hardware if hardware is not None else hardware_metadata(),
        "summary": summary,
        "gates": evaluate_gates(summary, thresholds),
        "sessions": [
            {key: value for key, value in asdict(item).items() if key != "start_monotonic"}
            for item in results
        ],
        "resource_samples": resource_samples,
        "capacity_claim": None,
        "capacity_claim_note": "This report does not infer capacity; define gates and run on the target deployment machine.",
    }
    json_path = output_dir / "report.json"
    csv_path = output_dir / "sessions.csv"
    json_path.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    rows = [asdict(item) for item in results]
    fieldnames = [
        "run_id", "mode", "session_id", "started_at", "audio_seconds", "frames_sent", "server_dropped_frames",
        "max_client_backlog_seconds", "client_late_frames", "first_partial_seconds", "final_text_seconds",
        "completion_after_stop_seconds",
        "speaker_seconds", "final_text", "speaker", "error", "completed", "job_id", "queue_seconds",
        "processing_seconds", "rtf",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path
