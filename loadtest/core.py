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
from dataclasses import asdict, dataclass, field
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
    on_progress: Callable[[int, float, int], None] | None = None,
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
        if on_progress is not None:
            on_progress(sent, max_lag, late_frames)
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
    server_error_code: str = ""
    server_error_message: str = ""
    websocket_close_code: int | None = None
    websocket_close_reason: str = ""
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
    concurrency_tier: int | None = None
    final_segment_count: int | None = None
    final_speaker_labeled_segment_count: int | None = None
    speaker_change_count: int | None = None
    speaker_label_latency_sample_count: int | None = None
    speaker_label_latency_p50_seconds: float | None = None
    speaker_label_latency_p95_seconds: float | None = None
    speaker_label_latency_p99_seconds: float | None = None
    # Kept only in memory for exact tier-wide percentiles; omitted from all reports.
    speaker_label_latency_samples: list[float] = field(default_factory=list, repr=False)


def _valid_revision(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        revision = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return revision if revision >= 0 else None


def _speaker_label(event: dict[str, Any]) -> str | None:
    # A local ID is stable within the session even if a registered name is attached later.
    for key in ("local_speaker_id", "registered_speaker_id", "speaker_name"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _revision_replaces(new_revision: int | None, old_revision: int | None,
                       new_sequence: int, old_sequence: int) -> bool:
    if new_revision is not None and old_revision is not None:
        return new_revision > old_revision or (new_revision == old_revision and new_sequence >= old_sequence)
    # For mixed/absent revision fields, use arrival order as the only common clock.
    return new_sequence >= old_sequence


def aggregate_final_speaker_events(
    events: Iterable[dict[str, Any]], *, audio_started_at: float | None
) -> dict[str, Any]:
    """Reduce final/corrected segment events to counts and timing only.

    Events must contain private `_received_monotonic` values supplied by the
    runner. Text and speaker identities are discarded from the returned metrics.
    Highest revision wins; arrival order breaks ties and is used when revisions
    are absent.
    """
    latest: dict[str, tuple[int | None, int, dict[str, Any]]] = {}
    for sequence, event in enumerate(events):
        if str(event.get("type", "")).lower() != "segment" or str(event.get("status", "")).lower() not in {"final", "corrected"}:
            continue
        if event.get("segment_id") is None:
            continue
        segment_id = str(event["segment_id"])
        revision = _valid_revision(event.get("revision"))
        previous = latest.get(segment_id)
        if previous is None:
            latest[segment_id] = (revision, sequence, event)
            continue
        old_revision, old_sequence, _ = previous
        if _revision_replaces(revision, old_revision, sequence, old_sequence):
            latest[segment_id] = (revision, sequence, event)

    ordered = sorted((row[2] for row in latest.values()), key=lambda event: (
        _finite_number(event.get("start")) if _finite_number(event.get("start")) is not None else float("inf"),
        str(event.get("segment_id")),
    ))
    labels = [_speaker_label(event) for event in ordered]
    labelled_count = sum(label is not None for label in labels)
    transitions = sum(
        left is not None and right is not None and left != right
        for left, right in zip(labels, labels[1:])
    )
    latencies: list[float] = []
    if audio_started_at is not None and math.isfinite(audio_started_at):
        for event in ordered:
            if _speaker_label(event) is None:
                continue
            end = _finite_number(event.get("end"))
            received = _finite_number(event.get("_received_monotonic"))
            if end is None or received is None:
                continue
            latency = received - (audio_started_at + end)
            if math.isfinite(latency) and latency >= 0:
                latencies.append(latency)
    return {
        "final_segment_count": len(latest) if latest else None,
        "final_speaker_labeled_segment_count": labelled_count if latest else None,
        "speaker_change_count": transitions if labelled_count else None,
        "speaker_label_latency_samples": latencies,
        "speaker_label_latency_sample_count": len(latencies) if latest else None,
        "speaker_label_latency_p50_seconds": percentile(latencies, 50),
        "speaker_label_latency_p95_seconds": percentile(latencies, 95),
        "speaker_label_latency_p99_seconds": percentile(latencies, 99),
    }


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


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
    server_errors: dict[str, int] = {}
    websocket_closes: dict[str, int] = {}
    for item in results:
        if item.error:
            key = item.error.split(":", 1)[0] or "unknown"
            errors[key] = errors.get(key, 0) + 1
        if item.server_error_code:
            code = item.server_error_code
            server_errors[code] = server_errors.get(code, 0) + 1
        if item.websocket_close_code is not None:
            code = str(item.websocket_close_code)
            websocket_closes[code] = websocket_closes.get(code, 0) + 1
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
    speaker_by_tier: dict[str, Any] = {}
    for tier in sorted({item.concurrency_tier for item in live_results if item.concurrency_tier is not None}):
        tier_results = [item for item in live_results if item.concurrency_tier == tier]
        measured = [item for item in tier_results if item.final_segment_count is not None]
        change_measured = [item for item in measured if item.speaker_change_count is not None]
        tier_latencies = [value for item in tier_results for value in item.speaker_label_latency_samples]
        speaker_by_tier[str(tier)] = {
            "session_count": len(tier_results),
            "measured_session_count": len(measured),
            "final_segment_count": sum(item.final_segment_count or 0 for item in measured) if measured else None,
            "speaker_labeled_segment_count": sum(
                item.final_speaker_labeled_segment_count or 0 for item in measured
            ) if measured else None,
            "speaker_change_count": sum(item.speaker_change_count or 0 for item in change_measured)
            if change_measured else None,
            "speaker_change_measured_session_count": len(change_measured),
            "speaker_label_latency_sample_count": len(tier_latencies),
            "speaker_label_latency_seconds": {
                f"p{p}": percentile(tier_latencies, p) for p in (50, 95, 99)
            },
        }
    return {
        "session_count": len(results),
        "success_count": succeeded,
        "error_count": sum(errors.values()),
        "error_counts": errors,
        "server_error_counts": server_errors,
        "websocket_close_code_counts": websocket_closes,
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
        "realtime_speaker_segments_by_concurrency": speaker_by_tier,
        "realtime_speaker_metrics_note": (
            "Segment counts, observed label changes and event latency measure event completeness and timing only; "
            "they do not measure speaker identity accuracy. Missing event data remains unevaluable."
        ),
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
            {key: value for key, value in asdict(item).items()
             if key not in {"start_monotonic", "speaker_label_latency_samples"}}
            for item in results
        ],
        "resource_samples": resource_samples,
        "capacity_claim": None,
        "capacity_claim_note": "This report does not infer capacity; define gates and run on the target deployment machine.",
        "speaker_metrics_note": (
            "Real-time final segment counts, speaker-label transitions, and label event latency describe event "
            "completeness/timing only, not identity accuracy. No additional transcript or voice data is stored."
        ),
    }
    json_path = output_dir / "report.json"
    csv_path = output_dir / "sessions.csv"
    json_path.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    rows = [asdict(item) for item in results]
    fieldnames = [
        "run_id", "mode", "session_id", "started_at", "audio_seconds", "frames_sent", "server_dropped_frames",
        "max_client_backlog_seconds", "client_late_frames", "first_partial_seconds", "final_text_seconds",
        "completion_after_stop_seconds",
        "speaker_seconds", "server_error_code", "server_error_message", "websocket_close_code",
        "websocket_close_reason", "concurrency_tier", "final_segment_count", "final_speaker_labeled_segment_count",
        "speaker_change_count", "speaker_label_latency_sample_count", "speaker_label_latency_p50_seconds",
        "speaker_label_latency_p95_seconds", "speaker_label_latency_p99_seconds",
        "final_text", "speaker", "error", "completed", "job_id", "queue_seconds",
        "processing_seconds", "rtf",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path
