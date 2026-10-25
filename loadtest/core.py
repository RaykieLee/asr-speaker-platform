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
    repeat_count: int = 1,
    max_audio_seconds: float | None = None,
) -> dict[str, float | int]:
    """Send PCM frames against a wall-clock audio timeline and expose client backlog."""
    if repeat_count < 1:
        raise ValueError("repeat_count must be positive")
    if max_audio_seconds is not None and (not math.isfinite(max_audio_seconds) or max_audio_seconds <= 0):
        raise ValueError("max_audio_seconds must be positive and finite")
    if not pcm:
        raise ValueError("PCM must not be empty")
    frame_bytes = pcm_frame_bytes(frame_ms, sample_rate, sample_width)
    frame_seconds = frame_ms / 1000
    bytes_per_second = sample_rate * sample_width
    available_bytes = len(pcm) * repeat_count
    if max_audio_seconds is not None:
        byte_limit = int(max_audio_seconds * bytes_per_second)
        byte_limit -= byte_limit % sample_width
        available_bytes = min(available_bytes, byte_limit)
        if available_bytes == 0:
            raise ValueError("max_audio_seconds is shorter than one PCM sample")
    started = clock()
    sent = 0
    max_lag = 0.0
    late_frames = 0
    for offset in range(0, available_bytes, frame_bytes):
        frame_end = min(available_bytes, offset + frame_bytes)
        logical_offset = offset
        frame_parts: list[bytes] = []
        while logical_offset < frame_end:
            source_offset = logical_offset % len(pcm)
            part_end = min(frame_end - logical_offset, len(pcm) - source_offset)
            frame_parts.append(pcm[source_offset : source_offset + part_end])
            logical_offset += part_end
        frame = b"".join(frame_parts)
        target = started + sent * frame_seconds
        delay = target - clock()
        if delay > 0:
            await sleep(delay)
        await send(frame)
        sent += 1
        # Compare against the amount of audio actually transmitted, including a
        # short tail frame, rather than assuming every final packet is full.
        audio_sent = min(available_bytes, offset + len(frame)) / bytes_per_second
        lag = max(0.0, clock() - (started + audio_sent))
        max_lag = max(max_lag, lag)
        if lag >= frame_seconds:
            late_frames += 1
        if on_progress is not None:
            on_progress(sent, max_lag, late_frames)
    elapsed = max(0.0, clock() - started)
    duration = available_bytes / bytes_per_second
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
    first_final_segment_seconds: float | None = None
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
    """Optionally sample one or more process roots and their unique descendants."""

    def __init__(self, interval_seconds: float = 1.0, pid: int | None = None,
                 process_targets: list[tuple[str, int]] | None = None) -> None:
        self.interval_seconds = interval_seconds
        self.pid = pid
        self.process_targets = list(process_targets or [])
        self.current_concurrency_tier: int | None = None
        if pid is not None and not self.process_targets:
            self.process_targets = [("api", pid)]
        self.samples: list[dict[str, Any]] = []
        # (process creation time, cumulative user+system CPU seconds, wall time)
        self._cpu_baselines: dict[int, tuple[float, float, float]] = {}
        self._io_baselines: dict[int, tuple[float, int, int, float]] = {}
        self._host_disk_baseline: tuple[int, int, float] | None = None
        self._network_baseline: tuple[int, int, float] | None = None
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
        while not self._stop.is_set():
            sampled_at = datetime.now(timezone.utc).isoformat()
            try:
                sample_monotonic = time.monotonic()
                targets = self.process_targets or [("loadtest_client", os.getpid())]
                process_candidates: dict[int, tuple[int, int, str, Any]] = {}
                target_errors: list[dict[str, Any]] = []
                cpu_errors: list[dict[str, Any]] = []
                io_errors: list[dict[str, Any]] = []
                for root_index, (role, root_pid) in enumerate(targets):
                    try:
                        root = psutil.Process(root_pid)
                        queue = [(root, 0)]
                        visited: set[int] = set()
                        while queue:
                            process, distance = queue.pop(0)
                            if process.pid in visited:
                                continue
                            visited.add(process.pid)
                            candidate = (distance, root_index, role, process)
                            existing = process_candidates.get(process.pid)
                            if existing is None or candidate[:2] < existing[:2]:
                                process_candidates[process.pid] = candidate
                            try:
                                queue.extend((child, distance + 1)
                                             for child in process.children(recursive=False))
                            except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                                target_errors.append({"role": role, "pid": process.pid,
                                                      "error": type(exc).__name__})
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                        target_errors.append({"role": role, "pid": root_pid, "error": type(exc).__name__})
                process_rows: list[dict[str, Any]] = []
                next_cpu_baselines: dict[int, tuple[float, float, float]] = {}
                next_io_baselines: dict[int, tuple[float, int, int, float]] = {}
                for process_pid, (_distance, _root_index, role, process) in sorted(process_candidates.items()):
                    try:
                        memory = process.memory_info()
                        cpu_percent: float | None = None
                        try:
                            identity = float(process.create_time())
                            cpu_times = process.cpu_times()
                            cumulative_cpu = float(cpu_times.user) + float(cpu_times.system)
                            previous = self._cpu_baselines.get(process_pid)
                            if (previous is not None and previous[0] == identity
                                    and sample_monotonic > previous[2]
                                    and cumulative_cpu >= previous[1]):
                                cpu_percent = (cumulative_cpu - previous[1]) / (sample_monotonic - previous[2]) * 100
                            next_cpu_baselines[process_pid] = (identity, cumulative_cpu, sample_monotonic)
                        except (psutil.NoSuchProcess, AttributeError, psutil.AccessDenied,
                                OSError, TypeError, ValueError) as exc:
                            if isinstance(exc, psutil.NoSuchProcess):
                                raise
                            cpu_errors.append({"pid": process_pid, "error": type(exc).__name__})
                        row: dict[str, Any] = {
                            "role": role, "pid": process_pid,
                            "cpu_percent": cpu_percent, "rss_bytes": memory.rss,
                            "threads": process.num_threads(),
                        }
                        io_stats = None
                        try:
                            identity = float(process.create_time())
                            io_stats = process.io_counters()
                            read_bytes, write_bytes = int(io_stats.read_bytes), int(io_stats.write_bytes)
                            previous_io = self._io_baselines.get(process_pid)
                            read_rate = write_rate = None
                            read_rate_reason = write_rate_reason = "baseline_unavailable"
                            if (previous_io is not None and previous_io[0] == identity
                                    and sample_monotonic > previous_io[3]):
                                elapsed = sample_monotonic - previous_io[3]
                                if read_bytes >= previous_io[1]:
                                    read_rate = (read_bytes - previous_io[1]) / elapsed
                                    read_rate_reason = None
                                else:
                                    read_rate_reason = "counter_reset"
                                if write_bytes >= previous_io[2]:
                                    write_rate = (write_bytes - previous_io[2]) / elapsed
                                    write_rate_reason = None
                                else:
                                    write_rate_reason = "counter_reset"
                            next_io_baselines[process_pid] = (identity, read_bytes, write_bytes, sample_monotonic)
                            row.update({"disk_read_bytes": read_bytes, "disk_write_bytes": write_bytes,
                                        "disk_read_bytes_per_second": read_rate,
                                        "disk_write_bytes_per_second": write_rate,
                                        "disk_read_rate_unavailable_reason": read_rate_reason,
                                        "disk_write_rate_unavailable_reason": write_rate_reason})
                        except (AttributeError, psutil.AccessDenied, OSError, TypeError, ValueError) as exc:
                            io_errors.append({"pid": process_pid, "metric": "disk_io", "error": type(exc).__name__})
                            row.update({"disk_read_bytes": None, "disk_write_bytes": None,
                                        "disk_read_bytes_per_second": None, "disk_write_bytes_per_second": None,
                                        "disk_io_unavailable_reason": type(exc).__name__})
                        try:
                            if hasattr(process, "num_fds"):
                                row["file_handles"] = int(process.num_fds())
                            elif hasattr(process, "num_handles"):
                                row["file_handles"] = int(process.num_handles())
                            else:
                                row["file_handles"] = None
                                row["file_handles_unavailable_reason"] = "unsupported_platform"
                        except (AttributeError, psutil.AccessDenied, OSError, TypeError, ValueError) as exc:
                            row["file_handles"] = None
                            row["file_handles_unavailable_reason"] = type(exc).__name__
                        try:
                            row["pss_bytes"] = process.memory_full_info().pss
                        except (AttributeError, psutil.AccessDenied, OSError):
                            row["pss_bytes"] = None
                        process_rows.append(row)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        target_errors.append({"pid": process_pid, "error": "process_unavailable"})
                if self.process_targets:
                    totals_available = bool(process_rows) and not target_errors
                    cpu_total_available = totals_available and all(row["cpu_percent"] is not None
                                                                   for row in process_rows)
                    sample = {
                        "sampled_at": sampled_at, "scope": "process_tree_total",
                        "concurrency_tier": self.current_concurrency_tier,
                        "disk_process_scope": "process_tree",
                        "cpu_percent": (sum(row["cpu_percent"] for row in process_rows)
                                        if cpu_total_available else None),
                        "rss_bytes": sum(row["rss_bytes"] for row in process_rows) if totals_available else None,
                        "threads": sum(row["threads"] for row in process_rows) if totals_available else None,
                        "pss_bytes": (sum(row["pss_bytes"] for row in process_rows)
                                      if totals_available and all(row["pss_bytes"] is not None for row in process_rows)
                                      else None),
                        "disk_read_bytes": (sum(row["disk_read_bytes"] for row in process_rows)
                                            if totals_available and all(row.get("disk_read_bytes") is not None for row in process_rows) else None),
                        "disk_write_bytes": (sum(row["disk_write_bytes"] for row in process_rows)
                                             if totals_available and all(row.get("disk_write_bytes") is not None for row in process_rows) else None),
                        "disk_read_bytes_per_second": (sum(row["disk_read_bytes_per_second"] for row in process_rows)
                                                       if totals_available and all(row.get("disk_read_bytes_per_second") is not None for row in process_rows) else None),
                        "disk_write_bytes_per_second": (sum(row["disk_write_bytes_per_second"] for row in process_rows)
                                                        if totals_available and all(row.get("disk_write_bytes_per_second") is not None for row in process_rows) else None),
                        "file_handles": (sum(row["file_handles"] for row in process_rows)
                                         if totals_available and all(row.get("file_handles") is not None for row in process_rows) else None),
                        "processes": process_rows,
                        "target_errors": target_errors,
                        "cpu_errors": cpu_errors,
                        "io_errors": io_errors,
                    }
                    for key in ("disk_read_bytes", "disk_write_bytes", "disk_read_bytes_per_second",
                                "disk_write_bytes_per_second", "file_handles"):
                        if sample[key] is None:
                            sample[f"{key}_unavailable_reason"] = (
                                "process_unavailable" if target_errors else
                                "process_metric_unavailable" if io_errors or any(row.get("file_handles") is None for row in process_rows)
                                else "baseline_unavailable"
                            )
                else:
                    row = process_rows[0] if process_rows else {}
                    sample = {"sampled_at": sampled_at, "scope": "loadtest_client_process",
                              "concurrency_tier": self.current_concurrency_tier,
                              "disk_process_scope": "loadtest_client_process", **row}
                    for key in ("disk_read_bytes", "disk_write_bytes", "disk_read_bytes_per_second",
                                "disk_write_bytes_per_second", "file_handles"):
                        if key not in sample:
                            sample[key] = None
                            sample[f"{key}_unavailable_reason"] = "process_metric_unavailable"
                    sample["system_cpu_percent"] = psutil.cpu_percent(interval=None)
                    sample["system_memory_percent"] = psutil.virtual_memory().percent
                try:
                    disk = psutil.disk_io_counters()
                    if disk is None:
                        raise OSError("host_disk_counters_unavailable")
                    read_bytes, write_bytes = int(disk.read_bytes), int(disk.write_bytes)
                    prior_disk = self._host_disk_baseline
                    sample.update({"disk_host_scope": "host", "disk_host_read_bytes": read_bytes,
                                   "disk_host_write_bytes": write_bytes})
                    if prior_disk is not None and sample_monotonic > prior_disk[2] and read_bytes >= prior_disk[0] and write_bytes >= prior_disk[1]:
                        elapsed = sample_monotonic - prior_disk[2]
                        sample["disk_host_read_bytes_per_second"] = (read_bytes - prior_disk[0]) / elapsed
                        sample["disk_host_write_bytes_per_second"] = (write_bytes - prior_disk[1]) / elapsed
                    else:
                        sample["disk_host_read_bytes_per_second"] = None
                        sample["disk_host_write_bytes_per_second"] = None
                        sample["disk_host_read_rate_unavailable_reason"] = (
                            "baseline_unavailable" if prior_disk is None else "counter_reset")
                        sample["disk_host_write_rate_unavailable_reason"] = (
                            "baseline_unavailable" if prior_disk is None else "counter_reset")
                    self._host_disk_baseline = (read_bytes, write_bytes, sample_monotonic)
                except (AttributeError, OSError, psutil.AccessDenied, TypeError, ValueError) as exc:
                    sample.update({"disk_host_scope": "host", "disk_host_read_bytes": None,
                                   "disk_host_write_bytes": None,
                                   "disk_host_read_bytes_per_second": None,
                                   "disk_host_write_bytes_per_second": None,
                                   "disk_host_read_rate_unavailable_reason": type(exc).__name__,
                                   "disk_host_write_rate_unavailable_reason": type(exc).__name__,
                                   "disk_host_read_unavailable_reason": type(exc).__name__,
                                   "disk_host_write_unavailable_reason": type(exc).__name__})
                try:
                    network = psutil.net_io_counters()
                    if network is None:
                        raise OSError("network_counters_unavailable")
                    sent, received = int(network.bytes_sent), int(network.bytes_recv)
                    prior = self._network_baseline
                    sample.update({"network_scope": "host", "network_bytes_sent": sent,
                                   "network_bytes_received": received})
                    if prior is not None and sample_monotonic > prior[2] and sent >= prior[0] and received >= prior[1]:
                        elapsed = sample_monotonic - prior[2]
                        sample["network_bytes_sent_per_second"] = (sent - prior[0]) / elapsed
                        sample["network_bytes_received_per_second"] = (received - prior[1]) / elapsed
                    else:
                        sample["network_bytes_sent_per_second"] = None
                        sample["network_bytes_received_per_second"] = None
                        sample["network_unavailable_reason"] = "baseline_unavailable" if prior is None else "counter_reset"
                    self._network_baseline = (sent, received, sample_monotonic)
                except (AttributeError, OSError, psutil.AccessDenied, TypeError, ValueError) as exc:
                    sample.update({"network_scope": "host", "network_bytes_sent": None,
                                   "network_bytes_received": None, "network_bytes_sent_per_second": None,
                                   "network_bytes_received_per_second": None,
                                   "network_unavailable_reason": type(exc).__name__})
                self.samples.append(sample)
                self._cpu_baselines = next_cpu_baselines
                self._io_baselines = next_io_baselines
            except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                self.samples.append({"sampled_at": sampled_at, "scope": "process_tree_total",
                                     "error": type(exc).__name__})
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
            thresholds.get(
                "max_completion_after_stop_p95_seconds",
                thresholds.get("max_final_p95_seconds"),
            ),
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
    elapsed_values = [
        item.first_final_segment_seconds
        if item.first_final_segment_seconds is not None else item.final_text_seconds
        for item in results
        if item.first_final_segment_seconds is not None or item.final_text_seconds is not None
    ]
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
            "first_final_segment": {f"p{p}": percentile(elapsed_values, p) for p in (50, 95, 99)},
            # Legacy alias: this is session-start to first final text/segment arrival,
            # not time relative to the end of the spoken audio.
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
        "first_final_segment_seconds",
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
