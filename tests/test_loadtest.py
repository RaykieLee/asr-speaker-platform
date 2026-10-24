import asyncio
import json
import sys
import time
from types import SimpleNamespace

import pytest

from loadtest.core import (SessionResult, aggregate_final_speaker_events, evaluate_gates,
                           pace_pcm, percentile, summarize, write_report)
from loadtest.runner import (_run_continuous_file_tier, build_parser, load_wav, run,
                            run_file_job, run_live_session, write_repeated_wav)


def test_percentile_uses_linear_interpolation_and_empty_values():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)
    assert percentile([], 95) is None


def test_final_speaker_events_deduplicate_by_segment_and_revision_without_exposing_payload():
    metrics = aggregate_final_speaker_events([
        {"type": "segment", "status": "final", "segment_id": "1", "revision": 1,
         "start": 0, "end": 0.5, "text": "secret transcript", "local_speaker_id": "A",
         "_received_monotonic": 10.6},
        {"type": "segment", "status": "corrected", "segment_id": "1", "revision": 0,
         "start": 0, "end": 0.5, "text": "stale", "local_speaker_id": "wrong",
         "_received_monotonic": 10.7},
        {"type": "segment", "status": "corrected", "segment_id": "1", "revision": 2,
         "start": 0, "end": 0.5, "text": "still secret", "local_speaker_id": "A",
         "_received_monotonic": 10.8},
        {"type": "segment", "status": "final", "segment_id": "2", "revision": 1,
         "start": 0.5, "end": 1.0, "text": "private", "local_speaker_id": "B",
         "_received_monotonic": 11.2},
        {"type": "segment", "status": "final", "segment_id": "3", "revision": 1,
         "start": 1.0, "end": 1.5, "text": "private", "_received_monotonic": 11.7},
    ], audio_started_at=10.0)
    assert metrics["final_segment_count"] == 3
    assert metrics["final_speaker_labeled_segment_count"] == 2
    assert metrics["speaker_change_count"] == 1
    assert metrics["speaker_label_latency_sample_count"] == 2
    assert metrics["speaker_label_latency_p50_seconds"] == pytest.approx(0.25)
    assert "text" not in metrics and "speaker" not in metrics
    empty = aggregate_final_speaker_events([], audio_started_at=10.0)
    assert empty["final_segment_count"] is None
    assert empty["speaker_label_latency_p95_seconds"] is None


def test_speaker_event_metrics_are_summarized_per_concurrency_tier_and_samples_not_written(tmp_path):
    results = [
        SessionResult("a", "realtime", completed=True, concurrency_tier=2,
                      final_segment_count=3, final_speaker_labeled_segment_count=2,
                      speaker_change_count=1, speaker_label_latency_samples=[0.1, 0.2]),
        SessionResult("b", "realtime", completed=True, concurrency_tier=2,
                      final_segment_count=2, final_speaker_labeled_segment_count=2,
                      speaker_change_count=1, speaker_label_latency_samples=[0.3]),
    ]
    report_json, _ = write_report(tmp_path, config={"thresholds": {}}, results=results,
                                  resource_samples=[], hardware={"system": "test"})
    report = json.loads(report_json.read_text(encoding="utf-8"))
    tier = report["summary"]["realtime_speaker_segments_by_concurrency"]["2"]
    assert tier["session_count"] == 2
    assert tier["final_segment_count"] == 5
    assert tier["speaker_labeled_segment_count"] == 4
    assert tier["speaker_change_count"] == 2
    assert tier["speaker_label_latency_seconds"]["p50"] == pytest.approx(0.2)
    assert report["sessions"][0]["speaker_label_latency_sample_count"] is None
    assert "speaker_label_latency_samples" not in report["sessions"][0]


def test_fake_websocket_counts_final_segments_revisions_changes_and_label_latency(monkeypatch):
    class FakeSocket:
        def __init__(self):
            self.queue = asyncio.Queue()
            self.queue.put_nowait(json.dumps({"type": "start_ack", "session_id": "fake-session"}))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def send(self, message):
            if not isinstance(message, str):
                return
            payload = json.loads(message)
            if payload.get("type") == "stop":
                for event in [
                    {"type": "segment", "status": "final", "segment_id": 1, "revision": 1,
                     "start": 0, "end": 0.01, "text": "舊句", "local_speaker_id": "speaker_a"},
                    {"type": "segment", "status": "corrected", "segment_id": 1, "revision": 2,
                     "start": 0, "end": 0.01, "text": "甲", "local_speaker_id": "speaker_a"},
                    {"type": "segment", "status": "final", "segment_id": 2, "revision": 1,
                     "start": 0.01, "end": 0.02, "text": "乙", "local_speaker_id": "speaker_b"},
                    {"type": "completed", "dropped_frames": 0},
                ]:
                    self.queue.put_nowait(json.dumps(event))

        async def recv(self):
            return await self.queue.get()

    fake_socket = FakeSocket()

    def connect(_endpoint, **_kwargs):
        return fake_socket

    monkeypatch.setitem(sys.modules, "websockets", SimpleNamespace(connect=connect))
    result = asyncio.run(run_live_session(
        pcm=bytes(1280), endpoint="ws://fake", run_id="fake-live", frame_ms=20,
        hotword_set_id=None, speaker_scope=None, token=None, timeout_seconds=1,
        concurrency_tier=2,
    ))
    assert result.error == ""
    assert result.completed
    assert result.final_segment_count == 2
    assert result.final_speaker_labeled_segment_count == 2
    assert result.speaker_change_count == 1
    assert result.speaker_label_latency_sample_count == 2
    assert result.speaker_label_latency_p50_seconds is not None
    assert result.speaker_label_latency_p95_seconds is not None
    assert result.speaker_label_latency_p99_seconds is not None
    assert result.final_text == "甲乙"


def test_live_server_error_event_is_preserved_with_close_code_and_partial_frame_metrics(monkeypatch):
    class FakeClosed(Exception):
        def __init__(self, code, reason):
            self.code = code
            self.reason = reason
            super().__init__(f"closed {code}: {reason}")

    class FakeSocket:
        def __init__(self):
            self.queue = asyncio.Queue()
            self.queue.put_nowait(json.dumps({"type":"start_ack","session_id":"overloaded"}))
            self.sent_audio = 0

        async def __aenter__(self): return self
        async def __aexit__(self, *_args): return None

        async def send(self, message):
            if isinstance(message, bytes):
                self.sent_audio += 1
                if self.sent_audio == 1:
                    self.queue.put_nowait(json.dumps({"type":"error","code":"queue_full","message":"capacity exceeded"}))
                    self.queue.put_nowait(FakeClosed(1013, "try again later"))
                    await asyncio.sleep(0)
                else:
                    raise FakeClosed(1013, "try again later")

        async def recv(self):
            item = await self.queue.get()
            if isinstance(item, BaseException): raise item
            return item

    fake = FakeSocket()
    monkeypatch.setitem(sys.modules, "websockets", SimpleNamespace(connect=lambda *_args, **_kwargs: fake))
    result = asyncio.run(run_live_session(
        pcm=bytes(640 * 5), endpoint="ws://fake", run_id="overloaded-live", frame_ms=20,
        hotword_set_id=None, speaker_scope=None, token=None, timeout_seconds=1,
    ))

    assert result.completed is False
    assert result.server_error_code == "queue_full"
    assert result.server_error_message == "capacity exceeded"
    assert result.websocket_close_code == 1013
    assert result.websocket_close_reason == "try again later"
    assert result.error == "server_error:queue_full"
    assert result.frames_sent == 1
    assert result.max_client_backlog_seconds is not None


def test_live_send_close_drains_queued_server_error_before_reader_cancel(monkeypatch):
    class FakeClosed(Exception):
        code = 1013
        reason = "capacity"

    class FakeSocket:
        def __init__(self):
            self.queue = asyncio.Queue()
            self.queue.put_nowait(json.dumps({"type":"start_ack"}))

        async def __aenter__(self): return self
        async def __aexit__(self, *_args): return None

        async def send(self, message):
            if isinstance(message, bytes):
                # Queue both frames and throw without an await/yield. The reader
                # has not had a scheduling turn when this send exception occurs.
                self.queue.put_nowait(json.dumps({"type":"error","code":"capacity_exceeded","message":"queue full"}))
                self.queue.put_nowait(FakeClosed())
                raise FakeClosed()

        async def recv(self):
            item = await self.queue.get()
            if isinstance(item, BaseException): raise item
            return item

    fake = FakeSocket()
    monkeypatch.setitem(sys.modules, "websockets", SimpleNamespace(connect=lambda *_args, **_kwargs: fake))
    result = asyncio.run(run_live_session(
        pcm=bytes(640 * 3), endpoint="ws://fake", run_id="racing-error", frame_ms=20,
        hotword_set_id=None, speaker_scope=None, token=None, timeout_seconds=1,
    ))

    assert result.server_error_code == "capacity_exceeded"
    assert result.server_error_message == "queue full"
    assert result.websocket_close_code == 1013
    assert result.error == "server_error:capacity_exceeded"
    assert result.frames_sent == 0
    assert result.max_client_backlog_seconds is None


def test_live_close_without_server_event_keeps_actual_partial_frame_count(monkeypatch):
    class FakeClosed(Exception):
        code = 1013
        reason = "busy"

    class FakeSocket:
        def __init__(self):
            self.queue = asyncio.Queue()
            self.queue.put_nowait(json.dumps({"type":"start_ack"}))
            self.sent_audio = 0

        async def __aenter__(self): return self
        async def __aexit__(self, *_args): return None

        async def send(self, message):
            if isinstance(message, bytes):
                self.sent_audio += 1
                if self.sent_audio == 1:
                    self.queue.put_nowait(FakeClosed("1013"))
                    await asyncio.sleep(0)
                else:
                    raise FakeClosed("1013")

        async def recv(self):
            item = await self.queue.get()
            if isinstance(item, BaseException): raise item
            return item

    fake = FakeSocket()
    monkeypatch.setitem(sys.modules, "websockets", SimpleNamespace(connect=lambda *_args, **_kwargs: fake))
    result = asyncio.run(run_live_session(
        pcm=bytes(640 * 4), endpoint="ws://fake", run_id="closed-live", frame_ms=20,
        hotword_set_id=None, speaker_scope=None, token=None, timeout_seconds=1,
    ))

    assert result.frames_sent == 1
    assert result.server_error_code == ""
    assert result.websocket_close_code == 1013
    assert "1013" in result.error
    assert result.max_client_backlog_seconds is not None


def test_pcm_pacing_tracks_audio_timeline_and_client_backlog():
    class FakeClock:
        now = 0.0

        def monotonic(self):
            return self.now

        async def sleep(self, duration):
            self.now += duration

    async def scenario():
        clock = FakeClock()
        sends = []

        async def send(frame):
            sends.append((clock.now, len(frame)))
            clock.now += 0.002  # simulate a small network write cost

        result = await pace_pcm(
            bytes(640 * 3), send, frame_ms=20, clock=clock.monotonic, sleep=clock.sleep
        )
        return result, sends

    metrics, sends = asyncio.run(scenario())
    assert [round(sent[0], 3) for sent in sends] == [0.0, 0.02, 0.04]
    assert all(frame_length == 640 for _, frame_length in sends)
    assert metrics["frames_sent"] == 3
    assert metrics["audio_seconds"] == 0.06
    assert metrics["client_late_frames"] == 0


def test_pcm_pacing_repeats_without_expanding_and_truncates_at_tier_duration():
    class FakeClock:
        now = 0.0

        def monotonic(self):
            return self.now

        async def sleep(self, duration):
            self.now += duration

    async def scenario():
        clock = FakeClock()
        frames = []

        async def send(frame):
            frames.append(frame)

        result = await pace_pcm(
            b"a" * 320, send, frame_ms=20, clock=clock.monotonic,
            sleep=clock.sleep, repeat_count=10, max_audio_seconds=0.055,
        )
        return result, frames

    metrics, frames = asyncio.run(scenario())
    assert [len(frame) for frame in frames] == [640, 640, 480]
    assert b"".join(frames) == (b"a" * 320) * 5 + (b"a" * 160)
    assert metrics["frames_sent"] == 3
    assert metrics["audio_seconds"] == pytest.approx(0.055)


def test_continuous_file_tier_bounds_parallelism_duration_and_total_jobs(tmp_path):
    class FakeClock:
        now = 100.0

        def monotonic(self):
            return self.now

    clock = FakeClock()
    active = 0
    peak = 0
    starts = []

    async def fake_job(**kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        starts.append((kwargs["run_id"], clock.now))
        index = int(kwargs["run_id"].rsplit("-", 1)[1])
        await asyncio.sleep(0)
        clock.now += 0.01
        active -= 1
        return SessionResult(
            kwargs["run_id"], "file", completed=index % 2 == 1,
            error="fake failure" if index % 2 == 0 else "",
            queue_seconds=0.1, processing_seconds=0.2,
        )

    results = asyncio.run(_run_continuous_file_tier(
        slots=2, duration_seconds=0.025, max_jobs=100, deadline=100.025,
        path=tmp_path / "fake.wav", audio_seconds=1.0, base_url="http://fake",
        run_tag="tier", hotword_set_id=None, token=None, poll_seconds=0.1,
        timeout_seconds=1.0, concurrency_tier=2, job_runner=fake_job,
        clock=clock.monotonic,
    ))

    assert 2 <= len(results) < 100
    assert peak <= 2 and active == 0
    assert all(start_time < 100.025 for _, start_time in starts)
    assert len({run_id for run_id, _ in starts}) == len(results)
    assert sum(item.completed for item in results) + sum(bool(item.error) for item in results) == len(results)
    assert all(item.concurrency_tier == 2 for item in results)
    assert all(item.queue_seconds == 0.1 and item.processing_seconds == 0.2 for item in results)


def test_mixed_continuous_tier_runs_live_and_file_load_together(monkeypatch, tmp_path):
    wav = tmp_path / "short.wav"
    write_repeated_wav(wav, bytes(320), 1)
    live_active = 0
    file_active = 0
    peak_files = 0
    observed_overlap = []

    class FakeSampler:
        def __init__(self, *_args): self.samples = []
        async def start(self): pass
        async def stop(self): pass

    async def fake_live(**kwargs):
        nonlocal live_active
        live_active += 1
        result = SessionResult(kwargs["run_id"], "realtime", completed=True,
                               audio_seconds=kwargs["max_audio_seconds"],
                               concurrency_tier=kwargs["concurrency_tier"])
        await asyncio.sleep(kwargs["max_audio_seconds"])
        live_active -= 1
        return result

    async def fake_file(**kwargs):
        nonlocal file_active, peak_files
        file_active += 1
        peak_files = max(peak_files, file_active)
        observed_overlap.append(live_active > 0)
        await asyncio.sleep(0.003)
        index = int(kwargs["run_id"].rsplit("-", 1)[1])
        file_active -= 1
        return SessionResult(kwargs["run_id"], "file", completed=index % 2 == 0,
                             error="fake failure" if index % 2 else "",
                             queue_seconds=0.02, processing_seconds=0.03)

    monkeypatch.setattr("loadtest.runner.ResourceSampler", FakeSampler)
    monkeypatch.setattr("loadtest.runner.run_live_session", fake_live)
    monkeypatch.setattr("loadtest.runner.run_file_job", fake_file)
    args = build_parser().parse_args([
        "--wav", str(wav), "--mode", "mixed", "--concurrency", "2",
        "--file-concurrency", "2", "--tier-duration-seconds", "0.03",
        "--max-file-jobs-per-tier", "4",
    ])
    results, samples, config = asyncio.run(run(args))

    live = [item for item in results if item.mode == "realtime"]
    files = [item for item in results if item.mode == "file"]
    assert len(live) == 2
    assert len(files) == 4
    assert peak_files <= 2 and file_active == 0 and live_active == 0
    assert all(observed_overlap)
    assert all(item.concurrency_tier == 2 for item in results)
    assert all(item.queue_seconds == 0.02 and item.processing_seconds == 0.03 for item in files)
    assert samples == []
    assert config["tier_duration_seconds"] == 0.03
    assert config["max_file_jobs_per_tier"] == 4
    assert config["file_jobs_by_concurrency_tier"]["2"] == len(files)


def test_continuous_realtime_stagger_keeps_sessions_inside_tier(monkeypatch, tmp_path):
    wav = tmp_path / "short.wav"
    write_repeated_wav(wav, bytes(320), 1)
    starts = []

    class FakeSampler:
        def __init__(self, *_args): self.samples = []
        async def start(self): pass
        async def stop(self): pass

    async def fake_live(**kwargs):
        await asyncio.sleep(kwargs["start_delay_seconds"])
        starts.append((time.monotonic(), kwargs["max_audio_seconds"]))
        await asyncio.sleep(kwargs["max_audio_seconds"])
        return SessionResult(kwargs["run_id"], "realtime", completed=True,
                             audio_seconds=kwargs["max_audio_seconds"])

    monkeypatch.setattr("loadtest.runner.ResourceSampler", FakeSampler)
    monkeypatch.setattr("loadtest.runner.run_live_session", fake_live)
    args = build_parser().parse_args([
        "--wav", str(wav), "--mode", "realtime", "--concurrency", "3",
        "--tier-duration-seconds", "0.05", "--start-stagger-ms", "10",
    ])
    results, _samples, _config = asyncio.run(run(args))

    assert len(results) == 3
    starts.sort(key=lambda row: row[0])
    assert starts[1][0] - starts[0][0] == pytest.approx(0.01, abs=0.01)
    assert starts[2][0] - starts[1][0] == pytest.approx(0.01, abs=0.01)
    assert [duration for _, duration in starts] == pytest.approx([0.05, 0.04, 0.03])


def test_continuous_realtime_rejects_stagger_outside_tier(monkeypatch, tmp_path):
    wav = tmp_path / "short.wav"
    write_repeated_wav(wav, bytes(320), 1)

    class FakeSampler:
        def __init__(self, *_args): self.samples = []
        async def start(self): pass
        async def stop(self): pass

    monkeypatch.setattr("loadtest.runner.ResourceSampler", FakeSampler)
    args = build_parser().parse_args([
        "--wav", str(wav), "--mode", "realtime", "--concurrency", "3",
        "--tier-duration-seconds", "0.01", "--start-stagger-ms", "10",
    ])
    with pytest.raises(ValueError, match="beyond its tier duration"):
        asyncio.run(run(args))


def test_continuous_cli_options_are_opt_in():
    args = build_parser().parse_args(["--wav", "sample.wav"])
    assert args.tier_duration_seconds is None
    assert args.max_file_jobs_per_tier == 1000
    assert args.start_stagger_ms == 0
    continuous = build_parser().parse_args([
        "--wav", "sample.wav", "--tier-duration-seconds", "600",
        "--max-file-jobs-per-tier", "200", "--start-stagger-ms", "250",
    ])
    assert continuous.tier_duration_seconds == 600
    assert continuous.max_file_jobs_per_tier == 200
    assert continuous.start_stagger_ms == 250


def test_repeated_pcm_can_be_wrapped_as_uploadable_wav(tmp_path):
    pcm = bytes(160 * 2)
    wav_path = tmp_path / "repeated.wav"
    write_repeated_wav(wav_path, pcm, 4)
    assert load_wav(wav_path) == pcm * 4


def test_summary_counts_errors_and_computes_latency_percentiles():
    results = [
        SessionResult("a", "realtime", completed=True, first_partial_seconds=0.1, final_text_seconds=0.3),
        SessionResult("b", "realtime", error="TimeoutError: timed out", first_partial_seconds=0.2),
        SessionResult("c", "file", error="job_failed: engine unavailable"),
    ]
    summary = summarize(results)
    assert summary["session_count"] == 3
    assert summary["success_count"] == 1
    assert summary["error_count"] == 2
    assert summary["error_counts"] == {"TimeoutError": 1, "job_failed": 1}
    assert summary["success_rate"] == 1 / 3
    assert summary["server_dropped_frames"] is None
    assert summary["server_drop_telemetry_coverage"] == {"reported": 0, "sessions": 2}
    assert summary["latency_seconds"]["first_partial"]["p50"] == pytest.approx(0.15)
    assert summary["latency_seconds"]["final_text"]["p50"] == pytest.approx(0.3)
    assert summary["latency_seconds"]["first_final_segment"]["p50"] == pytest.approx(0.3)


def test_first_final_segment_metric_is_named_and_legacy_fields_remain_compatible(tmp_path):
    item = SessionResult(
        "named-final", "realtime", completed=True, final_text_seconds=8.0,
        first_final_segment_seconds=8.0,
    )
    summary = summarize([item])
    assert summary["latency_seconds"]["first_final_segment"]["p50"] == 8.0
    assert summary["latency_seconds"]["final_text"]["p50"] == 8.0
    report_json, report_csv = write_report(
        tmp_path, config={"thresholds": {}}, results=[item], resource_samples=[],
    )
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["sessions"][0]["first_final_segment_seconds"] == 8.0
    assert report["sessions"][0]["final_text_seconds"] == 8.0
    csv_header = report_csv.read_text(encoding="utf-8-sig").splitlines()[0]
    assert "first_final_segment_seconds" in csv_header
    assert "final_text_seconds" in csv_header


def test_structured_live_diagnostics_are_summarized_and_reported_additively(tmp_path):
    item = SessionResult(
        "overload", "realtime", error="server_error:capacity_exceeded",
        server_error_code="capacity_exceeded", server_error_message="worker queue is full",
        websocket_close_code=1013, websocket_close_reason="try again later", frames_sent=4,
        max_client_backlog_seconds=0.02,
    )
    summary = summarize([item])
    assert summary["error_counts"] == {"server_error": 1}
    assert summary["server_error_counts"] == {"capacity_exceeded": 1}
    assert summary["websocket_close_code_counts"] == {"1013": 1}
    report_json, report_csv = write_report(tmp_path, config={"thresholds": {}}, results=[item], resource_samples=[])
    report = json.loads(report_json.read_text(encoding="utf-8"))
    session = report["sessions"][0]
    assert session["server_error_code"] == "capacity_exceeded"
    assert session["websocket_close_code"] == 1013
    csv_text = report_csv.read_text(encoding="utf-8-sig")
    assert "server_error_code" in csv_text and "websocket_close_code" in csv_text


def test_gate_fails_on_error_rate_and_marks_unreported_drops_unknown():
    results = [
        SessionResult("ok", "realtime", completed=True),
        SessionResult("bad", "realtime", error="TimeoutError: timeout"),
    ]
    gates = evaluate_gates(summarize(results), {
        "max_error_rate": 0.01,
        "max_first_partial_p95_seconds": 2,
        "max_final_p95_seconds": 8,
        "max_client_backlog_seconds": 0.5,
        "max_server_dropped_frames": 0,
    })
    assert gates["status"] == "fail"
    assert gates["checks"]["error_rate"]["status"] == "fail"
    assert gates["checks"]["server_dropped_frames"]["status"] == "not_evaluable"
    assert gates["checks"]["max_client_backlog_seconds"]["status"] == "not_evaluable"


def test_realtime_completion_gate_measures_after_stop_not_audio_duration():
    summary = summarize([SessionResult(
        "long-audio", "realtime", completed=True, audio_seconds=20,
        final_text_seconds=20.1, completion_after_stop_seconds=0.2,
        server_dropped_frames=0, max_client_backlog_seconds=0,
    )])
    gates = evaluate_gates(summary, {"max_final_p95_seconds": 8})
    assert gates["checks"]["completion_after_stop_p95_seconds"]["status"] == "pass"
    assert summary["latency_seconds"]["final_text"]["p95"] == pytest.approx(20.1)


def test_completion_gate_flag_aliases_and_report_threshold_keys(tmp_path):
    parser = build_parser()
    current = parser.parse_args(["--wav", "sample.wav", "--max-completion-after-stop-p95", "0.4"])
    legacy = parser.parse_args(["--wav", "sample.wav", "--max-final-p95", "0.4"])
    assert current.max_completion_after_stop_p95 == legacy.max_completion_after_stop_p95 == 0.4

    summary = summarize([SessionResult(
        "s", "realtime", completed=True, completion_after_stop_seconds=0.5,
    )])
    current_gate = evaluate_gates(summary, {"max_completion_after_stop_p95_seconds": 0.4})
    legacy_gate = evaluate_gates(summary, {"max_final_p95_seconds": 0.4})
    assert current_gate == legacy_gate
    assert current_gate["checks"]["completion_after_stop_p95_seconds"]["status"] == "fail"
    preferred = evaluate_gates(summary, {
        "max_completion_after_stop_p95_seconds": 0.4,
        "max_final_p95_seconds": 0.8,
    })
    assert preferred["checks"]["completion_after_stop_p95_seconds"]["limit"] == 0.4

    config = {"thresholds": {
        "max_completion_after_stop_p95_seconds": 0.4,
        "max_final_p95_seconds": 0.4,
    }}
    report_json, _ = write_report(tmp_path, config=config, results=[], resource_samples=[])
    report_config = json.loads(report_json.read_text(encoding="utf-8"))["config"]
    assert report_config["thresholds"]["max_completion_after_stop_p95_seconds"] == 0.4
    assert report_config["thresholds"]["max_final_p95_seconds"] == 0.4


def test_file_throughput_uses_successful_samples_and_seconds_ratio(tmp_path):
    results = [
        SessionResult("success", "file", completed=True, audio_seconds=3600, processing_seconds=3600),
        SessionResult("timeout", "file", audio_seconds=36000, error="TimeoutError: timeout"),
        SessionResult("missing-time", "file", completed=True, audio_seconds=1800),
    ]
    report_json, _ = write_report(
        tmp_path,
        config={"wall_elapsed_seconds": 3600, "thresholds": {}},
        results=results,
        resource_samples=[],
    )
    report = json.loads(report_json.read_text(encoding="utf-8"))
    summary = report["summary"]
    # Processing-hour uses one measured hour; wall-hour includes both successful jobs.
    assert summary["file_audio_hours_per_processing_hour"] == 1
    assert summary["file_audio_hours_per_wall_hour"] == 1.5
    assert summary["processing_seconds"] == 3600
    assert summary["file_processing_sample_count"] == 1
    assert summary["file_audio_seconds"] == 5400  # successful jobs only
    assert summary["file_attempted_audio_seconds"] == 41400


def test_file_job_rtf_is_computed_with_audio_duration(monkeypatch, tmp_path):
    captured = {}

    class Response:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self.body

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **kwargs):
            captured["post"] = kwargs
            return Response({"job_id": "fake-job"})

        async def get(self, *_args, **_kwargs):
            return Response({"status": "succeeded", "segments": [{"text": "旧文本", "local_speaker_id": "speaker_0"}],
                             "effective_segments": [{"text": "新文本", "speaker": {"display_name": "示例甲"}}],
                             "created_at": "2026-10-23T00:00:00+00:00",
                             "started_at": "2026-10-23T00:00:00.250+00:00",
                             "finished_at": "2026-10-23T00:00:01.250+00:00"})

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(AsyncClient=FakeClient))
    wav_path = tmp_path / "input.wav"
    wav_path.write_bytes(b"fake input")
    result = asyncio.run(run_file_job(
        path=wav_path,
        audio_seconds=2.0,
        base_url="http://asr.invalid",
        run_id="fake-file",
        hotword_set_id=None,
        token=None,
        poll_seconds=0.001,
        timeout_seconds=1,
    ))
    assert result.completed
    assert result.audio_seconds == 2.0
    assert result.rtf is not None and result.rtf > 0
    assert result.queue_seconds == 0.25
    assert result.processing_seconds == 1.0
    assert result.final_text == "新文本"
    assert result.speaker == "示例甲"
    assert captured["client"]["trust_env"] is False
    assert captured["post"]["data"]["channel_mode"] == "mixdown"


def test_report_has_metadata_summary_sessions_and_csv(tmp_path):
    item = SessionResult(
        "case-1", "realtime", session_id="s1", completed=True, final_text="你好", frames_sent=42
    )
    json_path, csv_path = write_report(
        tmp_path,
        config={"mode": "realtime", "thresholds": {"max_error_rate": 0.01}},
        results=[item],
        resource_samples=[],
        hardware={"system": "test-machine"},
    )
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["hardware"]["system"] == "test-machine"
    assert report["summary"]["success_count"] == 1
    assert report["sessions"][0]["final_text"] == "你好"
    assert report["capacity_claim"] is None
    csv_text = csv_path.read_text(encoding="utf-8-sig")
    assert "frames_sent" in csv_text
    assert "42" in csv_text
