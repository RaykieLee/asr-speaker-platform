import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from loadtest.core import (SessionResult, aggregate_final_speaker_events, evaluate_gates,
                           pace_pcm, percentile, summarize, write_report)
from loadtest.runner import build_parser, load_wav, run_file_job, run_live_session, write_repeated_wav


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
