import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from loadtest.core import SessionResult, evaluate_gates, pace_pcm, percentile, summarize, write_report
from loadtest.runner import load_wav, run_file_job, write_repeated_wav


def test_percentile_uses_linear_interpolation_and_empty_values():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)
    assert percentile([], 95) is None


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
    class Response:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self.body

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            return Response({"job_id": "fake-job"})

        async def get(self, *_args, **_kwargs):
            return Response({"status": "succeeded", "segments": []})

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
