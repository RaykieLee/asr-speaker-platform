import array
import time

import pytest

from backend.app.pipelines.live_asr import FinalAudio
from backend.app.pipelines.live_final import LiveFinalProcess, LiveFinalQueueFull


def _fake_process_worker(requests, responses, cancellations, config):
    responses.put(("__ready__", True, ""))
    cancelled_sessions = set()
    while True:
        while True:
            try:
                cancelled_sessions.add(cancellations.get_nowait())
            except Exception:
                break
        item = requests.get()
        if item is None:
            return
        request_id, session_id, pcm, hotwords = item
        if session_id in cancelled_sessions:
            responses.put((request_id, False, "session cancelled"))
            continue
        time.sleep(config.get("delay", 0.2))
        if config.get("fail"):
            responses.put((request_id, False, "offline recognition failed"))
        else:
            responses.put((request_id, True, f"{len(pcm)}:{','.join(hotwords)}"))


def _segment(segment_id=1):
    return FinalAudio("session-1", segment_id, 1, 0.0, 1.0, array.array("f", [0.25] * 16),
                      ("offline-token",), {"local_speaker_id": "speaker_0"})


def test_process_uses_bounded_queue_and_resolves_offline_result():
    worker = LiveFinalProcess({"delay": 0.3}, queue_size=1,
                              startup_timeout=5, _worker_target=_fake_process_worker)
    try:
        first = worker.submit(_segment(1))
        time.sleep(0.05)  # Let the child take the first request.
        second = worker.submit(_segment(2))
        with pytest.raises(LiveFinalQueueFull):
            worker.submit(_segment(3))
        assert first.result(timeout=3) == "64:offline-token"
        assert second.result(timeout=3) == "64:offline-token"
    finally:
        worker.close()


def test_cancel_session_skips_queued_audio_after_current_inference():
    worker = LiveFinalProcess({"delay": 0.3}, queue_size=2,
                              startup_timeout=5, _worker_target=_fake_process_worker)
    try:
        current = worker.submit(_segment())
        time.sleep(0.05)
        queued_segment = _segment(2)
        # Model a second socket that disconnected while its audio waited behind
        # the current inference.
        queued_segment = FinalAudio("disconnected-session", queued_segment.segment_id,
                                    queued_segment.revision, queued_segment.start,
                                    queued_segment.end, queued_segment.samples,
                                    queued_segment.hotwords, queued_segment.speaker)
        queued = worker.submit(queued_segment)
        worker.cancel_session("disconnected-session")
        assert current.result(timeout=3) == "64:offline-token"
        with pytest.raises(RuntimeError, match="session cancelled"):
            queued.result(timeout=3)
    finally:
        worker.close()


def test_process_close_fails_pending_futures_without_leaking_details():
    worker = LiveFinalProcess({"delay": 2.0}, queue_size=1,
                              startup_timeout=5, _worker_target=_fake_process_worker)
    pending = worker.submit(_segment())
    worker.close()
    with pytest.raises(RuntimeError, match="进程已退出|服务已关闭"):
        pending.result(timeout=1)


def test_subprocess_recognition_error_is_sanitized():
    worker = LiveFinalProcess({"fail": True}, startup_timeout=5,
                              _worker_target=_fake_process_worker)
    try:
        with pytest.raises(RuntimeError, match="offline recognition failed"):
            worker.submit(_segment()).result(timeout=3)
    finally:
        worker.close()
