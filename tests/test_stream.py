import threading

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.stream import router
from backend.app.pipelines.live_asr import LiveASRService
from backend.app.main import app as production_app


class FakeStream:
    def __init__(self, hotwords=None):
        self.hotwords = hotwords
        self.pending = False
        self.finished = False
        self.text = ""
        self.samples = 0

    def accept_waveform(self, sample_rate, samples):
        self.samples += len(samples)
        self.pending = True
        self.text = "sampled"

    def input_finished(self):
        self.finished = True
        self.pending = True


class FakeRecognizer:
    def __init__(self, *, block_feed=False):
        self.streams = []
        self.block_feed = block_feed
        self.feed_entered = threading.Event()
        self.release_feed = threading.Event()

    def create_stream(self, hotwords=None):
        stream = FakeStream(hotwords)
        self.streams.append(stream)
        return stream

    def is_ready(self, stream):
        return stream.pending

    def decode_stream(self, stream):
        if self.block_feed and not stream.finished:
            self.feed_entered.set()
            self.release_feed.wait(timeout=1)
        stream.pending = False

    def get_result(self, stream):
        return stream.text

    def is_endpoint(self, stream):
        return False


def client_for(recognizer, *, queue_size=8, hotword_resolver=None):
    app = FastAPI()
    app.include_router(router)
    app.state.live_asr_service = LiveASRService(
        recognizer, queue_size=queue_size, hotword_resolver=hotword_resolver
    )
    return TestClient(app)


def start_message(**kwargs):
    return {"type": "start", "protocol_version": 1, "sample_rate": 16000,
            "format": "s16le", "channels": 1, **kwargs}


def test_protocol_rejects_bad_initial_message():
    with client_for(FakeRecognizer()).websocket_connect("/api/v1/stream") as ws:
        ws.send_json({"type": "start", "sample_rate": 8000})
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "protocol_error"


def test_odd_pcm_payload_is_rejected_explicitly():
    with client_for(FakeRecognizer()).websocket_connect("/api/v1/stream") as ws:
        ws.send_json(start_message())
        ws.receive_json()
        ws.send_bytes(b"\x01")
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "invalid_audio"


def test_missing_model_service_does_not_return_fake_transcript():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app).websocket_connect("/api/v1/stream") as ws:
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "model_unavailable"
        assert "模型" in error["message"]


def test_production_app_registers_stream_route():
    with TestClient(production_app).websocket_connect("/api/v1/stream") as ws:
        error = ws.receive_json()
        assert error["code"] == "model_unavailable"


def test_stop_flushes_trailing_audio_and_emits_design_fields():
    with client_for(FakeRecognizer()).websocket_connect("/api/v1/stream") as ws:
        ws.send_json(start_message(hotwords=["李子熙"]))
        started = ws.receive_json()
        assert started["type"] == "started"
        assert started["protocol_version"] == 1
        ws.send_bytes(b"\x01\x00" * 1600)
        partial = ws.receive_json()
        assert partial["status"] == "partial"
        ws.send_json({"type": "stop"})
        final = ws.receive_json()
        completed = ws.receive_json()
        assert final["status"] == "final"
        assert final["session_id"] == started["session_id"]
        assert final["segment_id"] == 1
        assert final["revision"] > partial["revision"]
        assert final["start"] == 0
        assert final["end"] == 0.1
        assert {"local_speaker_id", "registered_speaker_id", "speaker_name", "overlap"} <= final.keys()
        assert completed["type"] == "completed"
        assert completed["duration"] == 0.1
        assert completed["dropped_frames"] == 0


def test_connections_get_independent_recognizer_streams():
    recognizer = FakeRecognizer()
    with client_for(recognizer).websocket_connect("/api/v1/stream") as first:
        first.send_json(start_message(hotwords=["甲"]))
        started_a = first.receive_json()
        first.send_bytes(b"\x01\x00" * 10)
        first.receive_json()
        with client_for(recognizer).websocket_connect("/api/v1/stream") as second:
            second.send_json(start_message(hotwords=["乙"]))
            started_b = second.receive_json()
            second.send_bytes(b"\x01\x00" * 20)
            second.receive_json()
            assert started_a["session_id"] != started_b["session_id"]
            assert len(recognizer.streams) == 2
            assert recognizer.streams[0] is not recognizer.streams[1]
            assert recognizer.streams[0].hotwords == ["甲"]
            assert recognizer.streams[1].hotwords == ["乙"]
            second.send_json({"type": "stop"})
            second.receive_json()
            second.receive_json()
        first.send_json({"type": "stop"})
        first.receive_json()
        first.receive_json()


def test_queue_overflow_reports_busy_and_disconnect_is_clean():
    recognizer = FakeRecognizer(block_feed=True)
    with client_for(recognizer, queue_size=1).websocket_connect("/api/v1/stream") as ws:
        ws.send_json(start_message())
        ws.receive_json()
        ws.send_bytes(b"\x01\x00" * 100)
        assert recognizer.feed_entered.wait(timeout=1)
        ws.send_bytes(b"\x01\x00" * 100)
        ws.send_bytes(b"\x01\x00" * 100)
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "queue_full"
        recognizer.release_feed.set()

    # A client that disconnects without stop is discarded without a final event.
    with client_for(FakeRecognizer()).websocket_connect("/api/v1/stream") as ws:
        ws.send_json(start_message())
        ws.receive_json()
        ws.send_bytes(b"\x01\x00" * 100)
        ws.receive_json()


def test_async_hotword_resolver_is_awaited_and_pinned():
    async def resolve(set_id):
        assert set_id == "set_1"
        return ["项 目"], 3

    recognizer = FakeRecognizer()
    with client_for(recognizer, hotword_resolver=resolve).websocket_connect(
        "/api/v1/stream"
    ) as ws:
        ws.send_json(start_message(hotword_set_id="set_1"))
        started = ws.receive_json()
        assert started["type"] == "started"
        assert started["hotword_set_version"] == 3
        assert recognizer.streams[0].hotwords == ["项 目"]
        ws.send_json({"type": "stop"})
        completed = ws.receive_json()
        assert completed["type"] == "completed"
        assert completed["dropped_frames"] == 0


def test_hotword_resolver_failure_is_structured_error():
    async def resolve(_set_id):
        raise LookupError("热词集不存在")

    with client_for(FakeRecognizer(), hotword_resolver=resolve).websocket_connect(
        "/api/v1/stream"
    ) as ws:
        ws.send_json(start_message(hotword_set_id="missing"))
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "model_unavailable"
        assert "热词集不存在" in error["message"]


def test_inference_failure_is_structured_error():
    class FailingRecognizer(FakeRecognizer):
        def decode_stream(self, stream):
            raise RuntimeError("推理失败")

    with client_for(FailingRecognizer()).websocket_connect("/api/v1/stream") as ws:
        ws.send_json(start_message())
        ws.receive_json()
        ws.send_bytes(b"\x01\x00" * 100)
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "recognition_error"
        assert "推理失败" in error["message"]
