import array

from backend.app.audio import AudioBuffer
from backend.app.pipelines.live_asr import LiveASRSession, StartOptions
from backend.app.speakers.live_diarization import LiveSpeakerTracker


class FakeVAD:
    def __init__(self):
        self.calls = 0

    def is_speech(self, samples, sample_rate):
        self.calls += 1
        return True


class FakeExtractor:
    model_version = "fake-v1"

    def __init__(self):
        self.calls = 0

    def extract(self, audio: AudioBuffer):
        self.calls += 1
        # Test audio is a clear +1/-1 step; this is a deterministic test fixture,
        # not an attempt to approximate a production speaker model.
        mean = sum(audio.samples) / max(1, len(audio.samples))
        return [1.0, 0.0] if mean >= 0 else [0.0, 1.0]


def fake_match(embedding, profiles):
    identity = "registered_a" if embedding[0] > embedding[1] else "registered_b"
    row = next((row for row in profiles if row["id"] == identity), None)
    return ({"local_speaker_id": "speaker_0", "registered_speaker_id": identity,
             "speaker_name": row["name"]} if row else None)


class FakeASRStream:
    def __init__(self):
        self.text = ""
        self.pending = False

    def accept_waveform(self, sample_rate, samples):
        self.text = "实时识别文本"
        self.pending = True

    def input_finished(self):
        self.pending = True


class FakeRecognizer:
    def create_stream(self, hotwords=None):
        return FakeASRStream()

    def is_ready(self, stream):
        return stream.pending

    def decode_stream(self, stream):
        stream.pending = False

    def get_result(self, stream):
        return stream.text

    def is_endpoint(self, stream):
        return False


def tracker(**kwargs):
    return LiveSpeakerTracker(
        vad=kwargs.get("vad", FakeVAD()), extractor=kwargs.get("extractor", FakeExtractor()),
        matcher=kwargs.get("matcher", fake_match),
        profiles=[{"id": "registered_a", "name": "甲"}, {"id": "registered_b", "name": "乙"}],
        window_seconds=1, step_seconds=0.5, min_speech_seconds=0.5,
        min_dwell_seconds=0.5, confirmations=2,
    )


def test_continuous_speech_switch_uses_hysteresis_and_stable_mapping():
    vad, extractor = FakeVAD(), FakeExtractor()
    state = tracker(vad=vad, extractor=extractor)
    audio = array.array("f", [1.0] * 32_000 + [-1.0] * 32_000)
    changes = state.feed(audio, len(audio))

    second_speaker = [change for change in changes
                      if change["speaker"]["registered_speaker_id"] == "registered_b"]
    assert len(second_speaker) == 1
    assert second_speaker[0]["speaker"]["speaker_name"] == "乙"
    assert second_speaker[0]["start"] >= 1.5
    assert vad.calls >= 4
    assert extractor.calls >= 4


def test_short_utterance_stays_unknown_without_embedding():
    extractor = FakeExtractor()
    state = tracker(extractor=extractor)
    changes = state.feed(array.array("f", [0.5] * 4_000), 4_000)

    assert changes == []
    assert extractor.calls == 0
    assert state.current.registered_speaker_id is None
    assert state.current.speaker_name is None


def test_missing_speaker_models_never_invent_registered_identity():
    state = LiveSpeakerTracker(vad=None, extractor=None, matcher=fake_match,
                               window_seconds=0.5, step_seconds=0.25,
                               min_speech_seconds=0.25)
    assert state.feed(array.array("f", [0.5] * 16_000), 16_000) == []
    assert state.current.registered_speaker_id is None
    assert state.current.speaker_name is None


def test_confirmed_switch_emits_higher_revision_for_same_segment_id():
    live = LiveASRSession(FakeRecognizer(), options=StartOptions(), speaker_tracker=tracker())
    first = live.feed((b"\xff\x7f") * 32_000)
    assert first[0]["status"] == "partial"
    original_id = first[0]["segment_id"]
    later_audio = (b"\x01\x80") * 48_000
    revised = live.feed(later_audio)
    correction = next(event for event in revised if event.get("status") == "corrected")

    assert correction["segment_id"] == original_id
    assert correction["revision"] > first[0]["revision"]
    assert correction["registered_speaker_id"] == "registered_b"
    assert correction["text"] == first[0]["text"]


def test_two_trackers_keep_independent_speaker_state():
    first, second = tracker(), tracker()
    samples_a = array.array("f", [1.0] * 24_000)
    samples_b = array.array("f", [-1.0] * 24_000)
    first.feed(samples_a, len(samples_a))
    second.feed(samples_b, len(samples_b))

    assert first is not second
    assert first.current.registered_speaker_id == "registered_a"
    assert second.current.registered_speaker_id == "registered_b"
    assert first.current.speaker_name == "甲"
    assert second.current.speaker_name == "乙"
