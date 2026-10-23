import array

from backend.app.audio import AudioBuffer
from backend.app.pipelines.live_asr import LiveASRSession, StartOptions
from backend.app.speakers.live_diarization import (
    LiveSpeakerTracker,
    SpeakerIdentity,
    profile_cosine_matcher,
)


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
    def __init__(self, hotwords=None):
        self.hotwords = hotwords
        self.samples = array.array("f")
        self.pending = False
        self.finished = False

    def accept_waveform(self, sample_rate, samples):
        self.samples.extend(samples)
        self.pending = True

    def input_finished(self):
        self.finished = True
        self.pending = True


class FakeRecognizer:
    def __init__(self):
        self.streams = []

    def create_stream(self, hotwords=None):
        stream = FakeASRStream(hotwords)
        self.streams.append(stream)
        return stream

    def is_ready(self, stream):
        return stream.pending

    def decode_stream(self, stream):
        stream.pending = False

    def get_result(self, stream):
        if not stream.samples:
            return ""
        return "甲的识别文本" if sum(stream.samples) >= 0 else "乙的识别文本"

    def is_endpoint(self, stream):
        return False


def tracker(**kwargs):
    return LiveSpeakerTracker(
        vad=kwargs.get("vad", FakeVAD()), extractor=kwargs.get("extractor", FakeExtractor()),
        matcher=kwargs.get("matcher", fake_match),
        profiles=[{"id": "registered_a", "name": "甲", "model_version": "fake-v1"},
                  {"id": "registered_b", "name": "乙", "model_version": "fake-v1"}],
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


def test_old_model_profile_is_not_used_for_live_identity():
    class VersionedExtractor(FakeExtractor):
        model_version = "model-v2"

    state = LiveSpeakerTracker(
        vad=FakeVAD(), extractor=VersionedExtractor(),
        matcher=profile_cosine_matcher(min_similarity=0.8, min_margin=0.05),
        profiles=[{"id": "old-profile", "name": "旧版本用户",
                   "embedding": [1.0, 0.0], "model_version": "model-v1"}],
        window_seconds=0.5, step_seconds=0.25, min_speech_seconds=0.25,
        min_dwell_seconds=0.25, confirmations=1,
    )

    state.feed(array.array("f", [0.2] * 8_000), 8_000)

    assert state.current.registered_speaker_id is None
    assert state.current.speaker_name is None


def test_profile_matcher_rejects_embedding_dimension_mismatch():
    matcher = profile_cosine_matcher(min_similarity=0.8, min_margin=0.05)

    result = matcher([1.0, 0.0], [{"id": "wrong-dimension", "name": "不应匹配",
                                   "embedding": [1.0, 0.0, 0.0]}])

    assert result is None


def test_local_unknown_centroids_are_bounded_and_existing_ids_stay_stable():
    class DistinctExtractor:
        model_version = "fake-v1"

        def __init__(self):
            self.index = 0

        def extract(self, audio):
            vector = [0.0] * 256
            vector[self.index] = 1.0
            self.index += 1
            return vector

    extractor = DistinctExtractor()
    state = LiveSpeakerTracker(
        vad=FakeVAD(), extractor=extractor, matcher=lambda _vector, _rows: None,
        max_local_speakers=4,
    )

    identities = [state._identify([0.1] * 8_000) for _ in range(200)]

    assert len(state._local_vectors) == 4
    assert [identity.local_speaker_id for identity in identities[:4]] == [
        "speaker_0", "speaker_1", "speaker_2", "speaker_3",
    ]
    assert all(identity == SpeakerIdentity() for identity in identities[4:])
    assert state.unknown_capacity_exhaustions == 196
    extractor.index = 0
    assert state._identify([0.1] * 8_000) == SpeakerIdentity(local_speaker_id="speaker_0")
    assert len(state._local_vectors) == 4
    assert state.unknown_capacity_exhaustions == 196


def test_capacity_exhaustion_moves_registered_current_to_unknown():
    class NovelExtractor:
        model_version = "fake-v1"

        def __init__(self):
            self.index = 0

        def extract(self, audio):
            self.index += 1
            return [0.0, 1.0] if self.index == 1 else [0.0, -1.0]

    extractor = NovelExtractor()
    state = LiveSpeakerTracker(
        vad=FakeVAD(), extractor=extractor, matcher=lambda _vector, _rows: None,
        profiles=(), sample_rate=16_000, window_seconds=0.25,
        step_seconds=0.25, min_speech_seconds=0.25,
        min_dwell_seconds=0.25, confirmations=1, max_local_speakers=1,
    )
    # One unknown centroid consumes the budget; the next unmatched voice must
    # be an explicit empty identity, not None (which would retain this label).
    assert state._identify([0.1] * 4_000) == SpeakerIdentity(local_speaker_id="speaker_0")
    state.current = SpeakerIdentity("speaker_0", "registered_a", "甲")

    changes = state.feed([0.1] * 4_000, 4_000)

    assert len(changes) == 1
    assert changes[0]["previous"]["registered_speaker_id"] == "registered_a"
    assert changes[0]["speaker"] == {
        "local_speaker_id": None,
        "registered_speaker_id": None,
        "speaker_name": None,
    }
    assert state.current == SpeakerIdentity()
    assert len(state._local_vectors) == 1


def test_confirmed_switch_finalizes_old_segment_and_starts_non_overlapping_segment():
    live = LiveASRSession(FakeRecognizer(), options=StartOptions(), speaker_tracker=tracker())
    first = live.feed((b"\xff\x7f") * 32_000)
    assert first[0]["status"] == "partial"
    original_id = first[0]["segment_id"]
    later_audio = (b"\x01\x80") * 48_000
    switched = live.feed(later_audio)
    final_old = next(event for event in switched if event["status"] == "final")
    partial_new = next(event for event in switched if event["status"] == "partial")

    assert final_old["segment_id"] == original_id
    assert final_old["revision"] > first[0]["revision"]
    assert final_old["text"] == "甲的识别文本"
    assert final_old["registered_speaker_id"] == "registered_a"
    assert partial_new["segment_id"] == original_id + 1
    assert partial_new["start"] == final_old["end"]
    assert partial_new["text"] == "乙的识别文本"
    assert partial_new["registered_speaker_id"] == "registered_b"
    assert all(event["status"] != "corrected" for event in switched)

    tail = live.finish()
    final_new = next(event for event in tail if event["status"] == "final")
    assert final_new["segment_id"] == partial_new["segment_id"]
    assert final_new["text"] == "乙的识别文本"
    assert final_new["start"] == final_old["end"]
    assert final_new["registered_speaker_id"] == "registered_b"


def test_speaker_change_does_not_create_text_when_recognizer_returns_empty():
    class EmptyRecognizer(FakeRecognizer):
        def get_result(self, stream):
            return ""

    live = LiveASRSession(EmptyRecognizer(), options=StartOptions(), speaker_tracker=tracker())
    assert live.feed((b"\xff\x7f") * 32_000) == []
    assert live.feed((b"\x01\x80") * 48_000) == []
    assert live.finish() == []


def test_long_no_endpoint_audio_rolls_over_without_loss_or_overlap():
    recognizer = FakeRecognizer()
    state = LiveSpeakerTracker()
    state.current = SpeakerIdentity("speaker_a", "registered_a", "甲")
    live = LiveASRSession(
        recognizer, options=StartOptions(hotwords=("关键字",)), speaker_tracker=state,
        max_segment_seconds=1.0,
    )
    total_samples = 40_000  # 2.5 seconds, sent as one frame across two rollovers.
    audio = (b"\xff\x7f") * total_samples
    events = live.feed(audio)
    partials = [event for event in events if event["status"] == "partial"]
    finals = [event for event in events if event["status"] == "final"]
    finals.extend(live.finish())
    finals = [event for event in finals if event["status"] == "final"]

    assert [(event["segment_id"], event["start"], event["end"]) for event in finals] == [
        (1, 0.0, 1.0), (2, 1.0, 2.0), (3, 2.0, 2.5),
    ]
    assert len({event["segment_id"] for event in finals}) == len(finals)
    assert all(event["registered_speaker_id"] == "registered_a" for event in finals)
    assert all(event["speaker_name"] == "甲" for event in finals)
    for final in finals[:2]:
        partial = next(event for event in partials if event["segment_id"] == final["segment_id"])
        assert final["revision"] > partial["revision"]
    assert sum(len(stream.samples) for stream in recognizer.streams) == total_samples
    assert all(stream.hotwords == ["关键字"] for stream in recognizer.streams)


def test_segment_cap_splits_exact_boundary_without_empty_segment():
    live = LiveASRSession(FakeRecognizer(), speaker_tracker=LiveSpeakerTracker(),
                          max_segment_seconds=1.0)

    events = live.feed((b"\xff\x7f") * 16_000)

    assert [event["status"] for event in events] == ["partial", "final"]
    assert events[-1]["start"] == 0.0
    assert events[-1]["end"] == 1.0
    assert events[-1]["segment_id"] == events[0]["segment_id"]
    assert live.segment_id == 2
    assert len(live.segment_samples) == 0


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
