import array
import io
import wave

import pytest

from backend.app.audio.core import (
    AudioBuffer, AudioFormatError, PCMFormatError, decode_wav, mixdown,
    normalize_audio, pcm_s16le_to_float32, separate_stereo,
)


def make_wav(samples: list[int], *, rate: int = 24000, channels: int = 1, width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(width)
        f.setframerate(rate)
        if width == 2:
            raw = array.array("h", samples)
            if __import__("sys").byteorder != "little":
                raw.byteswap()
            f.writeframes(raw.tobytes())
        else:
            f.writeframes(bytes(samples))
    return buf.getvalue()


def test_pcm_s16le_conversion_and_silence():
    converted = pcm_s16le_to_float32(b"\x00\x80\x00\x00\xff\x7f")
    assert converted.typecode == "f"
    assert list(converted) == [-1.0, 0.0, pytest.approx(32767 / 32768)]
    assert list(pcm_s16le_to_float32(b"\x00\x00" * 320)) == [0.0] * 320


def test_pcm_rejects_partial_sample_and_wrong_stream_shape():
    with pytest.raises(PCMFormatError, match="even"):
        pcm_s16le_to_float32(b"\x00")
    with pytest.raises(PCMFormatError, match="16 kHz"):
        pcm_s16le_to_float32(b"\x00\x00", sample_rate=8000)


def test_wav_decode_stereo_mix_and_separate():
    decoded = decode_wav(make_wav([32767, -32768, 0, 16384], rate=16000, channels=2))
    assert decoded.channels == 2
    left, right = separate_stereo(decoded)
    assert left.frame_count == right.frame_count == 2
    assert list(left.samples) == pytest.approx([32767 / 32768, 0.0])
    assert list(right.samples) == pytest.approx([-1.0, 0.5])
    mono = mixdown(decoded)
    assert mono.channels == 1
    assert list(mono.samples) == pytest.approx([(32767 / 32768 - 1) / 2, 0.25])


def test_wav_normalization_preserves_source_duration_and_timestamps():
    source = decode_wav(make_wav([0] * 1001, rate=24000))
    normalized = normalize_audio(source)
    assert normalized.sample_rate == 16000
    assert normalized.channels == 1
    assert normalized.frame_count == round(1001 * 16000 / 24000)
    assert normalized.duration == pytest.approx(1001 / 24000)
    assert normalized.timestamp_for_frame(0) == 0
    assert normalized.timestamp_for_frame(normalized.frame_count - 1) == pytest.approx(666 / 16000)
    assert normalized.timestamp_for_frame(normalized.frame_count) == pytest.approx(1001 / 24000)
    with pytest.raises(IndexError):
        normalized.timestamp_for_frame(normalized.frame_count + 1)


def test_wav_tail_is_retained_after_resampling():
    # Non-silent final source sample must survive as a final output sample.
    source = decode_wav(make_wav([0] * 99 + [20000], rate=24000))
    normalized = normalize_audio(source)
    assert normalized.frame_count == 67
    assert normalized.samples[-1] == pytest.approx(20000 / 32768)
    assert normalized.duration == pytest.approx(100 / 24000)


def test_wav_rejects_invalid_or_truncated_data():
    with pytest.raises(AudioFormatError):
        decode_wav(b"not a wav")
    good = make_wav([1, 2, 3])
    with pytest.raises(AudioFormatError):
        decode_wav(good[:-2])


def test_empty_input_has_zero_duration():
    empty = AudioBuffer(array.array("f"), 16000, 1)
    assert normalize_audio(empty).duration == 0
    assert normalize_audio(empty).frame_count == 0
