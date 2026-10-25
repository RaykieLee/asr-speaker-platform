#!/usr/bin/env python3
"""Export a time-bounded AISHELL-4 TextGrid interval as mono 16 kHz PCM WAV and annotation JSON."""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any


def clean_text(text: str) -> str:
    text = re.sub(r"<\s*(?:sil|#)\s*>", "", text, flags=re.IGNORECASE)
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("P")
                   and not ch.isspace())


def _resample_sinc(samples: Any, source_rate: int, target_rate: int = 16000) -> Any:
    """Small windowed-sinc resampler, using NumPy only (anti-aliases downsampling)."""
    import numpy as np

    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float64)
    ratio = source_rate / target_rate
    count = int(round(len(samples) / ratio))
    radius = 24
    cutoff = min(1.0, target_rate / source_rate)
    result = np.empty(count, dtype=np.float64)
    offsets = np.arange(-radius + 1, radius + 1)
    for first in range(0, count, 8192):
        positions = np.arange(first, min(first + 8192, count)) * ratio
        centers = np.floor(positions).astype(np.int64)
        indices = centers[:, None] + offsets[None, :]
        delta = positions[:, None] - indices
        window = 0.5 + 0.5 * np.cos(np.pi * delta / radius)
        window[np.abs(delta) >= radius] = 0
        weights = cutoff * np.sinc(cutoff * delta) * window
        valid = (indices >= 0) & (indices < len(samples))
        weights *= valid
        weights /= weights.sum(axis=1, keepdims=True)
        clipped = np.clip(indices, 0, max(0, len(samples) - 1))
        result[first:first + len(positions)] = (samples[clipped] * weights).sum(axis=1)
    return result


def _read_intervals(path: Path) -> list[tuple[str, float, float, str]]:
    try:
        from textgrid import TextGrid
    except ImportError as exc:
        raise RuntimeError("TextGrid parsing requires optional package 'textgrid' (uv add --dev textgrid)") from exc
    try:
        grid = TextGrid.fromFile(str(path))
    except Exception as exc:
        raise RuntimeError(f"cannot parse TextGrid {path}: {exc}") from exc
    rows = []
    for tier in grid:
        for interval in tier:
            text = clean_text(interval.mark)
            if text:
                rows.append((str(tier.name), float(interval.minTime), float(interval.maxTime), text))
    return rows


def prepare(audio: Path, textgrid_path: Path, start: float, end: float, wav_path: Path,
            annotation_path: Path, hotwords: list[str], channel: int | str = 0) -> dict[str, Any]:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("FLAC/WAV conversion requires optional package 'soundfile' (uv add --dev soundfile)") from exc
    if start < 0 or end <= start:
        raise ValueError("require 0 <= start < end")
    info = sf.info(str(audio))
    if end > info.duration:
        raise ValueError(f"requested end {end:g}s exceeds audio duration {info.duration:g}s")
    first = int(round(start * info.samplerate))
    frames = int(round((end - start) * info.samplerate))
    with sf.SoundFile(str(audio)) as source:
        source.seek(first)
        data = source.read(frames, dtype="float64", always_2d=True)
    if not len(data):
        raise ValueError("selected audio interval is empty")
    if channel == "mixdown":
        mono = data.mean(axis=1)
    elif isinstance(channel, int) and 0 <= channel < info.channels:
        mono = data[:, channel]
    else:
        raise ValueError(f"channel must be an index from 0 to {info.channels - 1}, or 'mixdown'")
    pcm = _resample_sinc(mono, info.samplerate)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_path), np.clip(pcm, -1, 1), 16000, subtype="PCM_16", format="WAV")

    segments = []
    for speaker, seg_start, seg_end, text in _read_intervals(textgrid_path):
        left, right = max(start, seg_start), min(end, seg_end)
        if right <= left:
            continue
        clipped = clean_text(text)
        if not clipped:
            continue
        segments.append({"start": round(left - start, 4), "end": round(right - start, 4),
                         "text": clipped, "lang": "zh", "speaker_id": speaker})
    segments.sort(key=lambda item: (item["start"], item["end"], item["speaker_id"]))
    annotation = {"segments": segments, "hotword_targets": hotwords,
                  "source": {"dataset": "AISHELL-4", "audio_file": audio.name,
                             "textgrid_file": textgrid_path.name,
                             "source_start_seconds": start, "source_end_seconds": end,
                             "source_channel": channel}}
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"wav": str(wav_path), "annotation": str(annotation_path), "segments": len(segments),
            "duration_seconds": len(pcm) / 16000, "sample_rate": 16000, "channels": 1,
            "subtype": "PCM_16", "source_channel": channel}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path, help="local AISHELL-4 FLAC")
    parser.add_argument("--textgrid", required=True, type=Path)
    parser.add_argument("--start", required=True, type=float, help="source start time in seconds")
    parser.add_argument("--end", required=True, type=float, help="source end time in seconds")
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--hotwords", default="", help="comma-separated terms, manually chosen for the test")
    parser.add_argument("--channel", default="0",
                        help="source channel index (default: 0), or 'mixdown' to average all channels")
    args = parser.parse_args(argv)
    try:
        channel: int | str = "mixdown" if args.channel == "mixdown" else int(args.channel)
        result = prepare(args.audio, args.textgrid, args.start, args.end, args.wav,
                         args.annotation, [clean_text(x) for x in args.hotwords.split(",") if clean_text(x)], channel)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"prepare error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
