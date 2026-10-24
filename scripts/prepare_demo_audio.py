#!/usr/bin/env python3
"""Build the small, attributed AISHELL-3 Mandarin demo audio set offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
import wave
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_DIR = PROJECT_ROOT / "demo_assets" / "aishell3"
_SAFE_ID = re.compile(r"^[A-Z0-9]+$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def transcript_text(source_pinyin_line: str) -> str:
    """Extract original Chinese character tokens, dropping alternating pinyin."""
    tokens = source_pinyin_line.strip().split()
    if not tokens or len(tokens) % 2:
        raise ValueError("invalid AISHELL-3 character/pinyin transcript line")
    characters = tokens[::2]
    if any(not token for token in characters):
        raise ValueError("empty transcript token")
    return "".join(characters)


def _read_pcm(path: Path) -> tuple[wave._wave_params, bytes]:
    try:
        with wave.open(str(path), "rb") as source:
            params = source.getparams()
            frames = source.readframes(params.nframes)
    except (OSError, wave.Error) as exc:
        raise ValueError("source audio is not a readable PCM WAV") from exc
    if params.comptype != "NONE" or params.nchannels != 1 or params.sampwidth not in (1, 2, 3, 4):
        raise ValueError("source clips must be uncompressed mono PCM WAV")
    if not params.framerate or params.nframes <= 0 or len(frames) != params.nframes * params.nchannels * params.sampwidth:
        raise ValueError("source WAV has invalid frame metadata")
    return params, frames


def concatenate_wavs(
    clips: list[tuple[str, Path]], output_path: Path, silence_ms: int
) -> list[dict[str, Any]]:
    """Concatenate complete WAV clips with digital silence; never resample/crop."""
    if not clips or not isinstance(silence_ms, int) or silence_ms < 0:
        raise ValueError("at least one clip and non-negative integer silence are required")
    decoded = [(clip_id, path, *_read_pcm(path)) for clip_id, path in clips]
    params = decoded[0][2]
    if any(
        (item[2].nchannels, item[2].sampwidth, item[2].framerate, item[2].comptype)
        != (params.nchannels, params.sampwidth, params.framerate, params.comptype)
        for item in decoded
    ):
        raise ValueError("all source WAVs must have identical PCM parameters")
    silence_frames = params.framerate * silence_ms // 1000
    silence = b"\0" * (silence_frames * params.nchannels * params.sampwidth)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cursor = 0
    segments = []
    with wave.open(str(output_path), "wb") as target:
        target.setnchannels(params.nchannels)
        target.setsampwidth(params.sampwidth)
        target.setframerate(params.framerate)
        for index, (clip_id, _path, clip_params, frames) in enumerate(decoded):
            duration = clip_params.nframes / params.framerate
            start = cursor / params.framerate
            target.writeframesraw(frames)
            cursor += clip_params.nframes
            end = cursor / params.framerate
            segments.append({
                "clip_id": clip_id,
                "start_seconds": round(start, 6),
                "end_seconds": round(end, 6),
                "source_clip_duration_seconds": round(duration, 6),
                "silence_after_seconds": round(silence_frames / params.framerate, 6)
                if index + 1 < len(decoded) else 0.0,
            })
            if index + 1 < len(decoded):
                target.writeframesraw(silence)
                cursor += silence_frames
        target.writeframes(b"")
    return segments


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ASR-AISHELL3-demo/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _source_url(revision: str, speaker_id: str, clip_id: str) -> str:
    return (
        f"https://huggingface.co/datasets/AISHELL/AISHELL-3/resolve/{revision}/"
        f"test/wav/{speaker_id}/{clip_id}.wav"
    )


def _verify_source_transcripts(selection: dict[str, Any]) -> None:
    dataset = selection["dataset"]
    url = (
        f"https://huggingface.co/datasets/AISHELL/AISHELL-3/resolve/"
        f"{dataset['huggingface_revision']}/{dataset['transcript_file']}"
    )
    content = _fetch(url).decode("utf-8")
    indexed = {}
    for line in content.splitlines():
        name, transcript = line.split("\t", 1)
        indexed[Path(name).stem] = transcript
    for clip_id, expected in selection["clips"].items():
        if indexed.get(clip_id) != expected["source_pinyin_line"]:
            raise ValueError(f"upstream transcript validation failed for {clip_id}")
        if transcript_text(expected["source_pinyin_line"]) != expected["reference_text"]:
            raise ValueError(f"reference text does not match source tokens for {clip_id}")


def _clip_path(asset_dir: Path, speaker_id: str, clip_id: str) -> Path:
    if not _SAFE_ID.fullmatch(speaker_id) or not _SAFE_ID.fullmatch(clip_id):
        raise ValueError("invalid dataset speaker or clip id")
    return asset_dir / "source" / speaker_id / f"{clip_id}.wav"


def prepare_assets(
    asset_dir: Path = DEFAULT_ASSET_DIR,
    *,
    download_missing: bool = False,
    verify_source_transcripts: bool = False,
) -> dict[str, Any]:
    selection_path = asset_dir / "source_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if verify_source_transcripts:
        _verify_source_transcripts(selection)

    speakers = {speaker["alias"]: speaker for speaker in selection["speakers"]}
    clip_ids = set(selection["clips"])
    referenced_ids = {
        clip_id
        for speaker in selection["speakers"]
        for clip_id in speaker["registration_clip_ids"]
    }
    referenced_ids.update(turn["clip_id"] for turn in selection["demo_turns"])
    if referenced_ids != clip_ids:
        raise ValueError("source selection must define exactly its registration and demo clips")

    source_paths: dict[str, Path] = {}
    speaker_for_clip: dict[str, str] = {}
    revision = selection["dataset"]["huggingface_revision"]
    for speaker in selection["speakers"]:
        speaker_id = speaker["dataset_speaker_id"]
        speaker_clip_ids = speaker["registration_clip_ids"] + [
            turn["clip_id"] for turn in selection["demo_turns"]
            if turn["speaker"] == speaker["alias"]
        ]
        for clip_id in speaker_clip_ids:
            speaker_for_clip[clip_id] = speaker_id
            path = _clip_path(asset_dir, speaker_id, clip_id)
            if not path.is_file() and download_missing:
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = _fetch(_source_url(revision, speaker_id, clip_id))
                expected_hash = selection["clips"][clip_id]["sha256"]
                if sha256_bytes(payload) != expected_hash:
                    raise ValueError(f"download checksum mismatch for {clip_id}")
                path.write_bytes(payload)
            if not path.is_file():
                raise FileNotFoundError("a selected source WAV is missing")
            expected = selection["clips"][clip_id]["sha256"]
            if sha256_bytes(path.read_bytes()) != expected:
                raise ValueError(f"source checksum mismatch for {clip_id}")
            source_paths[clip_id] = path

    source_records: dict[str, dict[str, Any]] = {}
    for clip_id in sorted(clip_ids):
        spec = selection["clips"][clip_id]
        extracted = transcript_text(spec["source_pinyin_line"])
        if extracted != spec["reference_text"]:
            raise ValueError(f"reference text does not match source tokens for {clip_id}")
        params, _frames = _read_pcm(source_paths[clip_id])
        source_records[clip_id] = {
            "file": source_paths[clip_id].relative_to(asset_dir).as_posix(),
            "source_url": _source_url(revision, speaker_for_clip[clip_id], clip_id),
            "source_split": "test",
            "sha256": spec["sha256"],
            "size_bytes": source_paths[clip_id].stat().st_size,
            "sample_rate_hz": params.framerate,
            "channels": params.nchannels,
            "sample_width_bytes": params.sampwidth,
            "duration_seconds": round(params.nframes / params.framerate, 6),
            "source_pinyin_line": spec["source_pinyin_line"],
            "reference_text": extracted,
        }

    silence_ms = selection["silence_between_clips_ms"]
    registrations = {}
    for alias, speaker in speakers.items():
        clip_ids_for_speaker = speaker["registration_clip_ids"]
        path = asset_dir / "registration" / f"speaker-{alias.lower()}.wav"
        concatenate_wavs(
            [(clip_id, source_paths[clip_id]) for clip_id in clip_ids_for_speaker],
            path,
            silence_ms,
        )
        registrations[alias] = {
            "dataset_speaker_id": speaker["dataset_speaker_id"],
            "file": path.relative_to(asset_dir).as_posix(),
            "source_clip_ids": clip_ids_for_speaker,
            "sha256": sha256_bytes(path.read_bytes()),
            "size_bytes": path.stat().st_size,
            "duration_seconds": round(
                _read_pcm(path)[0].nframes / _read_pcm(path)[0].framerate, 6
            ),
        }

    demo_path = asset_dir / "two-speaker-abab.wav"
    demo_audio_turns = [(turn["clip_id"], source_paths[turn["clip_id"]])
                        for turn in selection["demo_turns"]]
    timeline = concatenate_wavs(demo_audio_turns, demo_path, silence_ms)
    for item, turn in zip(timeline, selection["demo_turns"]):
        item["speaker"] = turn["speaker"]
        item["reference_text"] = selection["clips"][turn["clip_id"]]["reference_text"]
    demo_params, _demo_frames = _read_pcm(demo_path)
    manifest = {
        "format": "aishell3-small-demo-v1",
        "dataset": selection["dataset"],
        "license_file": "LICENSE-Apache-2.0.txt",
        "notice_file": "NOTICE.txt",
        "processing": {
            "sample_rate_change": False,
            "channel_change": False,
            "cropping": False,
            "gain_or_filtering": False,
            "operation": "concatenate complete original WAV frames and insert zero-valued PCM silence",
            "silence_between_clips_ms": silence_ms,
            "registration_sample_note": "Two independent source utterances for that dataset speaker, concatenated for one enrollment upload.",
            "timeline_note": "Intervals mark whole source-file boundaries in the concatenation, not phonetic speech onset/offset; source clips may themselves contain leading/trailing silence.",
            "reference_note": "Reference text is the original Chinese character tokens from test/content.txt, concatenated after dropping alternating pinyin tokens; no human correction or ASR prediction is included.",
        },
        "speakers": registrations,
        "source_clips": source_records,
        "demo_audio": {
            "file": demo_path.relative_to(asset_dir).as_posix(),
            "sha256": sha256_bytes(demo_path.read_bytes()),
            "size_bytes": demo_path.stat().st_size,
            "sample_rate_hz": demo_params.framerate,
            "channels": demo_params.nchannels,
            "sample_width_bytes": demo_params.sampwidth,
            "duration_seconds": round(demo_params.nframes / demo_params.framerate, 6),
            "turn_order": [turn["speaker"] for turn in selection["demo_turns"]],
            "segments": timeline,
        },
    }
    (asset_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--download-missing", action="store_true",
                        help="fetch only the selected WAVs from the pinned official dataset revision")
    parser.add_argument("--verify-source-transcripts", action="store_true",
                        help="fetch official test/content.txt and verify selected transcript lines")
    args = parser.parse_args(argv)
    try:
        manifest = prepare_assets(
            args.asset_dir,
            download_missing=args.download_missing,
            verify_source_transcripts=args.verify_source_transcripts,
        )
    except Exception:
        # Never print user paths, URLs containing credentials, or source text on failure.
        print(json.dumps({"status": "fail", "reason": "asset_validation_or_generation_failed"}))
        return 1
    print(json.dumps({
        "status": "pass",
        "source_clip_count": len(manifest["source_clips"]),
        "demo_seconds": manifest["demo_audio"]["duration_seconds"],
        "manifest": "manifest.json",
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
