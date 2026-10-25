#!/usr/bin/env python3
"""Run an offline ASR + diarization replay and write reproducible artifacts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote


class ReplayError(Exception):
    pass


INFERENCE_CONFIG = {
    "asr": {"modeling_unit": "cjkchar+bpe", "decoding_method": "modified_beam_search",
            "num_threads": 2, "provider": "cpu"},
    "file_asr": {"max_chunk_seconds": 60.0, "overlap_seconds": 0.8},
    "diarization": {"num_threads": 2, "provider": "cpu", "min_duration_on": 0.3, "min_duration_off": 0.5},
    "speaker_embedding": {"num_threads": 1, "provider": "cpu"},
    "speaker_matching": {"min_similarity": 0.55, "min_margin": 0.05,
                         "min_segment_seconds": 1.0, "max_segments_per_speaker": 5,
                         "min_segments_per_speaker": 2},
}


def _reject_wal(database: Path) -> None:
    wal = Path(f"{database}-wal")
    if wal.exists():
        raise ReplayError(f"registration database has a WAL file: {wal}; create a consistent backup or stop writers and remove sidecars before replay")


def _assert_unchanged(path: Path, expected_hash: str) -> None:
    if sha256(path) != expected_hash:
        raise ReplayError(f"registration database changed during replay: {path}; use a static snapshot")


def _snapshot_inputs(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    return {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()}


def _assert_inputs_unchanged(snapshot: dict[str, dict[str, str]]) -> None:
    for name, item in snapshot.items():
        if sha256(Path(item["path"])) != item["sha256"]:
            message = "registration database changed during replay" if name == "registration_database" else f"input changed during replay ({name})"
            raise ReplayError(f"{message}: {item['path']}")


def _runtime_provenance() -> dict[str, Any]:
    try:
        sherpa_version: str | None = importlib.metadata.version("sherpa-onnx")
    except importlib.metadata.PackageNotFoundError:
        sherpa_version = None
    repo_root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout
        git_status_available = True
        git_dirty = bool(status)
    except (OSError, subprocess.CalledProcessError):
        git_status_available = False
        git_dirty = True  # Fail conservatively: an unavailable status is not evidence of a clean tree.
    return {"python_version": platform.python_version(), "sherpa_onnx_version": sherpa_version,
            "git_commit": commit, "git_dirty": git_dirty,
            "git_status_available": git_status_available}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_private(path: Path, payload: str) -> None:
    """Create or replace a JSON artifact with private permissions from the outset."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)


def _one(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(path for path in directory.glob(pattern) if path.is_file())
    if len(matches) != 1:
        raise ReplayError(f"{label}: expected exactly one {pattern} in {directory}; found {len(matches)}")
    return matches[0].resolve()


class ReadOnlyProfiles:
    """Minimal SpeakerMatcher store backed by SQLite opened in mode=ro."""
    def __init__(self, path: Path):
        uri = f"file:{quote(str(path.resolve()))}?mode=ro&immutable=1"
        self.connection = sqlite3.connect(uri, uri=True)
        self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        self.connection.close()

    def list_speakers(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.connection.execute("SELECT * FROM speakers ORDER BY created_at, id")]

    def get_speaker_profile(self, speaker_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM speaker_profiles WHERE speaker_id=?", (speaker_id,)).fetchone()
        if row is None:
            return None
        value = dict(row)
        blob = value.get("embedding")
        if blob is not None:
            import struct
            value["embedding"] = list(struct.unpack(f"<{len(blob) // 4}f", blob))
        return value


def _model_files(model_dir: Path) -> tuple[Path, Path, Path, Path, Path | None]:
    root = model_dir.resolve()
    if not root.is_dir():
        raise ReplayError(f"ASR model directory does not exist: {root}")
    encoder = _one(root, "encoder*.onnx", "encoder")
    decoder = _one(root, "decoder*.onnx", "decoder")
    joiner = _one(root, "joiner*.onnx", "joiner")
    tokens = _one(root, "tokens.txt", "tokens")
    vocabs = sorted(root.glob("bpe.vocab"))
    return encoder, decoder, joiner, tokens, vocabs[0].resolve() if vocabs else None


def _load_recognizer(model_dir: Path, asr_version: str,
                     model_files: tuple[Path, Path, Path, Path, Path | None]) -> Any:
    from backend.app.engines.base import ModelArtifact
    from backend.app.engines.sherpa_offline import OfflineTransducerFiles, SherpaOfflineRecognizer

    encoder, decoder, joiner, tokens, vocab = model_files
    files = OfflineTransducerFiles(encoder, decoder, joiner, tokens)
    artifact = ModelArtifact(model_dir.name, "transducer", asr_version,
                             (encoder, decoder, joiner, tokens))
    return SherpaOfflineRecognizer.load(artifact, files, **INFERENCE_CONFIG["asr"], bpe_vocab=vocab)


def _create_file_asr(recognizer: Any, diarizer: Any, matcher: Any) -> Any:
    from backend.app.pipelines.file_asr import FileASRService

    return FileASRService(recognizer, diarizer=diarizer, speaker_matcher=matcher,
                          **INFERENCE_CONFIG["file_asr"])


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from backend.app.speakers.diarization import SherpaOfflineDiarizer
    from backend.app.speakers.matching import SpeakerMatcher
    from backend.app.speakers.service import SherpaSpeakerEmbeddingAdapter, SpeakerModel
    from evaluation.quality import evaluate

    audio, annotation, database = args.audio.resolve(), args.annotation.resolve(), args.registry.resolve()
    segmentation, embedding = args.segmentation_model.resolve(), args.speaker_model.resolve()
    model_dir = args.asr_model_dir.resolve()
    for path in (audio, annotation, database, segmentation, embedding):
        if not path.is_file():
            raise ReplayError(f"required input file does not exist: {path}")
    _reject_wal(database)
    model_files = _model_files(model_dir)
    encoder, decoder, joiner, tokens, vocab = model_files
    asr_version = args.asr_version or model_dir.name
    input_paths = {"audio": audio, "annotation": annotation, "registration_database": database,
                   "asr_encoder": encoder, "asr_decoder": decoder, "asr_joiner": joiner,
                   "asr_tokens": tokens, "segmentation_model": segmentation, "speaker_model": embedding}
    if vocab:
        input_paths["asr_bpe_vocab"] = vocab
    input_snapshot = _snapshot_inputs(input_paths)
    try:
        annotation_data = json.loads(annotation.read_text(encoding="utf-8"))
        if not isinstance(annotation_data, dict):
            raise ValueError("annotation root must be an object")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ReplayError(f"cannot read annotation {annotation}: {exc}") from exc
    recognizer = _load_recognizer(model_dir, asr_version, model_files)
    diarizer = SherpaOfflineDiarizer.load(segmentation, embedding,
        **INFERENCE_CONFIG["diarization"], num_speakers=args.num_speakers,
        cluster_threshold=args.cluster_threshold)
    extractor = SherpaSpeakerEmbeddingAdapter(
        SpeakerModel(embedding, args.speaker_model_version, **INFERENCE_CONFIG["speaker_embedding"]))
    profiles = ReadOnlyProfiles(database)
    try:
        matcher = SpeakerMatcher(profiles, extractor, **INFERENCE_CONFIG["speaker_matching"])
        result = _create_file_asr(recognizer, diarizer, matcher).transcribe_file(audio)
    finally:
        profiles.close()
    _reject_wal(database)
    _assert_inputs_unchanged(input_snapshot)

    output = {"model_version": result.model_version, "speaker_diarization": result.speaker_diarization,
              "speaker_mappings": list(result.speaker_mappings), "segments": [
                  {"start_seconds": segment.start, "end_seconds": segment.end, "text": segment.text,
                   "local_speaker_id": segment.local_speaker_id,
                   "registered_speaker_id": segment.registered_speaker_id, "overlap": segment.overlap,
                   "metadata": segment.metadata or {}}
                  for segment in result.segments]}
    metrics = evaluate(annotation_data, output)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parameters": {"cluster_threshold": args.cluster_threshold, "num_speakers": args.num_speakers,
                       "channel_mode": "mixdown", "asr_version": asr_version,
                       "speaker_model_version": args.speaker_model_version,
                       "inference": INFERENCE_CONFIG},
        "runtime": _runtime_provenance(),
        "models": {"asr_model_directory": str(model_dir), "asr_model_id": model_dir.name,
                   "segmentation_model": str(segmentation), "speaker_embedding_model": str(embedding)},
        "inputs": input_snapshot,
        "outputs": {},
    }
    return output, metrics, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path, help="existing registration SQLite database, opened read-only")
    parser.add_argument("--asr-model-dir", required=True, type=Path, help="directory with encoder*.onnx, decoder*.onnx, joiner*.onnx, tokens.txt")
    parser.add_argument("--asr-version", help="explicit ASR model version; defaults to model directory name")
    parser.add_argument("--segmentation-model", required=True, type=Path)
    parser.add_argument("--speaker-model", required=True, type=Path)
    parser.add_argument("--speaker-model-version", required=True)
    parser.add_argument("--cluster-threshold", required=True, type=float)
    parser.add_argument("--num-speakers", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        output, metrics, manifest = run(args)
        _reject_wal(args.registry.resolve())
        outdir = args.output_dir.resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        outdir.chmod(0o700)
        result_path, metrics_path, manifest_path = (outdir / "result.json", outdir / "metrics.json", outdir / "run-manifest.json")
        result_payload = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
        metrics_payload = json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
        # Write both artifacts before the manifest, which records their final hashes.
        _write_private(result_path, result_payload)
        _write_private(metrics_path, metrics_payload)
        manifest["outputs"] = {"result": {"path": str(result_path), "sha256": sha256(result_path)},
                               "metrics": {"path": str(metrics_path), "sha256": sha256(metrics_path)}}
        _write_private(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    except Exception as exc:
        print(f"quality replay failed: {exc}", file=sys.stderr)
        return 2
    print(f"result: {result_path}\nmetrics: {metrics_path}\nmanifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
