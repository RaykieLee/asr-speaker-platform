#!/usr/bin/env python3
"""Run an offline ASR + diarization replay and write reproducible artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote


class ReplayError(Exception):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _one(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(path for path in directory.glob(pattern) if path.is_file())
    if len(matches) != 1:
        raise ReplayError(f"{label}: expected exactly one {pattern} in {directory}; found {len(matches)}")
    return matches[0].resolve()


class ReadOnlyProfiles:
    """Minimal SpeakerMatcher store backed by SQLite opened in mode=ro."""
    def __init__(self, path: Path):
        uri = f"file:{quote(str(path.resolve()))}?mode=ro"
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
    return SherpaOfflineRecognizer.load(
        artifact, files, modeling_unit="cjkchar+bpe", bpe_vocab=vocab,
    )


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from backend.app.pipelines.file_asr import FileASRService
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
    try:
        annotation_data = json.loads(annotation.read_text(encoding="utf-8"))
        if not isinstance(annotation_data, dict):
            raise ValueError("annotation root must be an object")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ReplayError(f"cannot read annotation {annotation}: {exc}") from exc

    model_files = _model_files(model_dir)
    encoder, decoder, joiner, tokens, vocab = model_files
    asr_version = args.asr_version or model_dir.name
    recognizer = _load_recognizer(model_dir, asr_version, model_files)
    diarizer = SherpaOfflineDiarizer.load(segmentation, embedding,
        num_speakers=args.num_speakers, cluster_threshold=args.cluster_threshold)
    extractor = SherpaSpeakerEmbeddingAdapter(SpeakerModel(embedding, args.speaker_model_version))
    profiles = ReadOnlyProfiles(database)
    try:
        matcher = SpeakerMatcher(profiles, extractor)
        result = FileASRService(recognizer, diarizer=diarizer, speaker_matcher=matcher).transcribe_file(audio)
    finally:
        profiles.close()

    output = {"model_version": result.model_version, "speaker_diarization": result.speaker_diarization,
              "speaker_mappings": list(result.speaker_mappings), "segments": [
                  {"start_seconds": segment.start, "end_seconds": segment.end, "text": segment.text,
                   "local_speaker_id": segment.local_speaker_id,
                   "registered_speaker_id": segment.registered_speaker_id, "overlap": segment.overlap,
                   "metadata": segment.metadata or {}}
                  for segment in result.segments]}
    metrics = evaluate(annotation_data, output)
    input_paths = {"audio": audio, "annotation": annotation, "registration_database": database,
                   "asr_encoder": encoder, "asr_decoder": decoder, "asr_joiner": joiner,
                   "asr_tokens": tokens, "segmentation_model": segmentation, "speaker_model": embedding}
    if vocab:
        input_paths["asr_bpe_vocab"] = vocab
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parameters": {"cluster_threshold": args.cluster_threshold, "num_speakers": args.num_speakers,
                       "channel_mode": "mixdown", "asr_version": asr_version,
                       "speaker_model_version": args.speaker_model_version},
        "models": {"asr_model_directory": str(model_dir), "asr_model_id": model_dir.name,
                   "segmentation_model": str(segmentation), "speaker_embedding_model": str(embedding)},
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in input_paths.items()},
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
        outdir = args.output_dir.resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        result_path, metrics_path, manifest_path = (outdir / "result.json", outdir / "metrics.json", outdir / "run-manifest.json")
        result_payload = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
        metrics_payload = json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
        # Write both artifacts before the manifest, which records their final hashes.
        result_path.write_text(result_payload, encoding="utf-8")
        metrics_path.write_text(metrics_payload, encoding="utf-8")
        manifest["outputs"] = {"result": {"path": str(result_path), "sha256": sha256(result_path)},
                               "metrics": {"path": str(metrics_path), "sha256": sha256(metrics_path)}}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"quality replay failed: {exc}", file=sys.stderr)
        return 2
    print(f"result: {result_path}\nmetrics: {metrics_path}\nmanifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
