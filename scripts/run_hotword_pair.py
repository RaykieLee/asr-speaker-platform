#!/usr/bin/env python3
"""Run a paired, model-level offline hotword evaluation on one annotated WAV.

This intentionally exercises the local sherpa-onnx offline recognizer only. It
does not run the platform's upload, VAD, segmentation, diarization, or API path.
Raw hypotheses and the manifest are written only to the caller-selected output
directory; do not commit that directory when it contains evaluation text/audio.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.hotwords.compiler import HotwordCompiler
from backend.app.hotwords.model_tokenizer import load_vocabulary, mixed_cjk_bpe_tokenizer
from scripts.compare_public_outputs import edit_distance, normalize_zh


DEFAULT_MODEL_DIR = ROOT / "models" / "offline-multi"
DEFAULT_MODEL_VERSION = "zipformer-multi-zh-hans-2023-9-2"
MODEL_FILES = ("encoder-epoch-20-avg-1.int8.onnx", "decoder-epoch-20-avg-1.int8.onnx",
               "joiner-epoch-20-avg-1.int8.onnx", "tokens.txt", "bpe.model", "bpe.vocab")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_hotword(value: str) -> tuple[str, float]:
    """Parse a phrase with the decoder's per-phrase trailing ``:score`` syntax."""
    phrase, sep, raw_score = value.rpartition(":")
    if not sep:
        phrase, score = value, 1.0
    else:
        try:
            score = float(raw_score)
        except ValueError as exc:
            raise ValueError(f"hotword score must be numeric in {value!r}") from exc
    phrase = phrase.strip()
    if not phrase or not math.isfinite(score) or not 0 < score <= 20:
        raise ValueError(f"hotword must have text and a score in (0, 20]: {value!r}")
    return phrase, score


def _reference(annotation: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    segments = annotation.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("annotation must contain non-empty segments")
    checked: list[dict[str, Any]] = []
    for row in segments:
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise ValueError("annotation segments must contain string text")
        start = float(row.get("start", row.get("start_seconds", 0)))
        end = float(row.get("end", row.get("end_seconds", 0)))
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError("annotation segment times must be finite and end after start")
        checked.append({**row, "start": start, "end": end})
    ordered = sorted(checked, key=lambda row: (row["start"], row["end"]))
    # This script scores a single transcript, so overlapping references have no
    # unique global word order and are rejected rather than silently flattened.
    for left, right in zip(ordered, ordered[1:]):
        if right["start"] < left["end"]:
            raise ValueError("reference contains overlapping segments; select a non-overlap window")
    text = normalize_zh("".join(row["text"] for row in ordered))
    if not text:
        raise ValueError("normalized reference is empty")
    return text, ordered


def score_hypothesis(reference: str, hypothesis: str, targets: list[str]) -> dict[str, Any]:
    ref, hyp = normalize_zh(reference), normalize_zh(hypothesis)
    terms = [normalize_zh(term) for term in targets]
    terms = [term for term in terms if term]
    present = [term for term in terms if term in ref]
    absent = [term for term in terms if term not in ref]
    errors = edit_distance(ref, hyp)
    return {
        "cer": {"errors": errors, "reference_characters": len(ref),
                "value": errors / len(ref) if ref else None},
        "hotwords": {
            "target_hits": sum(term in hyp for term in present),
            "targets_present_in_reference": len(present),
            "target_hit_rate": sum(term in hyp for term in present) / len(present) if present else None,
            "false_insertions": sum(term in hyp for term in absent),
            "targets_absent_from_reference": len(absent),
            "false_insertion_rate": sum(term in hyp for term in absent) / len(absent) if absent else None,
            "details": [{"term": term, "present_in_reference": term in ref,
                         "present_in_output": term in hyp} for term in terms],
        },
    }


def _compile_hotwords(model_dir: Path, requested: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    parsed = [parse_hotword(item) for item in requested]
    if not parsed:
        raise ValueError("at least one --hotword phrase[:score] is required")
    tokenizer = mixed_cjk_bpe_tokenizer(model_dir / "bpe.model")
    compiler = HotwordCompiler(tokenizer, mode="offline", tokenizer_id="offline-multi:bpe.model",
                               modeling_unit="cjkchar+bpe", vocabulary=load_vocabulary(model_dir / "tokens.txt"))
    compiled: list[dict[str, Any]] = []
    for phrase, score in parsed:
        result = compiler.compile([{"word": phrase}], weight=1.0)
        item = result["entries"][0]
        compiled.append({"phrase": phrase, "tokens": item["tokens"], "score": score})
    return compiled, _format_hotwords(compiled)


def _format_hotwords(compiled: list[dict[str, Any]]) -> list[str]:
    # sherpa's dynamic offline hotword grammar accepts per-phrase trailing
    # scores as a separate final item: TOKEN TOKEN :score/OTHER :score.
    return [" ".join(item["tokens"]) + f" :{item['score']:g}" for item in compiled]


def _new_recognizer(model_dir: Path, *, threads: int, provider: str) -> Any:
    import sherpa_onnx

    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(model_dir / MODEL_FILES[0]), decoder=str(model_dir / MODEL_FILES[1]),
        joiner=str(model_dir / MODEL_FILES[2]), tokens=str(model_dir / "tokens.txt"),
        num_threads=threads, sample_rate=16_000, feature_dim=80,
        decoding_method="modified_beam_search", provider=provider,
        hotwords_score=1.0, modeling_unit="cjkchar+bpe", bpe_vocab=str(model_dir / "bpe.vocab"),
    )


def _decode(recognizer: Any, samples: Any, *, encoded_hotwords: list[str] | None) -> str:
    stream = (recognizer.create_stream(hotwords="/".join(encoded_hotwords))
              if encoded_hotwords else recognizer.create_stream())
    stream.accept_waveform(16_000, samples)
    recognizer.decode_stream(stream)
    return str(getattr(stream.result, "text", stream.result)).strip()


def run_pair(wav_path: Path, annotation_path: Path, output_dir: Path, *,
             model_dir: Path = DEFAULT_MODEL_DIR, model_version: str = DEFAULT_MODEL_VERSION,
             hotword_args: list[str], threads: int = 2, provider: str = "cpu",
             overwrite: bool = False,
             recognizer_factory: Callable[..., Any] = _new_recognizer,
             now: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(0o700)
    raw_paths = [output_dir / "disabled.raw.json", output_dir / "enabled.raw.json"]
    manifest_path = output_dir / "manifest.json"
    if not overwrite and any(path.exists() for path in [*raw_paths, manifest_path]):
        raise FileExistsError(f"output exists in {output_dir}; choose a new directory or pass --overwrite")

    import soundfile as sf
    info = sf.info(str(wav_path))
    if info.samplerate != 16_000 or info.channels != 1:
        raise ValueError("evaluation WAV must be mono 16 kHz, matching AISHELL-4 preparation")
    samples, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if sample_rate != 16_000 or samples.ndim != 1 or not len(samples):
        raise ValueError("evaluation WAV must contain non-empty mono 16 kHz audio")
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    reference, segments = _reference(annotation)
    targets = annotation.get("hotword_targets", [])
    if not isinstance(targets, list) or any(not isinstance(term, str) for term in targets):
        raise ValueError("annotation hotword_targets must be an array of strings")
    compiled, encoded = _compile_hotwords(model_dir, hotword_args)
    requested_terms = {normalize_zh(item["phrase"]) for item in compiled}
    annotated_terms = {normalize_zh(term) for term in targets}
    if requested_terms != annotated_terms:
        raise ValueError("annotation hotword_targets must exactly match the requested --hotword phrases")

    modes: dict[str, Any] = {}
    run_metadata: dict[str, Any] = {}
    raw_hypotheses: dict[str, str] = {}
    for mode, hotwords in (("disabled", None), ("enabled", encoded)):
        started = now()
        recognizer = recognizer_factory(model_dir, threads=threads, provider=provider)
        loaded = now()
        hypothesis = _decode(recognizer, samples, encoded_hotwords=hotwords)
        finished = now()
        modes[mode] = {"mode": mode, **score_hypothesis(reference, hypothesis, targets)}
        raw_hypotheses[mode] = hypothesis
        run_metadata[mode] = {"model_load_seconds": loaded - started,
                              "decode_seconds": finished - loaded,
                              "total_seconds": finished - started}

    model_hashes = {name: sha256_file(model_dir / name) for name in MODEL_FILES}
    try:
        import sherpa_onnx
        sherpa_version = getattr(sherpa_onnx, "__version__", "unknown")
    except ImportError:
        sherpa_version = "not-importable"
    manifest = {
        "schema_version": 1,
        "scope": "model-level offline sherpa-onnx inference; excludes platform upload, VAD, segmentation, diarization, and API",
        "dataset": annotation.get("source", {}).get("dataset", "AISHELL-4"),
        "inputs": {"wav": str(wav_path.resolve()), "wav_sha256": sha256_file(wav_path),
                   "annotation": str(annotation_path.resolve()),
                   "annotation_sha256": sha256_file(annotation_path),
                   "sample_rate": 16_000, "channels": 1, "duration_seconds": len(samples) / 16_000,
                   "reference_characters": len(reference), "reference_segments": len(segments),
                   "reference_overlap": False},
        "model": {"id": "offline-multi", "version": model_version, "directory": str(model_dir.resolve()),
                  "files_sha256": model_hashes},
        "runtime": {"sherpa_onnx": sherpa_version, "python": sys.version.split()[0],
                    "platform": sys.platform},
        "settings": {"provider": provider, "threads": threads,
                     "decoding_method": "modified_beam_search", "modeling_unit": "cjkchar+bpe",
                     "hotword_format": "model-tokenized phrase with trailing :score; phrases slash-separated",
                     "hotwords": compiled, "timings_seconds": run_metadata},
        "results": modes,
        "raw_results": {"disabled": raw_paths[0].name, "enabled": raw_paths[1].name},
        "transcript_policy": "raw hypotheses are stored in the selected output directory; transcript text is not duplicated in this manifest",
    }
    for index, mode in enumerate(("disabled", "enabled")):
        raw_paths[index].write_text(
            json.dumps({"mode": mode, "text": raw_hypotheses[mode]}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        raw_paths[index].chmod(0o600)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path.chmod(0o600)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="private/local result directory; raw transcript JSON is written here")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--hotword", action="append", default=[], metavar="PHRASE[:SCORE]",
                        help="repeat for each phrase; default score is 1.0")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        result = run_pair(args.wav, args.annotation, args.out_dir, model_dir=args.model_dir,
                          model_version=args.model_version, hotword_args=args.hotword,
                          threads=args.threads, provider=args.provider, overwrite=args.overwrite)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print(f"hotword pair evaluation error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"manifest": str((args.out_dir / "manifest.json").resolve()),
                      "results": result["results"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
