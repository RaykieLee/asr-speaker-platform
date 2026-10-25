#!/usr/bin/env python3
"""Compare two platform JSON outputs against one public-dataset annotation."""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.quality import EvaluationError, evaluate


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("segments"), list):
        raise ValueError(f"{path} must contain an object with a segments array")
    return value


def normalize_zh(text: str) -> str:
    """Remove Praat silence tags, whitespace and Unicode punctuation for CER."""
    text = re.sub(r"<\s*(?:sil|#)\s*>", "", text, flags=re.IGNORECASE)
    return "".join(ch for ch in text if not ch.isspace()
                   and not unicodedata.category(ch).startswith("P"))


def edit_distance(reference: str, hypothesis: str) -> int:
    prev = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        cur = [i]
        for j, b in enumerate(hypothesis, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1]


def _transcript(data: dict[str, Any], key: str = "text") -> str:
    segments = data["segments"]
    if any(not isinstance(seg, dict) or not isinstance(seg.get(key, ""), str) for seg in segments):
        raise ValueError("all segments must be objects with string text")
    ordered = sorted(segments, key=lambda seg: float(seg.get("start", seg.get("start_seconds", 0))))
    return normalize_zh("".join(seg.get(key, "") for seg in ordered))


def _hotwords(annotation: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    targets = annotation.get("hotword_targets")
    if not isinstance(targets, list) or any(not isinstance(word, str) for word in targets):
        raise ValueError("annotation must contain string hotword_targets")
    ref, hyp = _transcript(annotation), _transcript(output)
    present = [normalize_zh(word) for word in targets if normalize_zh(word) and normalize_zh(word) in ref]
    absent = [normalize_zh(word) for word in targets if normalize_zh(word) and normalize_zh(word) not in ref]
    hits = sum(word in hyp for word in present)
    false_insertions = sum(word in hyp for word in absent)
    return {"mode": "enabled" if output.get("hotwords_enabled") is True else
                     "disabled" if output.get("hotwords_enabled") is False else "unknown",
            "target_hits": hits, "targets_present_in_reference": len(present),
            "target_hit_rate": hits / len(present) if present else None,
            "false_insertions": false_insertions, "targets_absent_from_reference": len(absent),
            "false_insertion_rate": false_insertions / len(absent) if absent else None,
            "target_details": [{"term": term, "present_in_reference": term in ref,
                                "present_in_output": term in hyp} for term in
                               [normalize_zh(word) for word in targets if normalize_zh(word)]]}


def compare(annotation: dict[str, Any], outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reference = _transcript(annotation)
    result: dict[str, Any] = {"schema_version": 1, "cer_normalization":
                              "global concatenation ordered by start; remove whitespace, punctuation and <sil>/<#>; CER uses Unicode characters",
                              "reference_characters": len(reference), "platforms": {}}
    for name, output in outputs.items():
        hypothesis = _transcript(output)
        edits = edit_distance(reference, hypothesis)
        try:
            quality = evaluate(annotation, output)["metrics"]
        except EvaluationError as exc:
            raise ValueError(f"{name} output is incompatible with evaluation.quality: {exc}") from exc
        result["platforms"][name] = {
            "cer": {"value": edits / len(reference) if reference else None, "errors": edits,
                    "reference_characters": len(reference),
                    "status": "evaluated" if reference else "not_evaluable_empty_reference"},
            "speaker_metrics": {key: quality[key] for key in (
                "der_like", "registered_hit_rate", "unknown_false_attribution_rate", "boundary_error_seconds")},
            "hotwords": _hotwords(annotation, output),
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--enabled", required=True, type=Path, help="platform output with hotwords enabled")
    parser.add_argument("--disabled", required=True, type=Path, help="platform output with hotwords disabled")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        result = compare(_load(args.annotation), {"enabled": _load(args.enabled), "disabled": _load(args.disabled)})
    except (ValueError, TypeError, KeyError) as exc:
        print(f"comparison error: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
