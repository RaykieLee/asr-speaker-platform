"""Evaluate ASR JSON against explicit human annotations; no model or audio access."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


class EvaluationError(ValueError):
    pass


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"{path} must contain a JSON object")
    return value


def _segments(data: dict[str, Any], label: str) -> list[dict[str, Any]] | None:
    raw = data.get("segments")
    if not isinstance(raw, list):
        return None
    checked = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise EvaluationError(f"{label}.segments[{index}] must be an object")
        start, end = item.get("start", item.get("start_seconds")), item.get("end", item.get("end_seconds"))
        if not _finite_nonnegative(start) or not _finite_nonnegative(end) or end < start:
            raise EvaluationError(f"{label}.segments[{index}] needs finite start/end with end >= start")
        text = item.get("text")
        if text is not None and not isinstance(text, str):
            raise EvaluationError(f"{label}.segments[{index}].text must be a string")
        speaker = item.get("speaker_id", item.get("local_speaker_id"))
        checked.append({**item, "speaker_id": speaker, "_start": float(start), "_end": float(end), "_text": text or ""})
    return checked


def _finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        curr = [i]
        for j, right in enumerate(b, 1):
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + (left != right)))
        prev = curr
    return prev[-1]


def _tokens(text: str, lang: str) -> list[str]:
    if lang == "zh":
        return [char for char in text if not char.isspace()]
    if lang == "en":
        return re.findall(r"[\w]+(?:['’][\w]+)?", text.casefold(), flags=re.UNICODE)
    raise EvaluationError("segment lang must be 'zh' or 'en'")


def _rate(errors: int, total: int) -> dict[str, Any]:
    return {"value": errors / total if total else None, "errors": errors, "reference_units": total,
            "status": "evaluated" if total else "not_evaluable_empty_reference"}


def _maximum_overlap_mapping(refs: list[str], hyps: list[str],
                             overlap: dict[tuple[str, str], float]) -> dict[str, str]:
    """Maximum-weight one-to-one matching, allowing unmatched IDs on the larger side."""
    if not refs or not hyps:
        return {}
    # Hungarian algorithm for n rows <= m columns. Put the smaller identity set
    # on rows so every row is paired and excess IDs on the larger side stay unmatched.
    rows_are_refs = len(refs) <= len(hyps)
    rows, cols = (refs, hyps) if rows_are_refs else (hyps, refs)
    n, m = len(rows), len(cols)
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    p, way = [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float("inf"), 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                r, h = (rows[i0 - 1], cols[j - 1]) if rows_are_refs else (cols[j - 1], rows[i0 - 1])
                current = -overlap[(r, h)] - u[i0] - v[j]
                if current < minv[j]:
                    minv[j], way[j] = current, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    mapping: dict[str, str] = {}
    for j in range(1, m + 1):
        if p[j]:
            row_id, col_id = rows[p[j] - 1], cols[j - 1]
            hyp_id, ref_id = (col_id, row_id) if rows_are_refs else (row_id, col_id)
            mapping[hyp_id] = ref_id
    return mapping


def _transcript_metrics(ref: list[dict[str, Any]], hyp: list[dict[str, Any]] | None) -> dict[str, Any]:
    if hyp is None:
        return {"cer": {"value": None, "status": "not_evaluable_missing_output"},
                "wer": {"value": None, "status": "not_evaluable_missing_output"}}
    result = {}
    for lang, key in (("zh", "cer"), ("en", "wer")):
        refs = [s for s in ref if s.get("lang") == lang]
        if any(s.get("lang") not in ("zh", "en") for s in ref):
            raise EvaluationError("every annotated segment must include lang: 'zh' or 'en'")
        # Assign output text to a language using its midpoint and the human-labelled timeline.
        ref_ranges = [(s["_start"], s["_end"], s["lang"]) for s in ref if s["_end"] > s["_start"]]
        htext = []
        for out in hyp:
            mid = (out["_start"] + out["_end"]) / 2
            labels = [language for start, end, language in ref_ranges if start <= mid < end]
            if lang in labels:
                htext.append(out["_text"])
        reference = _tokens("".join(s["_text"] for s in refs), lang)
        hypothesis = _tokens("".join(htext), lang)
        result[key] = _rate(_edit_distance(reference, hypothesis), len(reference))
    return result


def _speaker_metrics(ref: list[dict[str, Any]], hyp: list[dict[str, Any]] | None) -> dict[str, Any]:
    if hyp is None:
        return {"der_like": {"value": None, "status": "not_evaluable_missing_output"},
                "registered_hit_rate": {"value": None, "status": "not_evaluable_missing_output"},
                "unknown_false_attribution_rate": {"value": None, "status": "not_evaluable_missing_output"},
                "boundary_error_seconds": {"value": None, "status": "not_evaluable_missing_output"}}
    required = all(isinstance(s.get("speaker_id"), str) and s["speaker_id"] for s in ref) and all(
        isinstance(s.get("speaker_id"), str) and s["speaker_id"] for s in hyp)
    if not required or not ref:
        unavailable = {"value": None, "status": "not_evaluable_missing_speaker_labels"}
        return {"der_like": unavailable, "registered_hit_rate": unavailable,
                "unknown_false_attribution_rate": unavailable, "boundary_error_seconds": unavailable}

    # Match arbitrary output-local IDs to human IDs by maximum overlap.
    refs = sorted({str(s.get("speaker_id")) for s in ref})
    hyps = sorted({str(s.get("speaker_id")) for s in hyp})
    overlap: dict[tuple[str, str], float] = {(a, b): 0.0 for a in refs for b in hyps}
    for r in ref:
        for h in hyp:
            d = max(0.0, min(r["_end"], h["_end"]) - max(r["_start"], h["_start"]))
            overlap[(str(r["speaker_id"]), str(h["speaker_id"]))] += d
    mapping = _maximum_overlap_mapping(refs, hyps, overlap)

    points = sorted({x for s in ref + hyp for x in (s["_start"], s["_end"])})
    total = errors = reg_total = reg_hit = unknown_total = unknown_false = excluded_overlap = 0.0
    for left, right in zip(points, points[1:]):
        if right <= left:
            continue
        mid, duration = (left + right) / 2, right - left
        active_ref = [s for s in ref if s["_start"] <= mid < s["_end"]]
        if len(active_ref) > 1:
            excluded_overlap += duration
            continue
        r = active_ref[0] if active_ref else None
        h = next((s for s in hyp if s["_start"] <= mid < s["_end"]), None)
        if r is None:
            continue
        total += duration
        if h is None or mapping.get(str(h["speaker_id"])) != str(r["speaker_id"]):
            errors += duration
        expected_registered = r.get("registered_identity")
        predicted_registered = h.get("registered_speaker_id") if h else None
        if expected_registered:
            reg_total += duration
            reg_hit += duration * (predicted_registered == expected_registered)
        else:
            unknown_total += duration
            unknown_false += duration * bool(predicted_registered)
    ordered_ref = sorted(ref, key=lambda x: x["_start"])
    ref_boundaries = [ordered_ref[i]["_start"] for i in range(1, len(ordered_ref))
                      if ordered_ref[i].get("speaker_id") != ordered_ref[i-1].get("speaker_id")]
    ordered_hyp = sorted(hyp, key=lambda x: x["_start"])
    hyp_boundaries = [ordered_hyp[i]["_start"] for i in range(1, len(ordered_hyp))
                      if ordered_hyp[i].get("speaker_id") != ordered_hyp[i-1].get("speaker_id")]
    paired_errors: list[float] = []
    remaining = list(hyp_boundaries)
    for boundary_time in ref_boundaries:
        if remaining:
            nearest = min(remaining, key=lambda value: abs(value - boundary_time))
            paired_errors.append(abs(nearest - boundary_time))
            remaining.remove(nearest)
    boundary = (sum(paired_errors) / len(paired_errors)
                if paired_errors and not excluded_overlap else None)
    return {
        "der_like": {"value": errors / total if total else None, "error_seconds": errors,
                     "reference_speech_seconds": total, "excluded_overlap_seconds": excluded_overlap,
                     "scoring_scope": "reference overlap intervals excluded",
                     "status": "evaluated" if total else
                              ("not_evaluable_only_overlapping_reference_speech" if excluded_overlap else
                               "not_evaluable_empty_reference")},
        "registered_hit_rate": {"value": reg_hit / reg_total if reg_total else None,
                                 "matched_seconds": reg_hit, "registered_reference_seconds": reg_total,
                                 "excluded_overlap_seconds": excluded_overlap,
                                 "status": "evaluated" if reg_total else "not_evaluable_no_registered_speech"},
        "unknown_false_attribution_rate": {"value": unknown_false / unknown_total if unknown_total else None,
                                           "false_attributed_seconds": unknown_false, "unknown_reference_seconds": unknown_total,
                                           "excluded_overlap_seconds": excluded_overlap,
                                           "status": "evaluated" if unknown_total else "not_evaluable_no_unknown_speech"},
        "boundary_error_seconds": {"value": boundary, "reference_boundaries": len(ref_boundaries),
                                   "hypothesis_boundaries": len(hyp_boundaries),
                                   "matched_boundaries": len(paired_errors),
                                   "missed_reference_boundaries": len(ref_boundaries) - len(paired_errors),
                                   "extra_hypothesis_boundaries": len(remaining),
                                   "excluded_overlap_seconds": excluded_overlap,
                                   "status": "not_evaluable_reference_overlap" if excluded_overlap else
                                            ("evaluated" if paired_errors else
                                            ("not_evaluable_no_output_boundaries" if ref_boundaries else
                                             "not_evaluable_no_reference_boundaries"))},
    }


def _hotword_metrics(annotation: dict[str, Any], output: dict[str, Any], ref: list[dict[str, Any]],
                     hyp: list[dict[str, Any]] | None) -> dict[str, Any]:
    targets = annotation.get("hotword_targets")
    enabled = output.get("hotwords_enabled")
    if not isinstance(targets, list) or not isinstance(enabled, bool) or hyp is None:
        return {"status": "not_evaluable_missing_annotations_or_output"}
    gold, decoded = "".join(s["_text"] for s in ref), "".join(s["_text"] for s in hyp)
    present_gold = [w for w in targets if isinstance(w, str) and w and w in gold]
    absent_gold = [w for w in targets if isinstance(w, str) and w and w not in gold]
    hits = sum(w in decoded for w in present_gold)
    insertions = sum(w in decoded for w in absent_gold)
    return {"status": "evaluated", "mode": "enabled" if enabled else "disabled",
            "target_hits": hits, "targets_present_in_reference": len(present_gold),
            "target_hit_rate": hits / len(present_gold) if present_gold else None,
            "false_insertions": insertions, "targets_absent_from_reference": len(absent_gold),
            "false_insertion_rate": insertions / len(absent_gold) if absent_gold else None}


def evaluate(annotation: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(annotation.get("segments"), list):
        raise EvaluationError("annotation must include a segments array")
    ref = _segments(annotation, "annotation")
    hyp = _segments(output, "output")
    assert ref is not None
    for i, segment in enumerate(ref):
        if not isinstance(segment.get("text"), str) or segment.get("lang") not in ("zh", "en"):
            raise EvaluationError(f"annotation.segments[{i}] requires text and lang ('zh' or 'en')")
        if not isinstance(segment.get("speaker_id"), str) or not segment["speaker_id"]:
            raise EvaluationError(f"annotation.segments[{i}] requires speaker_id")
    return {"schema_version": 1, "metrics": {
        **_transcript_metrics(ref, hyp),
        **_speaker_metrics(ref, hyp),
        "hotwords": _hotword_metrics(annotation, output, ref, hyp),
    }}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", required=True, type=Path, help="human-labelled JSON")
    parser.add_argument("--output", required=True, type=Path, help="platform result JSON")
    parser.add_argument("--out", type=Path, help="write report JSON here; default stdout")
    args = parser.parse_args(argv)
    try:
        report = evaluate(_load(args.annotation), _load(args.output))
    except EvaluationError as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
