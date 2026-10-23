import json

import pytest

from evaluation.quality import EvaluationError, evaluate, main


def fixture_data():
    annotation = {
        "segments": [
            {"start": 0, "end": 1, "text": "你好", "lang": "zh", "speaker_id": "alice", "registered_identity": "spk-a"},
            {"start": 1, "end": 2, "text": "hello world", "lang": "en", "speaker_id": "unknown", "registered_identity": None},
        ],
        "hotword_targets": ["你", "小行星"],
    }
    output = {
        "hotwords_enabled": True,
        "segments": [
            {"start_seconds": 0, "end_seconds": 1, "text": "你号", "local_speaker_id": "0", "registered_speaker_id": "spk-a"},
            {"start_seconds": 1, "end_seconds": 2, "text": "hello word", "local_speaker_id": "1", "registered_speaker_id": "spk-false"},
        ],
    }
    return annotation, output


def test_metrics_are_computed_and_local_speaker_ids_are_permutation_invariant():
    annotation, output = fixture_data()
    result = evaluate(annotation, output)["metrics"]
    assert result["cer"]["value"] == pytest.approx(0.5)
    assert result["wer"]["value"] == pytest.approx(0.5)
    assert result["der_like"]["value"] == 0
    assert result["registered_hit_rate"]["value"] == 1
    assert result["unknown_false_attribution_rate"]["value"] == 1
    assert result["boundary_error_seconds"]["value"] == 0
    assert result["hotwords"]["target_hit_rate"] == 1
    assert result["hotwords"]["false_insertion_rate"] == 0


def test_missing_output_is_explicitly_not_evaluable():
    annotation, _ = fixture_data()
    result = evaluate(annotation, {})["metrics"]
    assert result["cer"]["status"] == "not_evaluable_missing_output"
    assert result["der_like"]["value"] is None
    assert result["hotwords"]["status"] == "not_evaluable_missing_annotations_or_output"


def test_hotword_enabled_and_disabled_runs_are_reported_separately():
    annotation, output = fixture_data()
    output["hotwords_enabled"] = False
    metrics = evaluate(annotation, output)["metrics"]["hotwords"]
    assert metrics["mode"] == "disabled"
    assert metrics["target_hit_rate"] == 1


def test_annotation_requires_language_and_speaker_labels():
    annotation, output = fixture_data()
    annotation["segments"][0].pop("lang")
    with pytest.raises(EvaluationError, match="requires text and lang"):
        evaluate(annotation, output)
    annotation, output = fixture_data()
    annotation["segments"][0].pop("speaker_id")
    with pytest.raises(EvaluationError, match="requires speaker_id"):
        evaluate(annotation, output)


def test_invalid_interval_is_rejected():
    annotation, output = fixture_data()
    output["segments"][0]["end_seconds"] = -1
    with pytest.raises(EvaluationError, match="finite start/end"):
        evaluate(annotation, output)


def test_missing_speaker_labels_are_not_counted_as_errors_and_missing_change_is_visible():
    annotation, output = fixture_data()
    output["segments"][1]["local_speaker_id"] = None
    metrics = evaluate(annotation, output)["metrics"]
    assert metrics["der_like"]["status"] == "not_evaluable_missing_speaker_labels"

    annotation, output = fixture_data()
    output["segments"] = [output["segments"][0]]
    boundary = evaluate(annotation, output)["metrics"]["boundary_error_seconds"]
    assert boundary["value"] is None
    assert boundary["missed_reference_boundaries"] == 1
    assert boundary["status"] == "not_evaluable_no_output_boundaries"


def test_reference_overlap_is_excluded_and_reported_not_scored_as_single_speaker():
    annotation = {"segments": [
        {"start": 0, "end": 1, "text": "A", "lang": "en", "speaker_id": "A", "registered_identity": "reg-a"},
        {"start": 0.5, "end": 1.5, "text": "B", "lang": "en", "speaker_id": "B", "registered_identity": "reg-b"},
    ]}
    output = {"segments": [
        {"start": 0, "end": 1, "text": "A", "local_speaker_id": "a", "registered_speaker_id": "reg-a"},
        {"start": 0.5, "end": 1.5, "text": "B", "local_speaker_id": "b", "registered_speaker_id": "reg-b"},
    ]}
    metrics = evaluate(annotation, output)["metrics"]
    assert metrics["der_like"]["value"] == 0
    assert metrics["der_like"]["reference_speech_seconds"] == 1
    assert metrics["der_like"]["excluded_overlap_seconds"] == pytest.approx(0.5)
    assert metrics["registered_hit_rate"]["registered_reference_seconds"] == 1
    assert metrics["boundary_error_seconds"]["value"] is None
    assert metrics["boundary_error_seconds"]["status"] == "not_evaluable_reference_overlap"


def test_unequal_speaker_cardinality_matches_best_overlapping_output_id():
    annotation = {"segments": [
        {"start": 0, "end": 10, "text": "x", "lang": "en", "speaker_id": "A", "registered_identity": None},
    ]}
    output = {"segments": [
        {"start": 0, "end": 2, "text": "x", "local_speaker_id": "a"},
        {"start": 2, "end": 10, "text": "x", "local_speaker_id": "z"},
    ]}
    metrics = evaluate(annotation, output)["metrics"]
    assert metrics["der_like"]["error_seconds"] == 2
    assert metrics["der_like"]["value"] == pytest.approx(0.2)


def test_cli_writes_report_without_copying_transcript_text(tmp_path):
    annotation, output = fixture_data()
    a, o, report = tmp_path / "annotation.json", tmp_path / "output.json", tmp_path / "report.json"
    a.write_text(json.dumps(annotation), encoding="utf-8")
    o.write_text(json.dumps(output), encoding="utf-8")
    assert main(["--annotation", str(a), "--output", str(o), "--out", str(report)]) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert "segments" not in payload
    assert "你好" not in report.read_text(encoding="utf-8")
