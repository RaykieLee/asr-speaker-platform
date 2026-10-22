"""Build correction-aware transcription exports without mutating model output."""
from __future__ import annotations

import json
from typing import Any
from unicodedata import category


class ExportFormatError(ValueError):
    """The requested export format is not supported."""


def _mapping_index(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["local_speaker_id"]): item for item in items
            if item.get("local_speaker_id")}


def build_export_payload(job: dict[str, Any], database: Any) -> dict[str, Any]:
    """Return raw prediction, manual override, and effective speaker per segment."""
    metadata = job.get("result_metadata") or {}
    predictions = _mapping_index(metadata.get("speaker_mappings") or [])
    local_ids = list(dict.fromkeys(
        str(segment["local_speaker_id"]) for segment in job.get("segments", [])
        if segment.get("local_speaker_id")
    ))
    for local_id in predictions:
        if local_id not in local_ids:
            local_ids.append(local_id)

    overrides: dict[str, dict[str, Any]] = {}
    effective: dict[str, dict[str, Any]] = {}
    for local_id in local_ids:
        prediction = predictions.get(local_id, {})
        override = database.get_mapping_override(job["id"], local_id)
        if override:
            overrides[local_id] = override
        override_person = (database.get_speaker(override["registered_speaker_id"])
                          if override and override.get("registered_speaker_id") else None)
        predicted_person = (database.get_speaker(prediction["registered_speaker_id"])
                           if prediction.get("registered_speaker_id") else None)
        if override:
            display_name = override.get("display_name") or (
                override_person.get("name") if override_person else None
            ) or "未知说话人"
            effective[local_id] = {
                "local_speaker_id": local_id,
                "registered_speaker_id": override.get("registered_speaker_id"),
                "display_name": display_name,
                "source": "manual",
            }
        else:
            effective[local_id] = {
                "local_speaker_id": local_id,
                "registered_speaker_id": prediction.get("registered_speaker_id"),
                "display_name": prediction.get("speaker_name") or (
                    predicted_person.get("name") if predicted_person else None
                ) or f"未知说话人 {len(effective) + 1}",
                "source": "model" if prediction else "unknown",
            }

    segments: list[dict[str, Any]] = []
    for raw in job.get("segments", []):
        local_id = raw.get("local_speaker_id")
        local_id = str(local_id) if local_id else None
        prediction = predictions.get(local_id, {}) if local_id else {}
        current_override = overrides.get(local_id) if local_id else None
        segment = {
            "segment_id": raw["segment_id"],
            "start_seconds": float(raw["start_seconds"]),
            "end_seconds": float(raw["end_seconds"]),
            "text": raw.get("text", ""),
            "local_speaker_id": local_id,
            "registered_speaker_id": raw.get("registered_speaker_id"),
            "overlap": bool(raw.get("overlap", False)),
            "revision": raw.get("revision", 1),
            "metadata": raw.get("metadata") or {},
            "original_prediction": {
                "registered_speaker_id": prediction.get("registered_speaker_id",
                                                        raw.get("registered_speaker_id")),
                "speaker_name": prediction.get("speaker_name") or (raw.get("metadata") or {}).get("speaker_name"),
                "match_status": prediction.get("match_status") or (raw.get("metadata") or {}).get("speaker_match_status"),
            },
            "correction": current_override,
            "speaker": effective.get(local_id) if local_id else None,
        }
        segments.append(segment)

    return {
        "job_id": job["id"],
        "status": job.get("status"),
        "duration_seconds": (metadata.get("duration") if metadata.get("duration") is not None
                             else max((item["end_seconds"] for item in segments), default=0.0)),
        "model_version": job.get("model_version"),
        "speaker_diarization": metadata.get("speaker_diarization", "disabled"),
        "speaker_mappings": [effective[key] for key in local_ids],
        "segments": segments,
    }


def render_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _timestamp(seconds: float, *, srt: bool) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    separator = "," if srt else "."
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}"


def _safe_label(value: Any) -> str:
    """Prevent persisted legacy labels from injecting extra export lines."""
    cleaned = "".join(" " if category(character) in {"Cc", "Zl", "Zp"} else character
                      for character in str(value))
    return " ".join(cleaned.split())


def render_txt(payload: dict[str, Any]) -> str:
    lines = []
    for segment in payload["segments"]:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        speaker = segment.get("speaker")
        label = f"[{_safe_label(speaker['display_name'])}] " if speaker else ""
        lines.append(
            f"[{_timestamp(segment['start_seconds'], srt=False)} --> "
            f"{_timestamp(segment['end_seconds'], srt=False)}] {label}{text}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def render_srt(payload: dict[str, Any]) -> str:
    blocks = []
    for segment in payload["segments"]:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        number = len(blocks) + 1
        speaker = segment.get("speaker")
        if speaker:
            text = f"[{_safe_label(speaker['display_name'])}] {text}"
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        blocks.append(
            f"{number}\n{_timestamp(segment['start_seconds'], srt=True)} --> "
            f"{_timestamp(segment['end_seconds'], srt=True)}\n{text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")
