"""Public, privacy-limited response shapes for transcription task routes."""
from __future__ import annotations

import math
from typing import Any


_SEGMENT_METADATA_FIELDS = {
    "channel_index",
    "speaker_name",
    "speaker_match_status",
    "cross_channel_overlap",
    "duplicate_overlap_text_suppressed",
}
_MAPPING_FIELDS = {
    "local_speaker_id",
    "registered_speaker_id",
    "speaker_name",
    "match_status",
    "channel_index",
    "display_name",
    "source",
}
_ERROR_MESSAGES = {
    "model_unavailable": "识别模型暂不可用，请稍后重试",
    "invalid_audio": "音频文件无法识别，请检查格式与内容",
    "recognition_error": "文件识别失败，请稍后重试或联系管理员",
}


_STRING_LIMITS = {
    "job_id": 256, "segment_id": 256, "local_speaker_id": 256,
    "registered_speaker_id": 256, "speaker_name": 256, "display_name": 256,
    "match_status": 64, "speaker_match_status": 64, "source": 32,
    "status": 32, "model_version": 256, "hotword_set_id": 256,
    "channel_mode": 32, "speaker_diarization": 32, "text": 1_000_000,
    "created_at": 64, "updated_at": 64, "started_at": 64, "finished_at": 64,
}
_INTEGER_FIELDS = {"channel_index", "revision"}
_NUMBER_FIELDS = {"start_seconds", "end_seconds", "duration", "duration_seconds", "progress"}
_BOOLEAN_FIELDS = {"overlap", "cross_channel_overlap", "duplicate_overlap_text_suppressed"}


def _public_scalar(key: str, value: Any) -> Any:
    """Keep only bounded scalar values of the expected public type."""
    if key == "segment_id":
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        if isinstance(value, str) and len(value) <= _STRING_LIMITS[key]:
            return value
        return None
    if key in _STRING_LIMITS:
        if not isinstance(value, str) or len(value) > _STRING_LIMITS[key]:
            return None
        return value
    if key in _INTEGER_FIELDS:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        if value < 0 or value > 2**63 - 1 or (key == "channel_index" and value > 64):
            return None
        return value
    if key in _NUMBER_FIELDS:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        if not math.isfinite(number) or number < 0 or number > 1e12:
            return None
        return number
    if key in _BOOLEAN_FIELDS:
        return value if isinstance(value, bool) else None
    if key in {"hotword_set_version", "hotword_version"}:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value
    return None


def _pick(value: dict[str, Any], fields: set[str]) -> dict[str, Any]:
    result = {}
    for key in fields:
        if key in value:
            safe = _public_scalar(key, value[key])
            if safe is not None:
                result[key] = safe
    return result


def _public_mapping(value: Any) -> dict[str, Any] | None:
    return _pick(value, _MAPPING_FIELDS) if isinstance(value, dict) else None


def _public_segment(value: Any, *, effective: bool = False) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    fields = {
        "segment_id", "start_seconds", "end_seconds", "text", "local_speaker_id",
        "registered_speaker_id", "overlap", "revision",
    }
    result = _pick(value, fields)
    metadata = value.get("metadata")
    result["metadata"] = _pick(metadata, _SEGMENT_METADATA_FIELDS) if isinstance(metadata, dict) else {}
    if effective:
        for key in ("original_prediction", "correction", "speaker"):
            nested = value.get(key)
            if not isinstance(nested, dict):
                result[key] = None
                continue
            if key == "original_prediction":
                result[key] = _pick(nested, {"registered_speaker_id", "speaker_name", "match_status"})
            else:
                result[key] = _pick(nested, {
                    "local_speaker_id", "registered_speaker_id", "display_name", "source",
                })
    return result


def _public_result_metadata(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    fields = {
        "duration", "duration_seconds", "text", "hotword_set_id", "hotword_set_version",
        "hotword_version", "channel_mode", "speaker_diarization", "model_version",
    }
    result = _pick(value, fields)
    mappings = value.get("speaker_mappings")
    if isinstance(mappings, list):
        result["speaker_mappings"] = [mapped for item in mappings
                                      if (mapped := _public_mapping(item)) is not None]
    return result


def public_task_response(job: dict[str, Any], *, status_url: str,
                         effective_segments: list[dict[str, Any]] | None = None,
                         speaker_mappings: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build the stable web/API shape without serializing storage columns."""
    raw_job_id = job.get("job_id", job.get("id", ""))
    job_id = _public_scalar("job_id", raw_job_id)
    if job_id is None:
        job_id = ""
    metadata = job.get("result_metadata")
    result: dict[str, Any] = {
        "job_id": job_id,
        "id": job_id,  # Compatibility alias used by older clients.
        "status": _public_scalar("status", job.get("status")),
        "status_url": status_url if isinstance(status_url, str) and len(status_url) <= 1024 else "",
    }
    for key in ("progress", "created_at", "updated_at", "started_at", "finished_at"):
        if key in job:
            safe = _public_scalar(key, job[key])
            if safe is not None:
                result[key] = safe
    result["model_version"] = _public_scalar("model_version", job.get("model_version")) or (
        metadata.get("model_version") if isinstance(metadata, dict) else None
    )
    result["model_version"] = _public_scalar("model_version", result["model_version"])
    hotword_version = job.get("hotword_version")
    if hotword_version is None and isinstance(metadata, dict):
        hotword_version = metadata.get("hotword_set_version", metadata.get("hotword_version"))
    result["hotword_set_id"] = _public_scalar("hotword_set_id", job.get("hotword_set_id")) or (
        metadata.get("hotword_set_id") if isinstance(metadata, dict) else None
    )
    result["hotword_set_id"] = _public_scalar("hotword_set_id", result["hotword_set_id"])
    hotword_version = _public_scalar("hotword_set_version", hotword_version)
    result["hotword_set_version"] = hotword_version
    result["hotword_version"] = hotword_version  # Compatibility alias.

    status = job.get("status")
    code = job.get("error_code")
    if status == "failed":
        public_code = code if isinstance(code, str) and code in _ERROR_MESSAGES else "recognition_error"
        result["error_code"] = public_code
        result["error_message"] = _ERROR_MESSAGES[public_code]

    segments = job.get("segments")
    if isinstance(segments, list):
        result["segments"] = [item for value in segments
                               if (item := _public_segment(value)) is not None]
    safe_metadata = _public_result_metadata(metadata)
    if safe_metadata is not None:
        result["result_metadata"] = safe_metadata
    if effective_segments is not None:
        result["effective_segments"] = [item for value in effective_segments
                                        if (item := _public_segment(value, effective=True)) is not None]
    if speaker_mappings is not None:
        result["speaker_mappings"] = [item for value in speaker_mappings
                                       if (item := _public_mapping(value)) is not None]
    return result


def public_export_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Project export data through the same scalar and nested allowlists as GET."""
    safe_segments = []
    for segment in payload.get("segments", []):
        projected = _public_segment(segment, effective=True)
        if projected is not None:
            safe_segments.append(projected)
    mappings = payload.get("speaker_mappings", [])
    return {
        "job_id": _public_scalar("job_id", payload.get("job_id")) or "",
        "status": _public_scalar("status", payload.get("status")),
        "duration_seconds": _public_scalar("duration_seconds", payload.get("duration_seconds")),
        "model_version": _public_scalar("model_version", payload.get("model_version")),
        "speaker_diarization": _public_scalar("speaker_diarization", payload.get("speaker_diarization")),
        "speaker_mappings": [safe for item in mappings
                             if (safe := _public_mapping(item)) is not None]
        if isinstance(mappings, list) else [],
        "segments": safe_segments,
    }


def public_correction_response(job_id: str, local_id: str, prediction: Any,
                               correction: Any) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "local_speaker_id": local_id,
        "original_prediction": _pick(prediction, {"local_speaker_id", "registered_speaker_id",
                                                   "speaker_name", "match_status"})
        if isinstance(prediction, dict) else {},
        "correction": _pick(correction, {"local_speaker_id", "registered_speaker_id", "display_name"})
        if isinstance(correction, dict) else None,
    }
