"""Public, privacy-limited response shapes for transcription task routes."""
from __future__ import annotations

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
}
_ERROR_MESSAGES = {
    "model_unavailable": "识别模型暂不可用，请稍后重试",
    "invalid_audio": "音频文件无法识别，请检查格式与内容",
    "recognition_error": "文件识别失败，请稍后重试或联系管理员",
}


def _pick(value: dict[str, Any], fields: set[str]) -> dict[str, Any]:
    return {key: value[key] for key in fields if key in value}


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
    job_id = str(job.get("job_id", job.get("id", "")))
    metadata = job.get("result_metadata")
    result: dict[str, Any] = {
        "job_id": job_id,
        "id": job_id,  # Compatibility alias used by older clients.
        "status": job.get("status"),
        "status_url": status_url,
    }
    for key in ("progress", "created_at", "updated_at", "started_at", "finished_at"):
        if key in job:
            result[key] = job[key]
    result["model_version"] = job.get("model_version") or (
        metadata.get("model_version") if isinstance(metadata, dict) else None
    )
    hotword_version = job.get("hotword_version")
    if hotword_version is None and isinstance(metadata, dict):
        hotword_version = metadata.get("hotword_set_version", metadata.get("hotword_version"))
    result["hotword_set_id"] = job.get("hotword_set_id") or (
        metadata.get("hotword_set_id") if isinstance(metadata, dict) else None
    )
    result["hotword_set_version"] = hotword_version
    result["hotword_version"] = hotword_version  # Compatibility alias.

    status = job.get("status")
    code = job.get("error_code")
    if status == "failed":
        public_code = code if code in _ERROR_MESSAGES else "recognition_error"
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
