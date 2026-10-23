# Offline ASR quality evaluation

This tool compares a manually labelled JSON file with one platform result JSON. It reads JSON only: it does not load audio or invoke ASR/speaker models. It reports metrics only where the required reference and output fields exist; missing output is `not_evaluable`, never a fabricated zero error rate.

## Input schema

Human annotation:

```json
{
  "segments": [
    {"start": 0.0, "end": 1.2, "text": "你好", "lang": "zh", "speaker_id": "person-a", "registered_identity": "spk-123"},
    {"start": 1.2, "end": 2.0, "text": "hello", "lang": "en", "speaker_id": "unknown-1", "registered_identity": null}
  ],
  "hotword_targets": ["你好", "产品名"]
}
```

Each reference segment must have finite nonnegative `start`/`end`, `text`, `lang` (`zh` or `en`) and `speaker_id`. Use stable human identity labels for `speaker_id`; use `registered_identity` for the registered identity expected in the platform output, or `null` for a human-labelled unknown speaker. `hotword_targets` is the set of terms to check against the annotated transcript.

The platform result uses `segments` with `start_seconds`/`end_seconds` (or `start`/`end`), `text`, `local_speaker_id` (or `speaker_id`) and optional `registered_speaker_id`, plus boolean `hotwords_enabled` when known. The text, speaker label and times should come from the result under evaluation. Do not insert reference labels into the result to make a metric evaluable.

## Metrics and interpretation

- Chinese CER: character edit distance divided by annotated Chinese characters, ignoring whitespace.
- English WER: case-folded word edit distance divided by annotated English words.
- `der_like`: speech-time fraction misattributed on the union of reference/output interval boundaries. Output-local speaker IDs are optimally mapped to annotated speaker IDs by overlap. This is a simplified non-overlap DER-like score; overlapping speech is not scored as a separate speaker stream and should be reported separately.
- `boundary_error_seconds`: mean absolute timing error for one-to-one greedy nearest pairs of reference/output speaker-change boundaries. The report also lists missed reference and extra output boundaries; if there are no matches, the error value is null and the missed count remains visible.
- Registered hit rate and unknown false-attribution rate are duration-weighted by reference speech time.
- Hotword metrics report target hit rate for terms present in the human transcript and false insertion rate for target terms absent from it. Keep enabled and disabled runs separate and compare reports on the same annotated recording; this is not a causal estimate if the audio or settings differ.

An empty denominator produces `value: null` and a status explaining why. This tool does not set acceptance thresholds or claim quality without labelled samples.

## Run

From the repository root:

```sh
uv run python -m evaluation.quality \
  --annotation data/private-eval/meeting-01.annotation.json \
  --output data/private-eval/meeting-01.result.json \
  --out data/private-eval/meeting-01.metrics.json
```

Store `data/private-eval/` outside Git. Human labels and ASR transcripts can contain personal or confidential speech: use authorized recordings, restrict access, redact names/IDs where appropriate, and do not commit raw recordings, transcripts, or reports containing sensitive metadata. Reports contain aggregate metrics only and do not copy transcript text.
