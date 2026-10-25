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
- `der_like`: speech-time fraction misattributed on the union of reference/output interval boundaries. Output-local speaker IDs are mapped to annotated speaker IDs by maximum overlap. Reference intervals with more than one active segment are excluded from DER-like and registered/unknown identity denominators, and their duration is reported as `excluded_overlap_seconds`. This remains a simplified DER-like score; it does not evaluate overlapping speech as separate speaker streams.
- `boundary_error_seconds`: mean absolute timing error for one-to-one greedy nearest pairs of reference/output speaker-change boundaries. The report also lists missed reference and extra output boundaries; if there are no matches, the error value is null and the missed count remains visible. If reference overlap exists, boundary scoring is explicitly not evaluable because a single linear speaker timeline is ambiguous.
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

## Reproducible local model replay

`scripts/run_diarization_quality.py` runs the repository's offline Transducer recognizer, Sherpa speaker diarizer, registered-speaker matcher, and `evaluation.quality` against one local WAV and annotation. It writes `result.json`, `metrics.json`, and `run-manifest.json` into the requested output directory (directory mode `0700`; each JSON mode `0600`). The manifest records the actual diarization threshold and speaker count, model version identifiers, absolute input paths and SHA-256 hashes, plus hashes for the result and metrics files. The registration database is opened with SQLite `mode=ro&immutable=1`; it is never initialized or modified, and the command fails if a `-wal` file exists or the main database changes during inference. Supply a static, consistent database snapshot with no SQLite sidecars. SQLite immutable mode avoids creating `-wal`/`-shm` files alongside it. It writes only to `--output-dir`.

The ASR directory must contain exactly one `encoder*.onnx`, `decoder*.onnx`, `joiner*.onnx`, and `tokens.txt`; `bpe.vocab` is used when present. For example, use an AISHELL-4 two-person interval already exported as WAV and manually annotated using the schema above:

```sh
PYTHONPATH=. .venv/bin/python scripts/run_diarization_quality.py \
  --audio /data/aishell4/two-person.wav \
  --annotation /data/aishell4/two-person.annotation.json \
  --registry /var/lib/asr/speakers.sqlite3 \
  --asr-model-dir /opt/asr/models/offline-multi \
  --asr-version offline-multi-pinned-version \
  --segmentation-model /opt/asr/models/pyannote/model.onnx \
  --speaker-model /opt/asr/models/3dspeaker/model.onnx \
  --speaker-model-version 3dspeaker-eres2net-zh-cn-pinned-version \
  --cluster-threshold 0.7 \
  --num-speakers 2 \
  --output-dir /data/aishell4/replay-output
```

Keep source recordings, annotation, registration database, predictions, and generated artifacts outside Git. A rerun is comparable only when its input/model hashes and all manifest parameters match the baseline; the manifest preserves the exact threshold and speaker count for independent inspection. The command exits nonzero and prints the failure to stderr when input validation, model loading, inference, or metric evaluation fails.

## Public AISHELL-4 clip workflow

The helper scripts below accept an AISHELL-4 FLAC recording and its Praat TextGrid. Obtain both files from the official AISHELL-4 distribution and follow that distribution's access and use terms. Keep the downloaded corpus, generated clip, annotations, platform output JSON, and metrics outside Git.

The converter needs the optional `soundfile` and `textgrid` Python packages. They are not runtime dependencies of the service; install them in the evaluation environment as needed (for example, `uv add --dev soundfile textgrid`). NumPy is already a project dependency. By default it selects source channel 0, matching the clips uploaded for this evaluation; use `--channel mixdown` only when the upload was made from an all-channel average. Selection reads only the requested source range, resamples with a NumPy windowed-sinc filter when needed, then writes 16 kHz PCM16 WAV. TextGrid intervals are clipped to that range; times in the annotation are relative to the exported WAV, and the TextGrid tier name is retained as `speaker_id`. `<sil>` and `<#>` markers, whitespace, and Unicode punctuation are removed from reference text. Use the same source channel and time bounds as the upload to reproduce its audio bytes. The supplied `M_R003S02C01.flac` is already 16 kHz; exporting channel 0 over 24–90 seconds reproduces the uploaded `asr-hotword.wav` byte for byte.

```sh
uv run python scripts/prepare_aishell4_eval.py \
  --audio data/datasets/aishell4/M_R003S02C01.flac \
  --textgrid data/datasets/aishell4/M_R003S02C01.TextGrid \
  --start 60 --end 120 \
  --channel 0 \
  --wav /tmp/aishell4-60-120.wav \
  --annotation /tmp/aishell4-60-120.annotation.json \
  --hotwords '商场,客流量'
```

Run the same exported WAV twice with matched model/configuration and the same manually selected hotword list, changing only whether the platform hotword feature is enabled. Save each platform's own returned transcript and speaker IDs as JSON with the output schema above; never add reference text or hotword terms to a prediction. Then compare both outputs:

```sh
uv run python scripts/compare_public_outputs.py \
  --annotation /tmp/aishell4-60-120.annotation.json \
  --enabled /tmp/platform-enabled.json \
  --disabled /tmp/platform-disabled.json \
  --out /tmp/aishell4-comparison.json
```

The comparison calculates Chinese CER over the globally concatenated, start-time-ordered reference and hypothesis text after removing whitespace, punctuation, and the two TextGrid markers. TextGrid intervals from different speakers can overlap, making one global reference order ambiguous. If any reference overlap exists, CER is marked `not_evaluable_overlapping_reference`; the report gives overlapping seconds and the non-overlapping speech coverage ratio. It also includes the existing `evaluation.quality` speaker metrics and, for each run, hotword hits among terms present in the reference and insertions among terms absent from it. The enabled output must contain a non-empty `hotword_set_id`; the disabled output must contain `hotword_set_id: null`. The hotword list is a manually chosen test condition; it is never injected into either platform output. Treat the enabled/disabled difference as a paired observation, not a causal estimate if any other setting or input differs.
