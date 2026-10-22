# Model assets

Model weights are downloaded at setup time and are not stored in Git. The pinned inventory is [`manifest.json`](manifest.json). It records the tested sherpa-onnx release, upstream artifact URL, archive SHA-256, selected archive members, and each extracted file's SHA-256.

The model choices match the runtime smoke checks:

| Purpose | Manifest ID | Tested artifact |
| --- | --- | --- |
| Streaming ASR | `online-small` | sherpa-onnx small bilingual Zipformer, 2023-02-16 |
| File ASR | `offline-multi` | sherpa-onnx Zipformer multi-zh-hans, 2023-09-02 |
| Speaker embedding | `3dspeaker-eres2net-zh-cn` | 3D-Speaker ERes2Net Base, 16 kHz |
| Offline speaker segmentation | `pyannote-segmentation-3-0` | sherpa-onnx Pyannote segmentation 3.0, int8 |

Install the project dependencies first so `sentencepiece` is available, then download each model:

```bash
python3.11 -m pip install -e .
python3.11 models/download.py online-small
python3.11 models/download.py offline-multi
python3.11 models/download.py 3dspeaker-eres2net-zh-cn
python3.11 models/download.py pyannote-segmentation-3-0
```

The downloader stores source archives under `models/cache/`, verifies their pinned SHA-256, and extracts only listed regular-file members. It rejects symbolic links, hard links, duplicate members, path traversal, size mismatches, and file checksum mismatches. For the two SentencePiece ASR models it creates `bpe.vocab` from the pinned `bpe.model` and verifies the generated file against the manifest. `--cache-dir` and `--destination` can point to other directories; this is useful for a shared model cache or a temporary verification install.

For example, use an existing cache and write into a temporary models directory:

```bash
python3.11 models/download.py offline-multi \
  --cache-dir /path/to/model-cache \
  --destination /tmp/asr-models
```

The SHA-256 values pin the exact archives and files used by the integration tests. They are integrity checks, not upstream signatures. Review the upstream terms for each model before use or redistribution. The segmentation download retains the `LICENSE` and `README.md` shipped in its upstream archive; the manifest does not make a license determination for the other model artifacts.

## Upstream sources

- Online Zipformer: [sherpa-onnx online Zipformer model documentation](https://github.com/k2-fsa/sherpa/blob/master/docs/source/onnx/pretrained_models/online-transducer/zipformer-transducer-models.rst)
- Offline Zipformer: [sherpa-onnx offline Zipformer model documentation](https://github.com/k2-fsa/sherpa/blob/master/docs/source/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.rst)
- Pyannote segmentation: [sherpa-onnx speaker diarization model documentation](https://github.com/k2-fsa/sherpa/blob/master/docs/source/onnx/speaker-diarization/models.rst)
- 3D-Speaker embedding: [sherpa-onnx speaker embedding release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models)

The model archive may contain sample recordings and extra precision variants. They are not extracted by this project's downloader.
