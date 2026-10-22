# ASR and speaker recognition platform

A local FastAPI service for streaming transcription, file transcription, speaker diarization and registration, and hotword sets. The backend requires Python 3.11 and the models listed in [`models/manifest.json`](models/manifest.json). Model binaries and recordings are downloaded or supplied locally; they are excluded from Git and the container image.

## macOS setup

Install Python 3.11 and FFmpeg (`brew install ffmpeg`) for decoding compressed audio formats. From the repository root, install the application and fetch the tested model files:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python models/download.py online-small
python models/download.py offline-multi
python models/download.py 3dspeaker-eres2net-zh-cn
python models/download.py pyannote-segmentation-3-0
```

Set model paths and start the service on localhost:

```bash
export ASR_ONLINE_MODEL_DIR=models/online-small
export ASR_OFFLINE_MODEL_DIR=models/offline-multi
export ASR_SPEAKER_MODEL_PATH=models/speaker/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
export ASR_SPEAKER_MODEL_VERSION=3dspeaker-eres2net-base-zh-cn-v1
export ASR_DIARIZATION_MODEL_PATH=models/diarization/model.int8.onnx
export ASR_DATA_DIR=data
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

The version value above labels the selected embedding artifact in stored profiles. Keep it unchanged for those profile vectors; if you replace the model, set a new version and re-enroll profiles. The service stores the SQLite database and uploaded audio under `ASR_DATA_DIR`.

Open <http://127.0.0.1:8000/health> for the health check and <http://127.0.0.1:8000/docs> for the API.

## Linux container setup

Install Docker Engine with the Compose plugin. Download the model files on the host as shown above, then build and start the container:

```bash
python3.11 -m pip install -e .
python3.11 models/download.py online-small
python3.11 models/download.py offline-multi
python3.11 models/download.py 3dspeaker-eres2net-zh-cn
python3.11 models/download.py pyannote-segmentation-3-0
docker compose up --build
```

The compose service mounts model directories read-only and persists state in `./data`. The published port binds to `127.0.0.1:8000` by default. Model weights are not copied into the Docker image. To serve other machines, place the service behind the deployment's authentication and TLS layer and change the host port binding deliberately.

## Model and license notes

See [`models/README.md`](models/README.md) for model provenance, exact SHA-256 pins, download behavior, and license review guidance. The model pins identify the tested files; they do not grant permission to redistribute those files. The Pyannote model's upstream archive includes a `LICENSE` file, which the downloader preserves alongside the model.

## Development

Run the test suite with:

```bash
python -m pytest
```

The runtime package and development tools are pinned in `pyproject.toml` and `uv.lock`.
