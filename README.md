# 声笺：本地 ASR 与说话人识别平台

基于 FastAPI 与 sherpa-onnx 的本地语音服务，支持 WebSocket 流式识别、异步文件转写、热词集编辑、声纹注册及说话人分段。提供无需前端构建的网页 Demo 和并发压测工具。总体设计见 [详细设计与实施计划.md](详细设计与实施计划.md)，实施进度见 [TASKS.md](TASKS.md)。

后端需要 Python 3.11 和 [`models/manifest.json`](models/manifest.json) 中固定版本的模型。模型权重、录音、声纹样本和运行数据均保存在本机，排除在 Git 与容器镜像之外。模型来源、校验与许可说明见 [models/README.md](models/README.md)。

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

Speaker enrollment uploads are limited to 64 MiB and 120 seconds by default. Override with `ASR_SPEAKER_SAMPLE_MAX_BYTES` (positive integer) and `ASR_SPEAKER_SAMPLE_MAX_DURATION_SECONDS` (positive finite number). The multipart request body also has a hard cap before form parsing, so missing or incorrect `Content-Length` cannot bypass the limit. Oversized files and overlong samples return HTTP 413 before decoding/embedding. WAV duration is checked from its header; other formats require `ffprobe` (installed with FFmpeg) for a pre-decode duration check.

`/health` 是轻量存活检查：进程能响应时返回 200，不检查模型。`/ready` 是就绪检查：在线/离线 ASR、声纹嵌入、离线分段模型及存储初始化全部可用时返回 200，否则返回 503；响应只包含组件布尔状态，不含任务文字或音频。受鉴权保护的 `/api/v1/status` 提供组件可用性、实时连接数和文件任务队列负载摘要。Docker Compose 使用 `/ready` 判断容器就绪；启动/配置模型时可用 `/health` 区分进程存活与模型未就绪。也可打开 [交互式 API 文档](http://127.0.0.1:8000/docs)。

## 网页 Demo

在另一个终端运行 `node frontend/server.mjs`，打开 [http://127.0.0.1:4173](http://127.0.0.1:4173)。页面可进行实时录音、上传文件、查看说话人时间轴、人工修正说话人标签、下载 JSON/TXT/SRT、管理声纹人员和编辑版本化热词。双麦立体声文件可选择左右声道分别识别，结果显示声道号。若后端使用了其他端口，设置 `ASR_BACKEND_URL=http://127.0.0.1:端口` 后启动网页代理；若设置了 `ASR_API_TOKEN`，在页面顶部填写相同的 token。详见 [frontend/README.md](frontend/README.md)。

运行 `python demo/prepare_samples.py` 可从本机已验证的模型归档中生成两声音流程样本，再按 [demo/README.md](demo/README.md) 完成演示。样本是流程检查资料，不能代替业务录音的准确率评估。

## 压测与容量

压测工具支持实时、文件及混合负载，并输出会话、延迟、错误、进程 CPU 和内存 JSON/CSV 报告。输入必须为单声道 16 kHz PCM16 WAV。示例：

```bash
python -m loadtest --base-url http://127.0.0.1:8000 \
  --wav path/to/authorized-16k-mono.wav --mode realtime \
  --concurrency 1,2,4,8 --frame-ms 100 --repeats 60 \
  --server-pid 12345 --output data/baseline/my-run
```

将示例中的 `12345` 换成实际服务进程 PID。参数、验收门槛及离线 HTML 图表生成方法见 [loadtest/README.md](loadtest/README.md)。本机基线见 [reports/mac-baseline-2026-10-23.md](reports/mac-baseline-2026-10-23.md)；目标部署机器上的并发容量须在相同模型、代表性录音、热词和声纹规模下重新测量。

人工标注的质量基线可用 [质量评估工具](evaluation/README.md) 计算 CER/WER、说话人时间误差、换人边界、注册命中/未知误认和热词启停对照。没有相应人工标注时工具会标记“不可评估”。

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

本机 SQLite 和音频目录的备份、校验、恢复及 30 天任务原音频清理见 [存储维护说明](backend/app/storage/README.md)。备份中可能包含声纹与录音，应由部署方放在加密且限制访问的位置。

## 模型与许可

模型清单固定了测试文件的来源与 SHA-256，但不授予再分发许可。下载器保留 Pyannote 归档中的 LICENSE；使用前应核对各模型的适用许可，详见 [models/README.md](models/README.md)。

## 开发验证

Run the test suite with:

```bash
python -m pytest
node --test frontend/logic.test.mjs
```

运行依赖与开发工具锁定在 `pyproject.toml` 和 `uv.lock`。后端 HTTP/WebSocket 接口及返回结构以运行时 `/docs` 和实现为准。
