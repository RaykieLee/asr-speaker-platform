# ASR 并发压测工具

独立客户端为实时 WebSocket、文件任务或混合负载生成可重复的 JSON/CSV 报告。工具只遵循《详细设计与实施计划》草案的 API 约定；在服务端接口完成前，可以先用 fake client/测试验证本地指标与节奏逻辑。报告记录实际观测数据，不会根据开发机结果推断目标机器容量。

## 安装与快速运行

需要 Python 3.11。建议在与服务端分开的客户端机器运行，以免客户端 PCM 发送和解码占用被测服务资源。

```bash
python -m venv .venv-loadtest
source .venv-loadtest/bin/activate
python -m pip install -r loadtest/requirements.txt
# Optional CPU/RSS/system-memory sampling:
python -m pip install -r loadtest/requirements-resource.txt
```

输入 WAV 必须为单声道、16 kHz、16-bit 无压缩 PCM。示例：

```bash
python -m loadtest --base-url http://asr-host:8000 \
  --wav samples/authorized-mono-16k.wav \
  --mode realtime --concurrency 1,2,4,8 --frame-ms 40 --repeats 10 \
  --hotword-set-id hotwords-v3 --speaker-scope team-a \
  --server-pid 12345 --output results/2026-10-23-realtime
```

`--repeats` 循环拼接输入 PCM，让录音片段可持续播放到更长时长（例如 60 秒片段重复 10 次约为 10 分钟）。重复音频不等于丰富的业务样本，应记录循环方式并用经授权的代表性音频。`--mode file` 在每个阶梯档并行提交同档数量的文件任务；`--mode mixed` 每档启动指定路数实时连接，并额外用 `--file-concurrency` 个文件任务同时施压。文件任务在 `POST /api/v1/transcriptions` 创建，然后轮询 `GET /api/v1/transcriptions/{job_id}`。实时模式连接 `/api/v1/stream`，发 `start` JSON、按真实时间节奏发送 PCM 二进制帧，再发 `stop`。

鉴权可使用 `--token`，更推荐 `ASR_API_TOKEN` 环境变量。服务端进程资源采集需要本机可访问对应 PID；不传 PID 时 psutil 若可用则采集压测客户端进程及系统 CPU/内存，并在报告中标出作用范围。未安装 psutil 时跳过资源采样；它不会作为强制依赖装入后端。

## 输出和指标

每次运行产生 `report.json` 和 `sessions.csv`。报告包含配置、机器元数据、输入文件名与 SHA-256、每会话开始时间和 ID、首个 partial、最终文本及说话人时间、错误类型、已发送帧数、客户端发送积压/迟发帧、服务端显式报告的丢帧（如果协议提供）、文件队列/处理耗时、RTF、P50/P95/P99、资源采样以及门槛。文件队列时长优先使用服务端 `queue_seconds`；若服务端没有该字段，则取客户端首次观察到 `running` 的时间作为队列结束近似值。处理时长优先使用服务端 `processing_seconds`；否则按首次观察到 `running` 至终态的客户端时间估算。文件处理小时吞吐使用“成功且处理时长大于零的音频秒数 ÷ 同一批处理秒数”；墙钟小时吞吐使用“成功任务音频秒数 ÷ 整轮墙钟秒数”。秒/秒即小时/小时的比值，失败任务不进入吞吐分子；缺失的处理时长不会按零补入分母。报告含最终转写文本和说话人信息，需按录音数据的保密要求存放与分享。

可将已生成的 `report.json` 转成不依赖网络和额外 Python 包的离线 HTML 图表：

```bash
python -m loadtest.plot results/2026-10-23-realtime/report.json
# 可选指定输出路径
python -m loadtest.plot results/2026-10-23-realtime/report.json --output results/summary.html
```

默认在 JSON 旁生成 `report.html`。图表包含 CPU/RSS 随采样时间的曲线、按并发档位和负载模式计算的成功率、关键时延 P50/P95 表及 P95 图。缺少资源样本或某项时延时标为“不可评估”，不补零。HTML 不包含会话原始转写文本。

服务端没有报告 `dropped_frames` 时，`server_dropped_frames` 保持 `null`，不按零处理。客户端 `client_late_frames` 仅表示发送端落后于音频时间线，不能代替服务端丢帧计数。没有收到 partial/speaker 时延值时，分位数按可观测样本计算；不可观测延迟不会伪造为零。分位数采用线性插值定义。`gates` 给出已设置门槛的 pass/fail/not_evaluable 状态；任何关键指标未知都会使整体状态至少为 not_evaluable。

实时说话人事件按 `segment_id` 聚合，较高 `revision` 覆盖较低版本，同版本按后收到的事件处理。每会话记录最终片段数、带标签片段数、按片段开始时间排序后相邻且均有标签的片段之间的换人次数，以及最终说话人标签事件相对片段 `end` 音频时间的 P50/P95/P99 延迟。没有最终片段事件、没有完成事件或无有效时间戳时，相应统计保留为 `null`；没有可计算的延迟时分位数为 `null`。摘要按并发档位汇总，并列出实际观测到的会话数。以上指标反映事件完整性与时延，不验证说话人身份准确率，也不保存额外的逐段转写或声纹数据。

客户端机器元数据包含系统版本、架构、处理器描述、逻辑 CPU、内存总量（psutil 可用时）和 Python 版本。压测目标机的信息通过 `--target-os`、`--target-cpu-model`、`--target-cpu-cores`、`--target-memory-bytes`、`--target-gpu-model`、`--target-gpu-memory-bytes` 记录；模型版本、推理线程数、Worker 数、热词数量和音频特征也有对应 CLI 参数。未提供的值保留为空，不能误认为客户端配置代表服务端配置。

## 门槛与容量结论

CLI 默认门槛是待业务确认的初始检查值：错误率 ≤1%、首个 partial P95 ≤2 秒、发送 stop 后至 completed 的 P95 ≤8 秒、客户端积压 ≤0.5 秒、服务端丢帧为 0。用 `--max-error-rate`、`--max-first-partial-p95`、`--max-final-p95`、`--max-backlog`、`--max-server-drops` 修改。报告中的 `final_text` 延迟仍从会话开始计时，包含音频播放时间，仅供诊断；它不参与此门槛。门槛随配置写入报告；命令在任一可判定门槛失败时返回非零状态。缺少观测值时显示 `not_evaluable`，不当作门槛已通过，也不推断容量。

“稳定处理 N 路”需要在最终部署机器、业务认可的音频和热词/声纹组合上复测：预热模型后每档至少稳定运行 10 分钟；接近瓶颈时缩小并发步长；连续长稳负载观察内存增长。成功率、延迟、丢帧和资源持续增长均需满足预先约定的门槛。目标机 CPU 型号/核心数、内存、GPU/显存、OS、模型文件版本、推理线程数、Worker 数、热词数、音频样本特征、门槛和完整报告应一起归档。开发机报告不是生产容量依据。本工具不会声称当前机器已测出容量。

纯实时 WebSocket 当前按文档草案要求服务器在 `start` 后发 `ready`/`started`/`session_started`/`start_ack` 确认，并在 `stop` 后发 `completed`。服务端若使用不同事件名，请按正式 OpenAPI/WebSocket 契约调整客户端适配；文件接口需返回 `job_id`，状态需使用 `queued`、`running`、`succeeded`/`failed` 之一。

## 本地逻辑测试

```bash
python -m pytest tests/test_loadtest.py
```

这些测试验证帧节奏、分位数、错误聚合和报告格式，不会连接服务端或作为性能测试结果。

图表解析和输出测试同样只读本地 JSON，不启动模型或压测：

```bash
python -m pytest tests/test_loadtest_plot.py
```
