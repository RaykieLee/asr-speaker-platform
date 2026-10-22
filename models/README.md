# 模型清单与下载

模型权重不进入 Git。`manifest.json` 是候选模型及版本状态的权威清单。当前 sherpa-onnx 运行时版本、模型版本、来源 URL、文件清单和 SHA-256 都标记为“待固定”：它们需要在 P0 根据业务音频、许可证和目标部署环境共同选定。本仓库不会猜测哈希，也不会提供会下载未经校验权重的默认 URL。

固定模型时，为每个模型在 `files` 中填写 `{ "name": "model.onnx", "url": "https://…/model.onnx", "sha256": "64位十六进制值" }`，并记录来源版本、许可证和校验值取得方式。多文件模型（例如 encoder/decoder/joiner/token 表）逐个列出并逐个校验。可用 `python models/download.py <model-id>` 下载；文件先写入 `.part`，仅在 SHA-256 匹配后原子改名。缺 URL 或校验值时脚本明确拒绝。

## sherpa-onnx 热词支持

适配层按识别架构和在线/离线模式检查能力。在线 Transducer 对应 `OnlineRecognizer` 与 `OnlineStream`；离线 Transducer 对应 `OfflineRecognizer` 与 `OfflineStream`。使用热词时，配置需要采用兼容的 `modified_beam_search` 解码，并在创建 stream 时提供热词配置（Python API 的 `create_stream(hotwords)`）。实际可用性须在固定的 sherpa-onnx Python 绑定版本和具体模型组合上集成验证。

SenseVoice 使用 sherpa-onnx 的 SenseVoice 构造接口，不具备相同的原生热词参数。请求热词时适配层抛出 `HotwordsUnsupportedError`；上层应转换为 `HOTWORDS_UNSUPPORTED`，不得静默忽略，也不得用文本后处理冒充热词识别。

缺少模型文件时 `ModelArtifact.require_files()` 抛出包含模型 ID、版本状态、缺失路径和下载文档提示的 `MissingModelError`。这些接口不要求安装 sherpa-onnx 或实际权重，可在轻量环境中验证；加载推理器及模型质量/性能基线属于后续集成任务。
