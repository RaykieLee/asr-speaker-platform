# ASR 与说话人识别平台

一个面向本地部署的普通话与英语语音识别平台，目标能力包括实时转写、音频文件转写、说话人分段与声纹映射、热词管理，以及可演示的网页界面。详细范围和实施顺序见[详细设计与实施计划](详细设计与实施计划.md)。

当前仓库处于项目初始化阶段，提供 Python 3.11 后端包、FastAPI 健康检查端点和基础测试。识别引擎、数据存储、网页 Demo 与压测能力将在后续任务中实现。模型文件和录音样本不纳入代码仓库。

## 环境要求

- Python 3.11
- pip

## 安装与运行

在仓库根目录执行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
uvicorn backend.app.main:app --reload
```

打开 <http://127.0.0.1:8000/health> 可查看服务健康状态；交互式 API 文档位于 <http://127.0.0.1:8000/docs>。

## 测试

```bash
python -m pytest
```

运行时和开发依赖已在 `pyproject.toml` 中固定版本。后续任务会逐步补充音频处理、ASR/VAD/声纹模型适配、热词编辑、持久化、网页演示和并发容量测试。

## 设计资料

- [详细设计与实施计划](详细设计与实施计划.md)
- [实现与独立验证任务清单](TASKS.md)
