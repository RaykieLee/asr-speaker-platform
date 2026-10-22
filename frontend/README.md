# 声笺本地语音工作台

这是一个无前端构建依赖的本地演示界面，覆盖实时语音、文件转写、声纹样本录入和版本化热词管理。页面不会伪造识别文本或声纹结果；服务端模型未配置时会显示 API 返回的错误。

## 启动

在项目根目录启动后端：

```sh
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

另开终端启动前端代理：

```sh
node frontend/server.mjs
```

打开 [http://127.0.0.1:4173](http://127.0.0.1:4173)。代理默认转发到 `http://127.0.0.1:8000`，可通过 `ASR_BACKEND_URL` 修改；界面也可以填写其他服务地址。麦克风权限要求 localhost 或 HTTPS。

后端设置了 `ASR_API_TOKEN` 时，可在页面顶部填写同一个 token。HTTP 请求会使用 Bearer 认证，实时 WebSocket 会在首条 `start` 消息中发送 token。token 只保存在当前标签页的 session storage 中。

## 模型和功能状态

前端只连接项目 API，不会下载模型。根据后端配置启动相应能力：

- 实时识别需要 online ASR 模型及其配置。
- 文件转写需要 offline ASR 模型及其配置。上传后页面会轮询任务状态，并按服务返回的时间轴、转写片段和说话人标记展示。
- 文件结果支持使用后端导出 API 下载 JSON、TXT、SRT，包含当前有效的说话人标记。
- 文件转写完成后，可在结果下方按本地说话人选择已注册人员或录入显示名称进行纠错。纠错会调用转写任务的说话人 PATCH API；保存成功后重新读取任务的有效片段，再导出即可获得更新后的标签。
- 声纹人员可以在服务可用时创建；样本质量检查和 profile 生成需要 speaker embedding 模型。
- 热词方案使用后端的版本化编译和模型词表兼容检查。
- 声纹分段需要 diarization 模型；没有模型时服务应明确返回 disabled/unavailable，不会由界面推测说话人。

具体环境变量名称与模型路径以仓库根目录的部署说明和 `backend/app/main.py` 为准。服务健康不代表所有模型都已加载，首次调用会显示各功能自己的错误。

## 浏览器测试

前端纯逻辑测试只依赖 Node 内置测试运行器：

```sh
node --test frontend/logic.test.mjs
```
