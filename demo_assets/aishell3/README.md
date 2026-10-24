# 可随 Demo 分发的 AISHELL-3 普通话样本

此目录包含来自 AISHELL-3 官方测试集的少量原始普通话音频，以及两位不同说话人的注册样本和 A/B/A/B 拼接 Demo。OpenSLR SLR93 明确列出 Apache License 2.0；AISHELL/AISHELL-3 Hugging Face 数据集卡同样标注 `apache-2.0`。这里只从固定仓库 revision 下载八个独立测试 WAV，不下载 19 GB 全量归档。归属声明、论文引用和变更说明见 [NOTICE.txt](NOTICE.txt)，许可证原文见 [LICENSE-Apache-2.0.txt](LICENSE-Apache-2.0.txt)。

## 文件

- `registration/speaker-a.wav`、`registration/speaker-b.wav`：各由同一数据集说话人的两条独立录音完整拼接，用作声纹注册上传样本。
- `two-speaker-abab.wav`：四条源录音按 A、B、A、B 顺序拼接，片段之间插入 350 ms 数字静音。
- `source/SSB0686/`、`source/SSB0394/`：八条未经裁剪或重采样的原始 WAV。
- `source_selection.json`：选择的说话人/片段 ID、原始字符与拼音行、参考文字、固定 revision 和源 WAV 哈希。
- `manifest.json`：重建产物的 SHA-256、文件大小、时间轴、注册音频来源及处理方式。

时间轴的片段区间对应整条来源 WAV 在拼接文件中的起止位置；源文件可能包含前后静音，不能当成精确的声学换人时刻。参考文字从 AISHELL-3 `test/content.txt` 的字符/拼音交替行去掉拼音后生成，没有人工改写，也没有模型预测结果。A/B 是匿名示例标签，不表示已识别录音者身份或证明说话人分离准确率。

## 复现和校验

从仓库根目录运行：

```sh
python3 scripts/prepare_demo_audio.py
```

此命令完全离线重建两个注册文件、A/B/A/B Demo 音频和 Manifest。若原始 WAV 缺失，可只下载清单中的八个小文件：

```sh
python3 scripts/prepare_demo_audio.py --download-missing
```

联网核对所选逐句文本是否与官方 `test/content.txt` 一致（该文本表约 2.7 MB，不下载语音大归档）：

```sh
python3 scripts/prepare_demo_audio.py --verify-source-transcripts
```

所有源 WAV 均按固定 Hugging Face revision 和 SHA-256 校验；生成过程不重采样、不裁切、不改增益或滤波。重建输出会覆盖对应的三个小型生成文件和 `manifest.json`。
