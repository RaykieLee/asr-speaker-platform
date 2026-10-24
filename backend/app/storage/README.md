# 本地存储备份、恢复与任务音频保留期

本地部署将 SQLite 与音频文件放在同一数据目录：默认 `ASR_DATA_DIR=./data`，数据库为 `platform.sqlite3`，FileStore 为 `files/`。维护命令从项目根目录执行；也可通过 `--data-dir`、`--database` 和 `--files` 指定位置。

## 创建与核对备份

```sh
python scripts/storage_maintenance.py backup \
  --data-dir data \
  --output /secure-backups/asr-2026-10-23
python scripts/storage_maintenance.py verify \
  --backup /secure-backups/asr-2026-10-23
```

备份使用 SQLite online backup API 获取一致数据库快照，覆盖 WAL 模式下的提交数据。输出目录包含 `database.sqlite3`、`files/` 和 `manifest.json`。Manifest 记录数据库及每个被引用文件的大小、SHA-256 和数据库引用数；verify 会复核哈希、SQLite `integrity_check`，并确认快照中的每个路径引用都能在备份文件清单中找到。

默认保留 30 天的任务原音频：备份中的副本数据库会将已过期终态任务的 `file_path` 清空，结果、片段和注册样本不受影响；尚未完成的任务音频、30 天内终态任务音频及所有仍被注册样本引用的文件会备份。该过期处理只修改备份副本，不修改运行中的应用数据库。可用 `--retention-days N` 调整期限。

备份目录必须不存在；程序先写入同一父目录下的暂存目录，验证后再原子重命名。备份可能包含敏感音频、人员声纹样本和 embedding。工具不负责加密；部署方必须将备份保存在访问受限、加密的磁盘/卷或加密备份系统中，并单独管理密钥。不要将备份提交到 Git。

## 恢复

恢复前先核对 Manifest，再恢复到一个新的目录：

```sh
python scripts/storage_maintenance.py verify --backup /secure-backups/asr-2026-10-23
python scripts/storage_maintenance.py restore \
  --backup /secure-backups/asr-2026-10-23 \
  --to /srv/asr-restored-data
```

恢复拒绝覆盖已存在目标目录；恢复过程会再次验证复制后的数据库和文件。确认恢复结果后，将服务的 `ASR_DATA_DIR` 指向该新目录并启动服务。目录在同一文件系统内通过原子重命名发布；恢复不会删除或替换现有应用数据。

## 原始任务音频清理

先预览（默认就是 dry-run）：

```sh
python scripts/storage_maintenance.py cleanup --data-dir data
```

确认计划后显式应用：

```sh
python scripts/storage_maintenance.py cleanup --data-dir data --retention-days 30 --apply
```

只处理 `succeeded`、`completed`、`complete`、`failed`、`cancelled` 或 `canceled` 等终态任务，并按 `finished_at` 判断是否早于期限。未完成或没有可解析完成时间的任务不会删除。清理先提交 `file_path` 清空，再在第二个 SQLite 写事务中复核引用并删除文件；共享路径仍被其他任务或 `speaker_samples.audio_path` 使用时会保留。若进程在两步之间退出，可能遗留没有数据库引用的文件；当前清理工具只扫描仍有 `file_path` 的任务，不会自动发现这类孤儿。需通过备份 manifest 或部署方的文件对账流程识别和清理，不能直接重跑本工具来清除。任务、识别结果、人工修正和转写片段都会保留。删除或安全校验失败会逐项列在 JSON 输出中，此时任务引用已清空、文件保留，并以非零退出码结束。

工具针对单用户本机部署：维护工具、应用服务和数据目录应由同一可信账户运行，`ASR_DATA_DIR`、备份目录及其父目录应仅允许该账户写入（Unix 系统建议目录权限为 `700`）。它会拒绝检查时发现的路径符号链接，但纯标准库路径检查与 unlink 之间仍存在本机并发篡改的 TOCTOU 窗口；不防御能同时修改数据目录的恶意本地进程。SQLite 最终写事务会阻止其他数据库写入穿过文件删除前的引用复核。定期备份是恢复点，单独备份期间没有跨文件系统事务；若进程在文件复制中断电，暂存目录会保留为无效临时目录，不会成为正式备份，人工确认后可清理该隐藏 `.tmp-*` 目录。

## 定期运行建议

由部署方通过 cron、launchd 或系统任务调度每日先 dry-run 清理、检查输出，再按策略应用清理；备份按组织的 RPO/RTO 频率执行并异机/异卷加密保存。定期执行 `verify`，并在隔离新目录进行恢复演练。先检查维护命令的 JSON 输出和退出状态，不要将任务音频或备份内容写进公共日志。

存储权限、macOS FileVault 与 Linux LUKS 部署验收见 [SECURITY.md](SECURITY.md)。权限限制不代表静态加密。
