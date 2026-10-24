# Mac 本机存储加密与权限验收（2026-10-24）

范围：当前开发机的 `data/baseline/runtime` 数据库与音频目录，以及同机 `data/baseline/backup-smoke-20261023` 备份目录。检查器为 [`scripts/storage_security_check.py`](../scripts/storage_security_check.py)，检查项包括 SQLite 主库、存在时的 WAL/SHM、录音树、备份树的文件类型与私有权限，以及数据和备份各自所在卷的静态加密状态。

第一次检查发现运行目录、SQLite 主库/WAL/SHM 和音频/备份文件中有组或其他账户可读取项，整体失败；FileVault 和两个目录所在卷的加密检查已通过。将这两个现有目录树的组/其他账户权限收紧后重跑，所有检查通过，退出码为 0。`fdesetup status` 报告 FileVault 为 On；检查器还通过实际挂载卷信息核对了两个路径的卷归属和加密属性。原始音频、声纹和转写没有写入本报告。

为验证后续新建文件的默认行为，使用更新后的 `SQLiteStore`、`FileStore` 和备份函数创建全新的临时数据树及备份，运行备份完整性校验得到 `valid=true`、`integrity_check=ok`，再运行同一加密/权限检查器，所有项通过。SQLite 主库与音频/备份文件为 `0600`，新建数据和备份目录为 `0700`；测试期间没有 WAL/SHM 文件时，检查器明确显示可选缺失，不将其伪装为已检查通过的文件。

本结果只覆盖当前 Mac 和实际检查的两个目录。另一台目标机器、外置备份卷或容器挂载必须各自重新执行检查；加密密钥的保管和异地备份的加密由部署方按 [存储安全说明](../backend/app/storage/SECURITY.md) 验收。
