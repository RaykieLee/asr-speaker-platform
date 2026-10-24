from __future__ import annotations

import plistlib
import subprocess
from pathlib import Path

from scripts.storage_security_check import (
    inspect_permissions,
    linux_luks_encrypted,
    macos_volume_encrypted,
    run_checks,
)


def _completed(stdout: bytes = b"", returncode: int = 0) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=b"")


def test_permissions_checks_private_modes_and_tree(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    file_path = root / "audio.bin"
    file_path.write_bytes(b"audio")
    file_path.chmod(0o600)
    assert inspect_permissions(root, required=True, tree=True)[:2] == (True, "private_mode")

    file_path.chmod(0o640)
    assert inspect_permissions(root, required=True, tree=True)[:2] == (False, "group_or_other_access")


def test_permissions_fail_closed_on_symlinks_and_required_missing(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(target)
    assert inspect_permissions(link, required=True)[:2] == (False, "symlink")
    assert inspect_permissions(tmp_path / "missing", required=True)[:2] == (False, "missing")
    assert inspect_permissions(tmp_path / "missing", required=False)[:2] == (True, "optional_missing")


def test_macos_requires_filevault_and_encrypted_volume_for_path(tmp_path: Path) -> None:
    anchor = tmp_path
    encrypted = plistlib.dumps({"MountPoint": str(tmp_path), "FileVault": True})
    df_output = f"Filesystem 512-blocks Used Available Capacity Mounted on\n/dev/disk1s1 100 1 99 1% {tmp_path}\n".encode()
    calls = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        if command[0].endswith("fdesetup"):
            return _completed(b"FileVault is On.\n")
        if command[0].endswith("df"):
            return _completed(df_output)
        return _completed(encrypted)

    assert macos_volume_encrypted(anchor, runner)
    assert len(calls) == 3
    assert calls[0][1:] == ["status"]
    assert calls[1][1:2] == ["-P"]
    assert calls[2][1:3] == ["info", "-plist"]

    def unencrypted_volume(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        if command[0].endswith("fdesetup"):
            return _completed(b"FileVault is On.\n")
        if command[0].endswith("df"):
            return _completed(df_output)
        return _completed(plistlib.dumps({"MountPoint": str(tmp_path), "FileVault": False}))

    assert not macos_volume_encrypted(anchor, unencrypted_volume)

    def wrong_volume(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        if command[0].endswith("fdesetup"):
            return _completed(b"FileVault is On.\n")
        if command[0].endswith("df"):
            return _completed(df_output)
        return _completed(plistlib.dumps({"MountPoint": "/different-volume", "FileVault": True}))

    assert not macos_volume_encrypted(anchor, wrong_volume)

    def wrong_volume(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        if command[0].endswith("fdesetup"):
            return _completed(b"FileVault is On.\n")
        if command[0].endswith("df"):
            return _completed(df_output)
        return _completed(plistlib.dumps({"MountPoint": "/different-volume", "FileVault": True}))

    assert not macos_volume_encrypted(anchor, wrong_volume)


def test_macos_fails_closed_if_filevault_off_or_volume_cannot_be_mapped(tmp_path: Path) -> None:
    def off(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        return _completed(b"FileVault is Off.\n")

    assert not macos_volume_encrypted(tmp_path, off)
    df_output = f"Filesystem 512-blocks Used Available Capacity Mounted on\n/dev/disk1s1 100 1 99 1% {tmp_path}\n".encode()

    def missing_mount(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        if command[0].endswith("fdesetup"):
            return _completed(b"FileVault is On.\n")
        if command[0].endswith("df"):
            return _completed(df_output)
        return _completed(plistlib.dumps({"FileVault": True}))

    assert not macos_volume_encrypted(tmp_path, missing_mount)


def test_linux_requires_luks_mapper_for_the_requested_mount(tmp_path: Path) -> None:
    commands = []

    def luks(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        if "findmnt" in command[0]:
            return _completed(b"/dev/mapper/asrdata\n")
        return _completed(b"/dev/mapper/asrdata is active.\n  type: LUKS2\n  device: /dev/nvme0n1p3\n")

    assert linux_luks_encrypted(tmp_path, luks)
    assert commands[0][-1] == str(tmp_path)
    assert commands[1][-1] == "asrdata"

    def plain_mount(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        if "findmnt" in command[0]:
            return _completed(b"/dev/nvme0n1p3\n")
        raise AssertionError("must not query an unencrypted source")

    assert not linux_luks_encrypted(tmp_path, plain_mount)


def test_linux_fails_closed_when_cryptsetup_does_not_confirm_luks(tmp_path: Path) -> None:
    def unsupported(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        if "findmnt" in command[0]:
            return _completed(b"/dev/mapper/asrdata\n")
        return _completed(b"/dev/mapper/asrdata is active.\n  type: PLAIN\n  device: /dev/loop0\n")

    assert not linux_luks_encrypted(tmp_path, unsupported)


def test_run_checks_requires_backup_and_encryption_checks(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "data"
    backup = tmp_path / "backup"
    data.mkdir(mode=0o700)
    (data / "files").mkdir(mode=0o700)
    (data / "platform.sqlite3").write_bytes(b"sqlite")
    (data / "platform.sqlite3").chmod(0o600)
    backup.mkdir(mode=0o700)
    monkeypatch.setattr("scripts.storage_security_check.macos_volume_encrypted", lambda path: True)
    result = run_checks(data, [backup], platform="Darwin")
    assert result["status"] == "pass"
    assert [item["check"] for item in result["checks"]] == [
        "data_directory_permissions",
        "sqlite_main_permissions",
        "sqlite_wal_permissions",
        "sqlite_shm_permissions",
        "recordings_permissions",
        "backup_1_permissions",
        "data_at_rest_encryption",
        "backup_1_at_rest_encryption",
    ]

    monkeypatch.setattr("scripts.storage_security_check.macos_volume_encrypted", lambda path: False)
    failed = run_checks(data, [backup], platform="Darwin")
    assert failed["status"] == "fail"
    assert all("path" not in item for item in failed["checks"])
