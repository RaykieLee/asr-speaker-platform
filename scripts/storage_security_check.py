#!/usr/bin/env python3
"""Fail-closed checks for ASR storage permissions and at-rest encryption.

This tool reports only generic check identifiers and statuses. It never prints
filesystem paths, volume names, device names, database contents, or file names.
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable


Runner = Callable[..., subprocess.CompletedProcess[bytes]]


def _run(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(command, check=False, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, timeout=10)


def _check(name: str, ok: bool, code: str) -> dict[str, str | bool]:
    return {"check": name, "status": "pass" if ok else "fail", "code": code}


def _existing_anchor(path: Path) -> Path:
    candidate = path.expanduser().absolute()
    while not candidate.exists() and not candidate.is_symlink():
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    return candidate


def inspect_permissions(path: Path, *, required: bool, tree: bool = False) -> tuple[bool, str, int]:
    """Require private POSIX mode bits and reject symlinks in checked trees."""
    if not path.exists() and not path.is_symlink():
        return (not required, "missing" if required else "optional_missing", 0)
    count = 0
    pending = [path]
    while pending:
        current = pending.pop()
        try:
            info = current.lstat()
        except OSError:
            return False, "stat_failed", count
        if stat.S_ISLNK(info.st_mode):
            return False, "symlink", count
        count += 1
        if stat.S_IMODE(info.st_mode) & 0o077:
            return False, "group_or_other_access", count
        if tree and stat.S_ISDIR(info.st_mode):
            try:
                with os.scandir(current) as entries:
                    pending.extend(Path(entry.path) for entry in entries)
            except OSError:
                return False, "scan_failed", count
    return True, "private_mode", count


def macos_volume_encrypted(path: Path, runner: Runner = _run) -> bool:
    """Require FileVault On and the volume containing path to report encryption."""
    status = runner(["/usr/bin/fdesetup", "status"])
    if status.returncode != 0 or not re.search(
        rb"(?im)^FileVault is On\.?\s*$", status.stdout.strip()
    ):
        return False
    anchor = _existing_anchor(path)
    # diskutil info does not reliably resolve arbitrary directories. Ask df for
    # the mounted source and mountpoint, then inspect that volume's plist.
    mounted = runner(["/bin/df", "-P", str(anchor)])
    if mounted.returncode != 0:
        return False
    lines = mounted.stdout.decode("utf-8", "replace").splitlines()
    if len(lines) < 2:
        return False
    fields = lines[-1].split()
    if len(fields) < 6:
        return False
    source, mount_point_from_df = fields[0], " ".join(fields[5:])
    if not source.startswith("/dev/") or not mount_point_from_df.startswith("/"):
        return False
    info = runner(["/usr/sbin/diskutil", "info", "-plist", source])
    if info.returncode != 0:
        return False
    try:
        details = plistlib.loads(info.stdout)
    except Exception:
        return False
    mount_point = details.get("MountPoint")
    encrypted = details.get("FileVault", details.get("Encrypted"))
    if not isinstance(mount_point, str) or not mount_point or mount_point != mount_point_from_df:
        return False
    return encrypted is True or str(encrypted).strip().lower() in {"yes", "true", "on"}


def linux_luks_encrypted(path: Path, runner: Runner = _run) -> bool:
    """Require a mounted /dev/mapper source with an active cryptsetup mapping."""
    anchor = _existing_anchor(path)
    mounted = runner(["/usr/bin/findmnt", "--noheadings", "--output", "SOURCE", "--target", str(anchor)])
    if mounted.returncode != 0:
        return False
    source = mounted.stdout.decode("utf-8", "replace").strip().splitlines()
    if len(source) != 1:
        return False
    device = source[0].split("[", 1)[0].strip()
    prefix = "/dev/mapper/"
    if not device.startswith(prefix):
        return False
    mapping = device[len(prefix):]
    if not mapping or not re.fullmatch(r"[A-Za-z0-9_.+-]+", mapping):
        return False
    status = runner(["/sbin/cryptsetup", "status", mapping])
    if status.returncode != 0:
        return False
    output = status.stdout.decode("utf-8", "replace")
    return bool(re.search(r"(?im)^\s*type:\s*LUKS(?:1|2)\s*$", output)) and bool(
        re.search(r"(?im)^\s*device:\s*/dev/\S+\s*$", output)
    )


def _encryption_checks(paths: Iterable[tuple[str, Path]], platform: str) -> list[dict[str, str | bool]]:
    results = []
    for name, path in paths:
        if platform.lower().startswith("darwin"):
            ok = macos_volume_encrypted(path)
            code = "filevault_and_volume_encrypted" if ok else "filevault_or_volume_not_verified"
        elif platform.lower().startswith("linux"):
            ok = linux_luks_encrypted(path)
            code = "active_luks_mapping" if ok else "luks_mount_not_verified"
        else:
            ok, code = False, "unsupported_platform"
        results.append(_check(name, ok, code))
    return results


def run_checks(data_dir: Path, backup_dirs: list[Path], *, platform: str | None = None) -> dict[str, object]:
    platform = platform or sys.platform
    database = data_dir / "platform.sqlite3"
    checks: list[dict[str, str | bool]] = []
    data_ok, data_code, _ = inspect_permissions(data_dir, required=True)
    checks.append(_check("data_directory_permissions", data_ok, data_code))
    db_ok, db_code, _ = inspect_permissions(database, required=True)
    checks.append(_check("sqlite_main_permissions", db_ok, db_code))
    for suffix in ("-wal", "-shm"):
        ok, code, _ = inspect_permissions(Path(str(database) + suffix), required=False)
        checks.append(_check(f"sqlite_{suffix[1:]}_permissions", ok, code))
    recordings = data_dir / "files"
    ok, code, _ = inspect_permissions(recordings, required=True, tree=True)
    checks.append(_check("recordings_permissions", ok, code))
    for index, backup in enumerate(backup_dirs, start=1):
        ok, code, _ = inspect_permissions(backup, required=True, tree=True)
        checks.append(_check(f"backup_{index}_permissions", ok, code))

    encryption_paths: list[tuple[str, Path]] = [("data_at_rest_encryption", data_dir)]
    encryption_paths.extend((f"backup_{index}_at_rest_encryption", path)
                            for index, path in enumerate(backup_dirs, start=1))
    checks.extend(_encryption_checks(encryption_paths, platform))
    return {
        "schema_version": 1,
        "platform": platform,
        "status": "pass" if all(item["status"] == "pass" for item in checks) else "fail",
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check ASR storage permissions and at-rest encryption.")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(os.environ.get("ASR_DATA_DIR", "data")))
    parser.add_argument("--backup-dir", type=Path, action="append", required=True,
                        help="backup directory; repeat for every configured backup location")
    args = parser.parse_args(argv)
    try:
        result = run_checks(args.data_dir, args.backup_dir)
    except Exception:
        # Keep unexpected OS/tool errors fail-closed and never serialize paths or exception text.
        result = {"schema_version": 1, "platform": sys.platform, "status": "fail",
                  "checks": [_check("security_check", False, "check_failed") ]}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
