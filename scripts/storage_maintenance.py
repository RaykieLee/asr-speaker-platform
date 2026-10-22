#!/usr/bin/env python3
"""Back up, verify, restore, or prune local ASR SQLite/file storage."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.storage import FileStore, SQLiteStore
from backend.app.storage.maintenance import (
    cleanup_expired_task_audio,
    create_backup,
    restore_backup,
    verify_backup,
)


def _data_dir(value: str | None) -> Path:
    return Path(value or os.environ.get("ASR_DATA_DIR", "data")).expanduser()


def _storage_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", help="ASR data directory (defaults to ASR_DATA_DIR or ./data)")
    parser.add_argument("--database", type=Path, help="override SQLite path")
    parser.add_argument("--files", type=Path, help="override local file store root")


def _paths(args: argparse.Namespace) -> tuple[Path, Path]:
    data_dir = _data_dir(args.data_dir)
    return (args.database or data_dir / "platform.sqlite3", args.files or data_dir / "files")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    backup = commands.add_parser("backup", help="make an online SQLite and artifact backup")
    _storage_args(backup)
    backup.add_argument("--output", type=Path, required=True, help="new backup directory (must not exist)")
    backup.add_argument("--retention-days", type=int, default=30,
                        help="omit terminal task audio older than this from backup database/files (default: 30)")

    verify = commands.add_parser("verify", help="check backup hashes, SQLite integrity and references")
    verify.add_argument("--backup", type=Path, required=True)

    restore = commands.add_parser("restore", help="verify and restore into a new directory")
    restore.add_argument("--backup", type=Path, required=True)
    restore.add_argument("--to", type=Path, required=True, help="new directory; must not exist")

    cleanup = commands.add_parser("cleanup", help="prune expired original task audio; dry-run by default")
    _storage_args(cleanup)
    cleanup.add_argument("--retention-days", type=int, default=30)
    cleanup.add_argument("--apply", action="store_true", help="perform deletions (default is dry-run)")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "backup":
            database, files = _paths(args)
            result = create_backup(database, files, args.output, retention_days=args.retention_days)
        elif args.command == "verify":
            result = verify_backup(args.backup)
        elif args.command == "restore":
            result = {"restored_to": str(restore_backup(args.backup, args.to))}
        else:
            database_path, files_path = _paths(args)
            if not database_path.is_file():
                raise FileNotFoundError(f"database does not exist: {database_path}")
            if not files_path.is_dir():
                raise FileNotFoundError(f"file store does not exist: {files_path}")
            database = SQLiteStore(database_path)
            try:
                database.initialize()
                result = cleanup_expired_task_audio(
                    database, FileStore(files_path), retention_days=args.retention_days,
                    dry_run=not args.apply,
                )
            finally:
                database.close()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.command == "cleanup" and result.get("failures"):
            return 1
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"storage maintenance: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
