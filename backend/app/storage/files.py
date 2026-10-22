"""Safe local file storage with atomic writes and root-confined operations."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path, PurePath


class UnsafeStoragePath(ValueError):
    """Raised when a caller supplies a path outside the configured storage root."""


class FileStore:
    """Store binary artifacts under a configured directory.

    Paths passed to methods are always relative. Absolute paths, traversal
    components and symlink escapes are rejected after resolution. ``write``
    creates parent directories and replaces the final path atomically.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, relative_path: str | Path) -> Path:
        raw = str(relative_path)
        candidate_input = Path(raw)
        if not raw or candidate_input.is_absolute() or PurePath(raw).is_absolute():
            raise UnsafeStoragePath("storage path must be a non-empty relative path")
        if any(part == ".." for part in candidate_input.parts):
            raise UnsafeStoragePath("parent traversal is not allowed")
        candidate = (self.root / candidate_input).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise UnsafeStoragePath("path resolves outside storage root") from exc
        if candidate == self.root:
            raise UnsafeStoragePath("path must identify a file below storage root")
        return candidate

    def write(self, relative_path: str | Path, data: bytes) -> Path:
        target = self._resolve(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Resolve again after directory creation to detect a symlink race/escape.
        target = self._resolve(relative_path)
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            return target
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def read(self, relative_path: str | Path) -> bytes:
        return self._resolve(relative_path).read_bytes()

    def exists(self, relative_path: str | Path) -> bool:
        return self._resolve(relative_path).is_file()

    def delete(self, relative_path: str | Path, *, missing_ok: bool = True) -> bool:
        target = self._resolve(relative_path)
        try:
            target.unlink()
            return True
        except FileNotFoundError:
            if missing_ok:
                return False
            raise
