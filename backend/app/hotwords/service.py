"""Hotword application service and immutable resolver snapshots."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

from backend.app.engines import HotwordsUnsupportedError
from backend.app.hotwords.compiler import HotwordCompilationError, HotwordCompiler
from backend.app.storage import SQLiteStore

MAX_HOTWORD_WEIGHT = 20.0
MAX_WORD_LENGTH = 256


class HotwordNotFoundError(LookupError):
    """A set or requested immutable version does not exist or is unusable."""


class StaleHotwordVersionError(ValueError):
    """The editor attempted to update a version that has already changed."""


@dataclass(frozen=True)
class HotwordSnapshot:
    """Resolved immutable hotword version captured by a new job/session."""

    set_id: str
    version: int
    name: str
    weight: float
    entries: tuple[Any, ...]
    compiled: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "set_id": self.set_id,
            "version": self.version,
            "name": self.name,
            "weight": self.weight,
            "entries": _thaw(self.entries),
            "compiled": _thaw(self.compiled),
        }


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, dict) or isinstance(value, MappingProxyType):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def normalize_entries(entries: Any) -> list[dict[str, Any]]:
    if not isinstance(entries, list):
        raise ValueError("entries 必须是字符串或词条对象组成的数组。")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(entries):
        if isinstance(raw, str):
            item = {"word": raw}
        elif isinstance(raw, dict):
            item = raw
        else:
            raise ValueError(f"第 {index + 1} 个词条必须是字符串或对象。")
        word = item.get("word", item.get("phrase"))
        if not isinstance(word, str):
            raise ValueError(f"第 {index + 1} 个词条缺少 word。")
        word = _normalize_phrase(word, f"第 {index + 1} 个词条")
        aliases = item.get("aliases", [])
        if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
            raise ValueError(f"词条 {word!r} 的 aliases 必须是字符串数组。")
        clean_aliases: list[str] = []
        for alias in aliases:
            alias = _normalize_phrase(alias, f"词条 {word!r} 的别名")
            if alias.casefold() not in {word.casefold(), *(v.casefold() for v in clean_aliases)}:
                clean_aliases.append(alias)
        language = item.get("language", item.get("lang"))
        if language is not None:
            if not isinstance(language, str) or not language.strip():
                raise ValueError(f"词条 {word!r} 的 language 必须是非空字符串。")
            language = language.strip()
        enabled = item.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"词条 {word!r} 的 enabled 必须是布尔值。")
        for phrase in (word, *clean_aliases):
            key = phrase.casefold()
            if key in seen:
                raise ValueError(f"热词重复: {phrase!r}。")
            seen.add(key)
        normalized.append({
            "word": word,
            "aliases": clean_aliases,
            "language": language,
            "enabled": enabled,
        })
    if not normalized:
        raise ValueError("entries 不能为空。")
    if not any(item["enabled"] for item in normalized):
        raise ValueError("至少需要一个启用的热词。")
    return normalized


def _normalize_phrase(value: str, label: str) -> str:
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError(f"{label}包含不支持的控制字符。")
    value = re.sub(r"\s+", " ", value.strip())
    if not value:
        raise ValueError(f"{label}不能为空。")
    if len(value) > MAX_WORD_LENGTH:
        raise ValueError(f"{label}不能超过 {MAX_WORD_LENGTH} 个字符。")
    return value


def _validate_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name 不能为空。")
    value = name.strip()
    if len(value) > 100:
        raise ValueError("name 不能超过 100 个字符。")
    return value


def _validate_weight(weight: float) -> float:
    try:
        value = float(weight)
    except (TypeError, ValueError) as exc:
        raise ValueError("weight 必须是数字。") from exc
    if not 0 < value <= MAX_HOTWORD_WEIGHT:
        raise ValueError(f"weight 必须大于 0 且不超过 {MAX_HOTWORD_WEIGHT:g}。")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class HotwordService:
    """CRUD/versioning policy layered over the existing SQLiteStore methods."""

    def __init__(self, store: SQLiteStore, compiler: HotwordCompiler):
        self.store = store
        self.compiler = compiler

    def create(
        self,
        *,
        name: str,
        entries: Any,
        weight: float = 1.0,
        tenant_id: str | None = None,
        created_by: str | None = None,
    ) -> dict[str, Any]:
        clean_name = _validate_name(name)
        clean_entries = normalize_entries(entries)
        clean_weight = _validate_weight(weight)
        compiled = self.compiler.compile(clean_entries, clean_weight)
        created = self.store.create_hotword_set(
            clean_name, clean_entries, weight=clean_weight, tenant_id=tenant_id,
            created_by=created_by, compiled=compiled,
        )
        return self.get(created["set_id"]) or created

    def list_sets(self, *, status: str | None = "active", tenant_id: str | None = None) -> list[dict[str, Any]]:
        if status not in {None, "active", "inactive"}:
            raise ValueError("status 仅支持 active 或 inactive。")
        # SQLiteStore currently exposes get but no list method for hotword sets.
        # Keep the compatibility query in this feature module rather than
        # widening the shared storage layer outside this task's ownership.
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status=?")
            params.append(status)
        if tenant_id is not None:
            clauses.append("tenant_id=?")
            params.append(tenant_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.store._query_all(
            f"SELECT * FROM hotword_sets{where} ORDER BY updated_at DESC, id",
            tuple(params),
        )
        items: list[dict[str, Any]] = []
        for row in rows:
            current = self.store.get_hotword_version(row["id"], row["current_version"])
            items.append({**row, "entries": current["entries"] if current else [],
                          "weight": current["weight"] if current else None})
        return items

    def get(self, set_id: str, *, version: int | None = None) -> dict[str, Any] | None:
        hotword_set = self.store.get_hotword_set(set_id)
        if hotword_set is None:
            return None
        selected = version if version is not None else hotword_set["current_version"]
        snapshot = self.store.get_hotword_version(set_id, selected)
        if snapshot is None:
            return None
        return {**hotword_set, **snapshot}

    def update(
        self,
        set_id: str,
        *,
        expected_version: int,
        entries: Any,
        weight: float = 1.0,
        name: str | None = None,
        created_by: str | None = None,
    ) -> dict[str, Any]:
        clean_entries = normalize_entries(entries)
        clean_weight = _validate_weight(weight)
        clean_name = _validate_name(name) if name is not None else None
        current_set = self.store.get_hotword_set(set_id)
        if current_set is None:
            raise HotwordNotFoundError(set_id)
        if current_set["status"] != "active":
            raise ValueError("已停用的热词集不能继续编辑。")
        compiled = self.compiler.compile(clean_entries, clean_weight)
        if clean_name is None:
            try:
                self.store.append_hotword_version(
                    set_id, clean_entries, expected_version=expected_version,
                    weight=clean_weight, created_by=created_by, compiled=compiled,
                )
            except KeyError as exc:
                raise HotwordNotFoundError(set_id) from exc
            except ValueError as exc:
                if "stale" in str(exc).lower():
                    raise StaleHotwordVersionError(str(exc)) from exc
                raise
            return self.get(set_id, version=expected_version + 1)  # type: ignore[return-value]
        # Compatibility path for atomic rename + immutable version append:
        # SQLiteStore has optimistic append but no set-name update operation.
        now = _now()
        try:
            with self.store.transaction() as db:
                row = db.execute(
                    "SELECT current_version FROM hotword_sets WHERE id=?", (set_id,)
                ).fetchone()
                if row is None:
                    raise HotwordNotFoundError(set_id)
                if row[0] != expected_version:
                    raise StaleHotwordVersionError(
                        f"stale hotword version: expected {expected_version}, current {row[0]}"
                    )
                version = expected_version + 1
                db.execute("INSERT INTO hotword_set_versions VALUES (?,?,?,?,?,?,?)", (
                    set_id, version, json.dumps(clean_entries, ensure_ascii=False),
                    json.dumps(compiled, ensure_ascii=False), clean_weight, created_by, now,
                ))
                db.execute(
                    "UPDATE hotword_sets SET name=?,current_version=?,updated_at=? WHERE id=?",
                    (clean_name, version, now, set_id),
                )
        except HotwordNotFoundError:
            raise
        return self.get(set_id, version=version)  # type: ignore[return-value]

    def deactivate(self, set_id: str) -> bool:
        return self.store.delete_hotword_set(set_id)

    def resolve(
        self,
        set_id: str,
        *,
        version: int | None = None,
        mode: str | None = None,
    ) -> HotwordSnapshot:
        hotword_set = self.store.get_hotword_set(set_id)
        if hotword_set is None:
            raise HotwordNotFoundError(set_id)
        if version is None and hotword_set["status"] != "active":
            raise HotwordNotFoundError(f"热词集 {set_id} 已停用，不能用于新任务。")
        selected_version = version if version is not None else int(hotword_set["current_version"])
        row = self.store.get_hotword_version(set_id, selected_version)
        if row is None:
            raise HotwordNotFoundError(f"热词集 {set_id} 不存在版本 {selected_version}。")
        compiler = self.compiler
        if mode is not None and mode != compiler.mode:
            compiler = HotwordCompiler(
                compiler.tokenizer,
                architecture=compiler.architecture,
                decoding_method=compiler.decoding_method,
                mode=mode,
                tokenizer_id=compiler.tokenizer_id,
                modeling_unit=compiler.modeling_unit,
                vocabulary=compiler.vocabulary,
            )
        compiled = compiler.compile(row["entries"], float(row["weight"]))
        return HotwordSnapshot(
            set_id=set_id,
            version=int(row["version"]),
            name=str(hotword_set["name"]),
            weight=float(row["weight"]),
            entries=_freeze(row["entries"]),
            compiled=_freeze(compiled),
        )
