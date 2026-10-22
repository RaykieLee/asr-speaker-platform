#!/usr/bin/env python3
"""Fetch pinned ASR assets, verify them, and extract only selected model files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import BinaryIO


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.json"
DEFAULT_CACHE = ROOT / "cache"
_CHUNK_SIZE = 1024 * 1024


class DownloadError(RuntimeError):
    pass


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _valid_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdefABCDEF" for char in value
    ):
        raise DownloadError(f"{label} 的 SHA-256 格式无效。")
    return value.lower()


def _safe_relative(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise DownloadError(f"非法路径: {label}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise DownloadError(f"非法路径: {label}")
    return path


def _checked_target(root: Path, relative: object, label: str) -> Path:
    safe = _safe_relative(relative, label)
    root = root.resolve()
    target = root.joinpath(*safe.parts)
    if not target.resolve().is_relative_to(root):
        raise DownloadError(f"非法模型文件路径: {label}")
    return target


def _matches(path: Path, expected: str, size: int | None = None) -> bool:
    try:
        return path.is_file() and (size is None or path.stat().st_size == size) and digest(path) == expected
    except OSError:
        return False


def _copy_verified(source: BinaryIO, target: Path, expected: str, *,
                   expected_size: int | None = None) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    sha = hashlib.sha256()
    size = 0
    try:
        with partial.open("wb") as output:
            while chunk := source.read(_CHUNK_SIZE):
                output.write(chunk)
                sha.update(chunk)
                size += len(chunk)
        if expected_size is not None and size != expected_size:
            raise DownloadError(f"{target.name} 文件大小不匹配: 期望 {expected_size}，实际 {size}。")
        actual = sha.hexdigest()
        if actual != expected:
            raise DownloadError(f"{target.name} SHA-256 不匹配: 期望 {expected}，实际 {actual}。")
        partial.replace(target)
        return target
    finally:
        partial.unlink(missing_ok=True)


def _download_to_cache(url: str, target: Path, expected: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    sha = hashlib.sha256()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "asr-model-fetch/2.0"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
            while chunk := response.read(_CHUNK_SIZE):
                output.write(chunk)
                sha.update(chunk)
        actual = sha.hexdigest()
        if actual != expected:
            raise DownloadError(f"{target.name} SHA-256 不匹配: 期望 {expected}，实际 {actual}。")
        partial.replace(target)
        return target
    finally:
        partial.unlink(missing_ok=True)


def _source_file(source: dict, cache_dir: Path) -> Path:
    filename = _safe_relative(source.get("filename"), "source.filename")
    if len(filename.parts) != 1:
        raise DownloadError("source.filename 必须是文件名，不能包含目录。")
    expected = _valid_digest(source.get("sha256"), filename.name)
    url = source.get("url")
    if not isinstance(url, str) or not url.startswith("https://"):
        raise DownloadError(f"{filename.name} 缺少有效下载 URL。")
    cached = cache_dir / filename.name
    if cached.exists():
        if not _matches(cached, expected):
            raise DownloadError(f"缓存 {cached} 校验失败；删除缓存文件后重试。")
        return cached
    return _download_to_cache(url, cached, expected)


def _extract_member(archive: tarfile.TarFile, member_name: object, target: Path,
                    expected: str, expected_size: int | None) -> Path:
    member_path = _safe_relative(member_name, "archive member")
    matches = [item for item in archive.getmembers() if item.name == member_path.as_posix()]
    if len(matches) != 1:
        raise DownloadError(f"归档成员不存在或重复: {member_path}")
    info = matches[0]
    if not info.isfile():
        raise DownloadError(f"拒绝提取非普通文件归档成员: {member_path}")
    if expected_size is not None and info.size != expected_size:
        raise DownloadError(f"归档成员大小不匹配: {member_path}")
    source = archive.extractfile(info)
    if source is None:
        raise DownloadError(f"无法读取归档成员: {member_path}")
    with source:
        return _copy_verified(source, target, expected, expected_size=expected_size)


def _write_sentencepiece_vocab(source: Path, target: Path, expected: str,
                               expected_size: int | None) -> Path:
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise DownloadError("生成 bpe.vocab 需要已安装项目依赖 sentencepiece；先安装 Python 依赖再下载模型。") from exc
    try:
        processor = spm.SentencePieceProcessor(model_file=str(source))
        content = "".join(
            f"{processor.id_to_piece(index)}\t{processor.get_score(index)}\n"
            for index in range(processor.get_piece_size())
        ).encode("utf-8")
    except Exception as exc:
        raise DownloadError(f"无法从 {source.name} 生成 SentencePiece 词表: {exc}") from exc
    import io
    return _copy_verified(io.BytesIO(content), target, expected, expected_size=expected_size)


def _download_legacy(model: dict, destination: Path) -> list[Path]:
    """Keep direct-file manifests used by older callers and small utility tests."""
    model_id = model.get("id", "<unknown>")
    artifacts = model.get("files") or []
    if not artifacts:
        version = model.get("version") or model.get("version_status", "版本未固定")
        raise DownloadError(
            f"{model_id} ({version}) 尚无已固定的下载文件和 SHA-256。"
            "请先按 models/README.md 确认上游版本、许可证和校验值；拒绝下载未校验模型。"
        )
    results = []
    model_root = _checked_target(Path(destination), model_id, "model id")
    for item in artifacts:
        filename = _safe_relative(item.get("name"), "files[].name")
        expected = _valid_digest(item.get("sha256"), filename.name)
        url = item.get("url")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise DownloadError(f"{model_id}/{filename} 缺少有效下载 URL。")
        target = _checked_target(model_root, filename.as_posix(), filename.name)
        if _matches(target, expected):
            results.append(target)
            continue
        request = urllib.request.Request(url, headers={"User-Agent": "asr-model-fetch/2.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            results.append(_copy_verified(response, target, expected))
    return results


def download_model(model: dict, destination: Path = ROOT, *,
                   cache_dir: Path | None = None) -> list[Path]:
    """Download one model from a pinned file or archive into its model directory."""
    if not isinstance(model, dict):
        raise DownloadError("模型清单条目必须是对象。")
    if "source" not in model:
        return _download_legacy(model, Path(destination))

    model_id = model.get("id", "<unknown>")
    target_root = _checked_target(Path(destination), model.get("directory", model_id), "directory")
    cache_root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE
    source = model.get("source")
    if not isinstance(source, dict) or source.get("type") not in {"file", "archive"}:
        raise DownloadError(f"{model_id} source.type 必须是 file 或 archive。")
    artifacts = model.get("artifacts")
    generated = model.get("generated_artifacts", [])
    if not isinstance(artifacts, list) or not artifacts:
        raise DownloadError(f"{model_id} 没有固定 artifacts，拒绝下载。")
    if not isinstance(generated, list):
        raise DownloadError(f"{model_id} generated_artifacts 格式无效。")
    for item in generated:
        if not isinstance(item, dict):
            raise DownloadError(f"{model_id} generated_artifacts 条目格式无效。")
        _valid_digest(item.get("sha256"), str(item.get("name")))
        _checked_target(target_root, item.get("name"), "generated artifact name")
        size = item.get("size")
        if size is not None and (not isinstance(size, int) or size < 0):
            raise DownloadError(f"{model_id}/{item.get('name')} size 必须是非负整数。")

    targets: list[tuple[dict, Path, str, int | None]] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise DownloadError(f"{model_id} artifacts 条目格式无效。")
        name = artifact.get("name")
        if len(_safe_relative(name, "artifacts[].name").parts) != 1:
            raise DownloadError(f"{model_id} artifact name 必须是文件名。")
        expected = _valid_digest(artifact.get("sha256"), str(name))
        size = artifact.get("size")
        if size is not None and (not isinstance(size, int) or size < 0):
            raise DownloadError(f"{model_id}/{name} size 必须是非负整数。")
        targets.append((artifact, _checked_target(target_root, name, str(name)), expected, size))

    # Existing verified files are useful when switching to a worktree that shares
    # an ignored model directory; they still must match the pinned contents.
    for artifact, target, expected, size in targets:
        if not _matches(target, expected, size):
            break
    else:
        all_generated_valid = True
        for item in generated:
            expected = _valid_digest(item.get("sha256"), str(item.get("name")))
            target = _checked_target(target_root, item.get("name"), "generated artifact name")
            size = item.get("size")
            if not _matches(target, expected, size):
                all_generated_valid = False
                break
        if all_generated_valid:
            return [target for _, target, _, _ in targets] + [
                _checked_target(target_root, item["name"], "generated artifact name") for item in generated
            ]

    source_path = _source_file(source, cache_root)
    if source["type"] == "file":
        if len(targets) != 1:
            raise DownloadError(f"{model_id} file source 必须对应一个 artifact。")
        _, target, expected, size = targets[0]
        if not _matches(target, expected, size):
            with source_path.open("rb") as data:
                _copy_verified(data, target, expected, expected_size=size)
    else:
        try:
            with tarfile.open(source_path, mode="r:bz2") as archive:
                for artifact, target, expected, size in targets:
                    if _matches(target, expected, size):
                        continue
                    _extract_member(archive, artifact.get("member"), target, expected, size)
        except (tarfile.TarError, OSError) as exc:
            raise DownloadError(f"无法读取模型归档 {source_path.name}: {exc}") from exc

    for item in generated:
        if not isinstance(item, dict) or item.get("generator") != "sentencepiece_vocab_tsv":
            raise DownloadError(f"{model_id} 存在不支持的 generated_artifact。")
        source_name = _safe_relative(item.get("source"), "generated_artifacts[].source")
        if len(source_name.parts) != 1:
            raise DownloadError(f"{model_id} 生成来源必须是模型目录内的文件名。")
        vocab_source = _checked_target(target_root, source_name.as_posix(), source_name.name)
        target = _checked_target(target_root, item.get("name"), "generated_artifacts[].name")
        expected = _valid_digest(item.get("sha256"), target.name)
        size = item.get("size")
        if size is not None and (not isinstance(size, int) or size < 0):
            raise DownloadError(f"{model_id}/{target.name} size 必须是非负整数。")
        if not _matches(target, expected, size):
            _write_sentencepiece_vocab(vocab_source, target, expected, size)
        targets.append((item, target, expected, size))
    return [target for _, target, _, _ in targets]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", help="manifest.json 中的模型 ID")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--destination", type=Path, help="模型文件输出目录，默认为 manifest 所在目录")
    parser.add_argument("--cache-dir", type=Path, help="归档/模型缓存目录，默认为 models/cache")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        model = next((item for item in manifest["models"] if item["id"] == args.model_id), None)
        if model is None:
            parser.error(f"未知模型 ID: {args.model_id}")
        destination = args.destination or args.manifest.resolve().parent
        cache_dir = args.cache_dir or args.manifest.resolve().parent / "cache"
        for path in download_model(model, destination, cache_dir=cache_dir):
            print(path)
    except (DownloadError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
