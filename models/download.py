#!/usr/bin/env python3
"""Download only manifest-pinned model assets and verify their SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.json"


class DownloadError(RuntimeError):
    pass


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def download_model(model: dict, destination: Path = ROOT) -> list[Path]:
    model_id = model.get("id", "<unknown>")
    version = model.get("version") or model.get("version_status", "版本未固定")
    artifacts = model.get("files") or []
    if not artifacts:
        raise DownloadError(
            f"{model_id} ({version}) 尚无已固定的下载文件和 SHA-256。"
            "请先按 models/README.md 确认上游版本、许可证和校验值；拒绝下载未校验模型。"
        )
    results: list[Path] = []
    for item in artifacts:
        url, filename, expected = item.get("url"), item.get("name"), item.get("sha256")
        if not url or not filename or expected is None or expected == "":
            raise DownloadError(f"{model_id} 的下载项缺少 URL、文件名或 SHA-256，拒绝下载。")
        if not isinstance(expected, str) or len(expected) != 64 or any(
            c not in "0123456789abcdefABCDEF" for c in expected
        ):
            raise DownloadError(f"{model_id}/{filename} 的 SHA-256 格式无效。")
        target = (destination / model_id / filename).resolve()
        if not target.is_relative_to(destination.resolve()):
            raise DownloadError(f"非法模型文件路径: {filename}")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "asr-model-fetch/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            actual = digest(partial)
            if actual.lower() != expected.lower():
                raise DownloadError(
                    f"{model_id}/{filename} SHA-256 不匹配: 期望 {expected}，实际 {actual}。"
                )
            partial.replace(target)
            results.append(target)
        finally:
            partial.unlink(missing_ok=True)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", help="manifest.json 中的模型 ID")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    model = next((m for m in manifest["models"] if m["id"] == args.model_id), None)
    if model is None:
        parser.error(f"未知模型 ID: {args.model_id}")
    try:
        for path in download_model(model, args.manifest.resolve().parent):
            print(path)
    except (DownloadError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
