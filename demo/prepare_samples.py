"""Build a local two-voice demo from upstream model test recordings.

The source recordings are downloaded with the pinned ASR model archives. This
script writes only ignored local data; it does not redistribute the recordings.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ROOT / "models/online-small/test_wavs/0.wav",
    ROOT / "models/offline-multi/test_wavs/0.wav",
)
SOURCE_MEMBERS = (
    ("online-small", "sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/test_wavs/0.wav",
     "7d93384ca14702cc584a7a33fe2fed92e89e708549161cb12ea38c916882103b"),
    ("offline-multi", "sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/test_wavs/0.wav",
     "668bf8df51a10027b84d5d8816a1ce11ae93545538dc05cfe2aa6811d399c250"),
)
OUTPUT = ROOT / "data/demo"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (16000, 1, 2, "NONE"):
            raise ValueError(f"示例源文件必须是 16 kHz 单声道 PCM16 WAV: {path}")
        return wav.readframes(wav.getnframes())


def ensure_source(index: int) -> None:
    path = SOURCES[index]
    model_id, member_name, expected = SOURCE_MEMBERS[index]
    if path.is_file():
        if sha256_file(path) != expected:
            raise ValueError(f"示例源文件校验失败: {path}")
        return
    manifest = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8"))
    model = next((item for item in manifest["models"] if item["id"] == model_id), None)
    if model is None or "source" not in model:
        raise SystemExit(f"请先安装 {model_id}，缺少 {path}")
    archive = ROOT / "models/cache" / model["source"]["filename"]
    if not archive.is_file():
        raise SystemExit(f"请先运行 python models/download.py {model_id}，缺少 {archive}")
    if sha256_file(archive) != model["source"]["sha256"]:
        raise ValueError(f"模型归档校验失败: {archive}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".wav.part")
    try:
        with tarfile.open(archive, "r:bz2") as package:
            member = package.getmember(member_name)
            if not member.isfile():
                raise ValueError("示例音频归档成员不是普通文件")
            source = package.extractfile(member)
            if source is None:
                raise ValueError("示例音频归档成员不可读取")
            with source, partial.open("wb") as output:
                shutil.copyfileobj(source, output)
        if sha256_file(partial) != expected:
            raise ValueError(f"示例音频校验失败: {member_name}")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def main() -> None:
    for index in range(len(SOURCES)):
        ensure_source(index)
    clips = [read_pcm(path) for path in SOURCES]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    silence = b"\x00\x00" * 16_000
    sequence = (0, 1, 0, 1)
    output = OUTPUT / "two-speaker-demo.wav"
    timeline = []
    elapsed = 0.0
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        for index in sequence:
            duration = len(clips[index]) / 32_000
            timeline.append({"source": f"speaker_{'a' if index == 0 else 'b'}",
                             "start": elapsed, "end": elapsed + duration})
            wav.writeframesraw(clips[index])
            wav.writeframesraw(silence)
            elapsed += duration + 1.0
        wav.writeframes(b"")
    for label, path in zip(("a", "b"), SOURCES):
        shutil.copyfile(path, OUTPUT / f"speaker_{label}_enroll.wav")
    metadata = {
        "description": "上游模型附带的两段测试录音交替拼接，仅供本机演示；说话人标签为示例别名。",
        "source_files": [{"path": str(path.relative_to(ROOT)),
                          "sha256": sha256_file(path)}
                         for path in SOURCES],
        "timeline": timeline,
        "output": output.name,
    }
    (OUTPUT / "two-speaker-demo.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
