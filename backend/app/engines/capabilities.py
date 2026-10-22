"""sherpa-onnx recognition feature matrix, kept separate from model loading."""

from dataclasses import dataclass

from .base import HotwordsUnsupportedError, EngineConfigError


@dataclass(frozen=True)
class EngineCapabilities:
    architecture: str
    online_hotwords: bool = False
    offline_hotwords: bool = False
    notes: str = ""

    def require(self, *, hotwords: bool = False, mode: str = "offline") -> None:
        if mode not in {"online", "offline"}:
            raise EngineConfigError(f"未知识别模式: {mode}")
        if hotwords and not getattr(self, f"{mode}_hotwords"):
            raise HotwordsUnsupportedError(
                f"模型架构 {self.architecture} 不支持 {mode} 原生热词；"
                "请改用支持热词的 Transducer，或明确关闭热词。"
            )


_CAPABILITIES = {
    # sherpa-onnx OnlineRecognizer/OfflineRecognizer Transducer support
    # hotwords through create_stream(hotwords) when configured for
    # modified_beam_search and the pinned binding/model combination supports it.
    "transducer": EngineCapabilities(
        "transducer", True, True,
        "Requires modified_beam_search and compatible sherpa-onnx binding/model.",
    ),
    "sense_voice": EngineCapabilities(
        "sense_voice", False, False,
        "sherpa-onnx SenseVoice factory has no native hotwords parameter.",
    ),
    "ctc": EngineCapabilities("ctc"),
}


def capabilities_for(architecture: str) -> EngineCapabilities:
    try:
        return _CAPABILITIES[architecture]
    except KeyError as exc:
        raise EngineConfigError(f"未知模型架构: {architecture}") from exc
