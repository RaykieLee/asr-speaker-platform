"""Speech engine interfaces and sherpa-onnx capability validation."""

from .base import EngineConfigError, HotwordsUnsupportedError, MissingModelError, ModelArtifact
from .capabilities import EngineCapabilities, capabilities_for

__all__ = [
    "EngineCapabilities",
    "EngineConfigError",
    "HotwordsUnsupportedError",
    "MissingModelError",
    "ModelArtifact",
    "capabilities_for",
]
