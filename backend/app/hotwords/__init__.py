"""Versioned hotword management and model-specific compilation."""

from .compiler import (
    HotwordCompilationError,
    HotwordCompiler,
    HotwordVocabularyUnavailableError,
)
from backend.app.engines import HotwordsUnsupportedError
from .service import HotwordNotFoundError, HotwordService, HotwordSnapshot, StaleHotwordVersionError

__all__ = [
    "HotwordCompilationError",
    "HotwordCompiler",
    "HotwordNotFoundError",
    "HotwordService",
    "HotwordSnapshot",
    "HotwordVocabularyUnavailableError",
    "HotwordsUnsupportedError",
    "StaleHotwordVersionError",
]
