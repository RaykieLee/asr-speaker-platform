"""Speaker enrollment, sample quality checks and embedding extraction."""
from .service import (
    EmbeddingModelVersionError,
    MissingSpeakerModelError,
    SampleQualityError,
    SherpaSpeakerEmbeddingAdapter,
    SpeakerModel,
    SpeakerRegistrationService,
    assess_audio_quality,
    l2_normalize,
)

__all__ = [
    "EmbeddingModelVersionError", "MissingSpeakerModelError", "SampleQualityError",
    "SherpaSpeakerEmbeddingAdapter", "SpeakerModel", "SpeakerRegistrationService",
    "assess_audio_quality", "l2_normalize",
]
