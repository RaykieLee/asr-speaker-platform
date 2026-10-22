"""Lightweight speech activity gate for live speaker embedding windows."""

from __future__ import annotations

import math
from collections.abc import Sequence


class EnergyVAD:
    """Reject silent windows before running the speaker embedding model.

    This is an energy gate, not a noise classifier. The speaker matcher remains
    conservative and emits unknown when evidence is insufficient.
    """

    def __init__(self, *, frame_seconds: float = 0.03, min_rms: float = 0.008,
                 min_voiced_ratio: float = 0.3) -> None:
        if frame_seconds <= 0 or min_rms <= 0 or not 0 < min_voiced_ratio <= 1:
            raise ValueError("无效的实时语音检测参数")
        self.frame_seconds = frame_seconds
        self.min_rms = min_rms
        self.min_voiced_ratio = min_voiced_ratio

    def is_speech(self, samples: Sequence[float], sample_rate: int) -> bool:
        if not samples or sample_rate <= 0:
            return False
        frame_size = max(1, round(self.frame_seconds * sample_rate))
        voiced = 0
        for start in range(0, len(samples), frame_size):
            frame = samples[start:start + frame_size]
            rms = math.sqrt(sum(float(value) ** 2 for value in frame) / len(frame))
            if rms >= self.min_rms:
                voiced += len(frame)
        return voiced / len(samples) >= self.min_voiced_ratio
