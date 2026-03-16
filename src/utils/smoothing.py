"""Smoothing utilities — EMA filters, cooldown timers, flicker suppression."""

import time
from collections import deque
from typing import Optional


class EMAFilter:
    """Exponential moving average filter for smoothing noisy signals."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._value: Optional[float] = None

    def update(self, raw: float) -> float:
        if self._value is None:
            self._value = raw
        else:
            self._value = self.alpha * raw + (1.0 - self.alpha) * self._value
        return self._value

    @property
    def value(self) -> Optional[float]:
        return self._value

    def reset(self):
        self._value = None


class CooldownTimer:
    """Prevents an action from firing more often than a minimum interval."""

    def __init__(self, cooldown_seconds: float = 1.0):
        self.cooldown = cooldown_seconds
        self._last_fire: float = 0.0

    def ready(self) -> bool:
        return (time.time() - self._last_fire) >= self.cooldown

    def fire(self):
        self._last_fire = time.time()

    def try_fire(self) -> bool:
        if self.ready():
            self.fire()
            return True
        return False


class FlickerSuppressor:
    """Suppresses rapid changes in a discrete value (e.g., step label).

    Only emits a new value after it has been stable for `min_stable_frames`.
    """

    def __init__(self, min_stable_frames: int = 5):
        self.min_stable = min_stable_frames
        self._current = None
        self._candidate = None
        self._candidate_count = 0

    def update(self, value):
        """Feed a new raw value. Returns the stable output value."""
        if value == self._current:
            self._candidate = None
            self._candidate_count = 0
            return self._current

        if value == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = value
            self._candidate_count = 1

        if self._candidate_count >= self.min_stable:
            self._current = self._candidate
            self._candidate = None
            self._candidate_count = 0

        return self._current

    @property
    def value(self):
        return self._current

    def reset(self):
        self._current = None
        self._candidate = None
        self._candidate_count = 0


class ConfidenceBuffer:
    """Maintains a rolling buffer of confidence values per class."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self._buffers: dict[str, deque] = {}

    def update(self, class_name: str, confidence: float) -> float:
        """Add a confidence value and return the rolling average."""
        if class_name not in self._buffers:
            self._buffers[class_name] = deque(maxlen=self.window_size)
        buf = self._buffers[class_name]
        buf.append(confidence)
        return sum(buf) / len(buf)

    def get_average(self, class_name: str) -> float:
        buf = self._buffers.get(class_name)
        if not buf:
            return 0.0
        return sum(buf) / len(buf)

    def reset(self):
        self._buffers.clear()
