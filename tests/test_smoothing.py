"""Tests for smoothing utilities (src/utils/smoothing.py)."""

import time
import pytest
from src.utils.smoothing import EMAFilter, CooldownTimer, FlickerSuppressor, ConfidenceBuffer


class TestEMAFilter:
    def test_first_value_is_raw(self):
        ema = EMAFilter(alpha=0.3)
        assert ema.update(10.0) == pytest.approx(10.0)

    def test_smoothing(self):
        ema = EMAFilter(alpha=0.5)
        ema.update(10.0)
        val = ema.update(20.0)
        # 0.5 * 20 + 0.5 * 10 = 15
        assert val == pytest.approx(15.0)

    def test_alpha_1_follows_raw(self):
        ema = EMAFilter(alpha=1.0)
        ema.update(10.0)
        assert ema.update(99.0) == pytest.approx(99.0)

    def test_alpha_0_stays_at_first(self):
        ema = EMAFilter(alpha=0.0)
        ema.update(10.0)
        assert ema.update(99.0) == pytest.approx(10.0)

    def test_value_property(self):
        ema = EMAFilter()
        assert ema.value is None
        ema.update(5.0)
        assert ema.value == pytest.approx(5.0)

    def test_reset(self):
        ema = EMAFilter()
        ema.update(5.0)
        ema.reset()
        assert ema.value is None


class TestCooldownTimer:
    def test_ready_initially(self):
        timer = CooldownTimer(cooldown_seconds=0.0)
        assert timer.ready() is True

    def test_not_ready_during_cooldown(self):
        timer = CooldownTimer(cooldown_seconds=10.0)
        timer.fire()
        assert timer.ready() is False

    def test_try_fire_returns_true_when_ready(self):
        timer = CooldownTimer(cooldown_seconds=0.0)
        assert timer.try_fire() is True

    def test_try_fire_returns_false_on_cooldown(self):
        timer = CooldownTimer(cooldown_seconds=10.0)
        timer.fire()
        assert timer.try_fire() is False

    def test_ready_after_cooldown_expires(self):
        timer = CooldownTimer(cooldown_seconds=0.01)
        timer.fire()
        time.sleep(0.02)
        assert timer.ready() is True


class TestFlickerSuppressor:
    def test_initial_returns_none(self):
        fs = FlickerSuppressor(min_stable_frames=3)
        assert fs.update("A") is None  # not yet stable

    def test_becomes_stable(self):
        fs = FlickerSuppressor(min_stable_frames=3)
        fs.update("A")
        fs.update("A")
        result = fs.update("A")
        assert result == "A"

    def test_stays_stable_during_flicker(self):
        fs = FlickerSuppressor(min_stable_frames=3)
        for _ in range(3):
            fs.update("A")
        # Now flicker to B briefly
        assert fs.update("B") == "A"  # not enough persistence for B
        assert fs.update("A") == "A"  # back to A

    def test_transitions_after_persistence(self):
        fs = FlickerSuppressor(min_stable_frames=2)
        for _ in range(2):
            fs.update("A")
        fs.update("B")
        result = fs.update("B")
        assert result == "B"

    def test_value_property(self):
        fs = FlickerSuppressor(min_stable_frames=1)
        assert fs.value is None
        fs.update("X")
        assert fs.value == "X"

    def test_reset(self):
        fs = FlickerSuppressor(min_stable_frames=1)
        fs.update("X")
        fs.reset()
        assert fs.value is None


class TestConfidenceBuffer:
    def test_single_value(self):
        buf = ConfidenceBuffer(window_size=5)
        avg = buf.update("screw", 0.8)
        assert avg == pytest.approx(0.8)

    def test_rolling_average(self):
        buf = ConfidenceBuffer(window_size=3)
        buf.update("screw", 0.6)
        buf.update("screw", 0.9)
        avg = buf.update("screw", 1.2)
        assert avg == pytest.approx((0.6 + 0.9 + 1.2) / 3)

    def test_window_drops_old(self):
        buf = ConfidenceBuffer(window_size=2)
        buf.update("x", 1.0)
        buf.update("x", 3.0)
        avg = buf.update("x", 5.0)
        # Window: [3.0, 5.0], dropped 1.0
        assert avg == pytest.approx(4.0)

    def test_separate_classes(self):
        buf = ConfidenceBuffer(window_size=5)
        buf.update("a", 1.0)
        buf.update("b", 2.0)
        assert buf.get_average("a") == pytest.approx(1.0)
        assert buf.get_average("b") == pytest.approx(2.0)

    def test_get_average_unknown_class(self):
        buf = ConfidenceBuffer()
        assert buf.get_average("unknown") == 0.0

    def test_reset(self):
        buf = ConfidenceBuffer()
        buf.update("x", 1.0)
        buf.reset()
        assert buf.get_average("x") == 0.0
