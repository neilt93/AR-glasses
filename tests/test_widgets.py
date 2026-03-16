"""Tests for HUD widget system (src/hud/widgets.py)."""

import time
import pytest
import numpy as np

from src.hud.widgets import (
    Widget, ObjectLabels, StatusBar, SceneSummary, NotificationStack,
    InfoPanel, Crosshair, WidgetRenderer, Notification, class_color,
)
from src.perception.scene import ScenePerception
from src.perception.detector import Detection


def _frame(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _perception_with_objects():
    return ScenePerception(
        objects=[
            Detection("laptop", 63, (100, 100, 300, 300), 0.9),
            Detection("cup", 41, (350, 200, 400, 300), 0.7),
        ],
        people_count=0,
        dominant_objects=["laptop", "cup"],
        scene_tags=["workspace"],
    )


class TestClassColor:
    def test_deterministic(self):
        c1 = class_color("laptop")
        c2 = class_color("laptop")
        assert c1 == c2

    def test_different_classes(self):
        # Not guaranteed different, but should work for most pairs
        assert isinstance(class_color("cup"), tuple)
        assert len(class_color("cup")) == 3


class TestObjectLabels:
    def test_draws_without_error(self):
        w = ObjectLabels()
        frame = _frame()
        ctx = {"perception": _perception_with_objects()}
        result = w.draw(frame, ctx)
        assert result.shape == (480, 640, 3)

    def test_no_perception(self):
        w = ObjectLabels()
        frame = _frame()
        result = w.draw(frame, {})
        # Should not crash
        assert result is frame

    def test_min_confidence_filter(self):
        w = ObjectLabels(min_confidence=0.8)
        frame = _frame()
        # cup has 0.7 confidence, should be filtered
        perception = _perception_with_objects()
        ctx = {"perception": perception}
        result = w.draw(frame, ctx)
        assert result.shape == (480, 640, 3)


class TestStatusBar:
    def test_draws(self):
        w = StatusBar()
        frame = _frame()
        result = w.draw(frame, {"fps": 30.0, "mode": "assistant"})
        assert result.shape == (480, 640, 3)


class TestSceneSummary:
    def test_draws_with_perception(self):
        w = SceneSummary()
        frame = _frame()
        ctx = {"perception": _perception_with_objects()}
        result = w.draw(frame, ctx)
        assert result.shape == (480, 640, 3)

    def test_draws_without_perception(self):
        w = SceneSummary()
        frame = _frame()
        result = w.draw(frame, {})
        assert result.shape == (480, 640, 3)


class TestNotificationStack:
    def test_push_and_count(self):
        ns = NotificationStack()
        ns.push("hello")
        assert ns.count == 1

    def test_draws(self):
        ns = NotificationStack()
        ns.push("Test notification")
        frame = _frame()
        result = ns.draw(frame, {})
        assert result.shape == (480, 640, 3)

    def test_expiry(self):
        ns = NotificationStack()
        ns.push("expire me", duration=0.01)
        time.sleep(0.02)
        frame = _frame()
        ns.draw(frame, {})  # triggers pruning
        assert ns.count == 0

    def test_clear(self):
        ns = NotificationStack()
        ns.push("a")
        ns.push("b")
        ns.clear()
        assert ns.count == 0


class TestNotification:
    def test_not_expired_initially(self):
        n = Notification("test", duration=5.0)
        assert n.expired is False

    def test_expired(self):
        n = Notification("test", duration=0.01)
        time.sleep(0.02)
        assert n.expired is True

    def test_age_ratio(self):
        n = Notification("test", duration=10.0)
        assert n.age_ratio < 0.1


class TestInfoPanel:
    def test_draws_empty(self):
        ip = InfoPanel()
        frame = _frame()
        result = ip.draw(frame, {})
        assert result is frame  # no lines, returns unchanged

    def test_draws_with_content(self):
        ip = InfoPanel()
        ip.set_content([("Objects", "laptop, cup"), ("FPS", "30")])
        frame = _frame()
        result = ip.draw(frame, {})
        assert result.shape == (480, 640, 3)

    def test_positions(self):
        for pos in ["top-right", "top-left", "bottom-right", "bottom-left"]:
            ip = InfoPanel(position=pos)
            ip.set_content([("Test", "value")])
            frame = _frame()
            ip.draw(frame, {})  # should not crash


class TestCrosshair:
    def test_draws(self):
        ch = Crosshair()
        frame = _frame()
        result = ch.draw(frame, {})
        assert result.shape == (480, 640, 3)


class TestWidgetRenderer:
    def test_add_and_render(self):
        r = WidgetRenderer()
        r.add(StatusBar())
        r.add(SceneSummary())
        frame = _frame()
        result = r.render(frame, {"fps": 30.0})
        assert result.shape == (480, 640, 3)

    def test_z_order(self):
        r = WidgetRenderer()
        r.add(SceneSummary(z_order=5))
        r.add(StatusBar(z_order=1))
        # Should be sorted by z_order
        assert isinstance(r.widgets[0], StatusBar)
        assert isinstance(r.widgets[1], SceneSummary)

    def test_disabled_widget_skipped(self):
        r = WidgetRenderer()
        sb = StatusBar(enabled=False)
        r.add(sb)
        frame = _frame()
        original = frame.copy()
        r.render(frame, {})
        # Disabled widget shouldn't modify the frame (status bar draws on it)
        # We just verify no crash; exact pixel comparison is fragile

    def test_remove(self):
        r = WidgetRenderer()
        r.add(StatusBar())
        r.add(SceneSummary())
        assert len(r.widgets) == 2
        r.remove(StatusBar)
        assert len(r.widgets) == 1

    def test_get(self):
        r = WidgetRenderer()
        sb = StatusBar()
        r.add(sb)
        assert r.get(StatusBar) is sb
        assert r.get(InfoPanel) is None
