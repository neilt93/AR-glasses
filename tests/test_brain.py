"""Tests for the assistant brain (src/assistant/brain.py)."""

import time
import pytest

from src.assistant.brain import AssistantBrain
from src.assistant.voice import MockVoiceEngine
from src.perception.scene import ScenePerception, TextRegion
from src.perception.detector import Detection
from src.perception.hands import HandResult, Gesture
from src.hud.widgets import (
    NotificationStack, InfoPanel, ObjectLabels, TrackedObjectLabels,
    HandSkeleton,
)


class TestAssistantBrain:
    def test_default_renderer_has_widgets(self):
        brain = AssistantBrain(enable_tracking=False)
        assert brain.renderer.get(NotificationStack) is not None
        assert brain.renderer.get(InfoPanel) is not None

    def test_tracking_renderer_has_tracked_labels(self):
        brain = AssistantBrain(enable_tracking=True)
        assert brain.renderer.get(TrackedObjectLabels) is not None

    def test_hands_renderer_has_skeleton(self):
        brain = AssistantBrain(enable_tracking=False, enable_hands=True)
        assert brain.renderer.get(HandSkeleton) is not None

    def test_notifies_on_new_object(self):
        brain = AssistantBrain(notification_cooldown=0.0, enable_tracking=False)
        perception = ScenePerception(objects=[
            Detection("laptop", 63, (0, 0, 100, 100), 0.9),
        ])
        brain.think(perception)
        ns = brain.renderer.get(NotificationStack)
        assert ns.count >= 1

    def test_no_duplicate_notification_within_cooldown(self):
        brain = AssistantBrain(notification_cooldown=999.0, enable_tracking=False)
        perception = ScenePerception(objects=[
            Detection("laptop", 63, (0, 0, 100, 100), 0.9),
        ])
        brain.think(perception)
        count_after_first = brain.renderer.get(NotificationStack).count

        brain.think(perception)
        count_after_second = brain.renderer.get(NotificationStack).count
        assert count_after_second == count_after_first

    def test_notifies_on_people_change(self):
        brain = AssistantBrain(notification_cooldown=0.0, enable_tracking=False)
        p1 = ScenePerception(people_count=0)
        brain.think(p1)
        initial_count = brain.renderer.get(NotificationStack).count

        p2 = ScenePerception(
            objects=[Detection("person", 0, (0, 0, 50, 100), 0.9)],
            people_count=1,
        )
        brain.think(p2)
        assert brain.renderer.get(NotificationStack).count > initial_count

    def test_notifies_on_scene_tag_change(self):
        brain = AssistantBrain(notification_cooldown=0.0, enable_tracking=False)
        p1 = ScenePerception(scene_tags=["workspace"])
        brain.think(p1)
        count1 = brain.renderer.get(NotificationStack).count

        p2 = ScenePerception(scene_tags=["kitchen/dining"])
        brain.think(p2)
        assert brain.renderer.get(NotificationStack).count > count1

    def test_updates_info_panel(self):
        brain = AssistantBrain(enable_tracking=False)
        perception = ScenePerception(
            objects=[Detection("cup", 41, (0, 0, 50, 50), 0.8)],
            scene_tags=["kitchen/dining"],
        )
        brain.think(perception)
        ip = brain.renderer.get(InfoPanel)
        assert ip is not None
        assert len(ip._lines) > 0

    def test_toggle_widget(self):
        brain = AssistantBrain(enable_tracking=False)
        labels = brain.renderer.get(ObjectLabels)
        assert labels.enabled is True
        brain.toggle_widget(ObjectLabels)
        assert labels.enabled is False
        brain.toggle_widget(ObjectLabels)
        assert labels.enabled is True

    def test_reset(self):
        brain = AssistantBrain(notification_cooldown=0.0, enable_tracking=False)
        perception = ScenePerception(objects=[
            Detection("laptop", 63, (0, 0, 100, 100), 0.9),
        ])
        brain.think(perception)
        brain.reset()
        assert brain._known_objects == set()
        assert brain._known_people_count == 0
        ns = brain.renderer.get(NotificationStack)
        assert ns.count == 0

    def test_external_notify(self):
        brain = AssistantBrain(enable_tracking=False)
        brain.notify("Hello from external", "warning")
        ns = brain.renderer.get(NotificationStack)
        assert ns.count == 1


class TestBrainVoice:
    def test_voice_speaks_on_new_object(self):
        voice = MockVoiceEngine()
        brain = AssistantBrain(
            notification_cooldown=0.0,
            enable_tracking=False,
            voice=voice,
        )
        perception = ScenePerception(objects=[
            Detection("laptop", 63, (0, 0, 100, 100), 0.9),
        ])
        brain.think(perception)
        assert any("laptop" in msg for msg in voice.spoken)

    def test_voice_speaks_on_people(self):
        voice = MockVoiceEngine()
        brain = AssistantBrain(
            notification_cooldown=0.0,
            enable_tracking=False,
            voice=voice,
        )
        p = ScenePerception(
            objects=[Detection("person", 0, (0, 0, 50, 100), 0.9)],
            people_count=1,
        )
        brain.think(p)
        assert any("person" in msg.lower() or "Person" in msg for msg in voice.spoken)

    def test_speak_method(self):
        voice = MockVoiceEngine()
        brain = AssistantBrain(
            enable_tracking=False,
            voice=voice,
        )
        brain.speak("test message")
        assert "test message" in voice.spoken

    def test_shutdown_cleans_voice(self):
        voice = MockVoiceEngine()
        brain = AssistantBrain(enable_tracking=False, voice=voice)
        brain.shutdown()  # should not raise


class TestBrainTracking:
    def test_tracker_produces_tracked_objects(self):
        brain = AssistantBrain(
            notification_cooldown=0.0,
            enable_tracking=True,
        )
        perception = ScenePerception(objects=[
            Detection("cup", 41, (10, 10, 50, 50), 0.9),
        ])
        brain.think(perception)
        assert hasattr(brain, '_last_tracked')
        assert brain._last_tracked is not None
        assert len(brain._last_tracked) == 1

    def test_render_context_includes_tracked(self):
        brain = AssistantBrain(
            notification_cooldown=0.0,
            enable_tracking=True,
        )
        perception = ScenePerception(objects=[
            Detection("cup", 41, (10, 10, 50, 50), 0.9),
        ])
        brain.think(perception)
        ctx = brain.get_render_context(perception, fps=30.0)
        assert "tracked_objects" in ctx
        assert len(ctx["tracked_objects"]) == 1

    def test_info_panel_shows_tracking_count(self):
        brain = AssistantBrain(enable_tracking=True)
        perception = ScenePerception(objects=[
            Detection("cup", 41, (10, 10, 50, 50), 0.9),
            Detection("laptop", 63, (200, 200, 400, 400), 0.85),
        ])
        brain.think(perception)
        ip = brain.renderer.get(InfoPanel)
        labels = [label for label, _ in ip._lines]
        assert "Tracking" in labels
