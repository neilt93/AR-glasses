"""Tests for the assistant brain (src/assistant/brain.py)."""

import time
import pytest

from src.assistant.brain import AssistantBrain
from src.perception.scene import ScenePerception, TextRegion
from src.perception.detector import Detection
from src.hud.widgets import NotificationStack, InfoPanel, ObjectLabels


class TestAssistantBrain:
    def test_default_renderer_has_widgets(self):
        brain = AssistantBrain()
        assert brain.renderer.get(NotificationStack) is not None
        assert brain.renderer.get(InfoPanel) is not None

    def test_notifies_on_new_object(self):
        brain = AssistantBrain(notification_cooldown=0.0)
        perception = ScenePerception(objects=[
            Detection("laptop", 63, (0, 0, 100, 100), 0.9),
        ])
        brain.think(perception)
        ns = brain.renderer.get(NotificationStack)
        assert ns.count >= 1

    def test_no_duplicate_notification_within_cooldown(self):
        brain = AssistantBrain(notification_cooldown=999.0)
        perception = ScenePerception(objects=[
            Detection("laptop", 63, (0, 0, 100, 100), 0.9),
        ])
        brain.think(perception)
        count_after_first = brain.renderer.get(NotificationStack).count

        # Same perception again — should NOT notify again (cooldown)
        brain.think(perception)
        count_after_second = brain.renderer.get(NotificationStack).count
        assert count_after_second == count_after_first

    def test_notifies_on_people_change(self):
        brain = AssistantBrain(notification_cooldown=0.0)
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
        brain = AssistantBrain(notification_cooldown=0.0)
        p1 = ScenePerception(scene_tags=["workspace"])
        brain.think(p1)
        count1 = brain.renderer.get(NotificationStack).count

        p2 = ScenePerception(scene_tags=["kitchen/dining"])
        brain.think(p2)
        assert brain.renderer.get(NotificationStack).count > count1

    def test_updates_info_panel(self):
        brain = AssistantBrain()
        perception = ScenePerception(
            objects=[Detection("cup", 41, (0, 0, 50, 50), 0.8)],
            scene_tags=["kitchen/dining"],
        )
        brain.think(perception)
        ip = brain.renderer.get(InfoPanel)
        assert ip is not None
        assert len(ip._lines) > 0

    def test_toggle_widget(self):
        brain = AssistantBrain()
        labels = brain.renderer.get(ObjectLabels)
        assert labels.enabled is True
        brain.toggle_widget(ObjectLabels)
        assert labels.enabled is False
        brain.toggle_widget(ObjectLabels)
        assert labels.enabled is True

    def test_reset(self):
        brain = AssistantBrain(notification_cooldown=0.0)
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
        brain = AssistantBrain()
        brain.notify("Hello from external", "warning")
        ns = brain.renderer.get(NotificationStack)
        assert ns.count == 1
