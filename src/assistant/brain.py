"""Assistant brain — the decision layer for the AR smart assistant.

Observes the scene, maintains context, and decides what to show on
the HUD. This is the core intelligence loop that makes the glasses
a smart assistant rather than just a camera viewer.

Responsibilities:
- Track what's in the scene over time (not just per-frame)
- Decide when to surface notifications (new object, person, text)
- Manage contextual info panels
- Rate-limit updates to avoid flickering
"""

import time
from typing import Optional

from src.perception.scene import ScenePerception
from src.hud.widgets import (
    NotificationStack, InfoPanel, WidgetRenderer, StatusBar,
    ObjectLabels, SceneSummary, Crosshair,
)


class AssistantBrain:
    """Core decision loop for the AR assistant.

    Each frame, call `think(perception)` to update the brain's state.
    The brain updates widgets (notifications, info panel) based on
    what it observes.
    """

    def __init__(
        self,
        renderer: Optional[WidgetRenderer] = None,
        notification_cooldown: float = 10.0,  # seconds between similar notifications
    ):
        # Widgets
        self.renderer = renderer or self._default_renderer()
        self._notifications = self.renderer.get(NotificationStack)
        self._info_panel = self.renderer.get(InfoPanel)

        # Tracking state
        self._known_objects: set[str] = set()        # objects we've already notified about
        self._known_people_count: int = 0
        self._last_scene_tags: list[str] = []
        self._last_notification_times: dict[str, float] = {}
        self._cooldown = notification_cooldown
        self._frame_count = 0
        self._scene_stable_count = 0
        self._last_dominant: list[str] = []

    def _default_renderer(self) -> WidgetRenderer:
        """Build the default HUD widget stack."""
        renderer = WidgetRenderer()
        renderer.add(ObjectLabels(z_order=0))
        renderer.add(Crosshair(z_order=1, enabled=False))
        renderer.add(StatusBar(z_order=10))
        renderer.add(SceneSummary(z_order=10))
        renderer.add(NotificationStack(z_order=20))
        renderer.add(InfoPanel(position="top-right", z_order=15))
        return renderer

    def think(self, perception: ScenePerception):
        """Process a frame's perception and update the HUD accordingly."""
        self._frame_count += 1

        # Detect new objects entering the scene
        self._check_new_objects(perception)

        # Track people
        self._check_people(perception)

        # Track scene changes
        self._check_scene_change(perception)

        # Update info panel with current scene context
        self._update_info_panel(perception)

    def _check_new_objects(self, perception: ScenePerception):
        """Notify when new object types appear in the scene."""
        current_names = set(perception.object_names)
        new_objects = current_names - self._known_objects

        for name in new_objects:
            if self._can_notify(f"new_obj_{name}"):
                count = len(perception.objects_of_class(name))
                if count > 1:
                    self._notify(f"Detected: {count}x {name}", "info")
                else:
                    self._notify(f"Detected: {name}", "info")

        # Update known set (but allow re-notification after cooldown)
        self._known_objects = current_names

    def _check_people(self, perception: ScenePerception):
        """Notify on people count changes."""
        count = perception.people_count
        if count != self._known_people_count:
            if count > self._known_people_count and count > 0:
                if self._can_notify("people_increase"):
                    if count == 1:
                        self._notify("Person detected", "info")
                    else:
                        self._notify(f"{count} people detected", "info")
            elif count == 0 and self._known_people_count > 0:
                if self._can_notify("people_gone"):
                    self._notify("No people in view", "info")
            self._known_people_count = count

    def _check_scene_change(self, perception: ScenePerception):
        """Notify when the scene context changes significantly."""
        tags = perception.scene_tags
        if tags != self._last_scene_tags and tags:
            new_tags = set(tags) - set(self._last_scene_tags)
            for tag in new_tags:
                if self._can_notify(f"scene_{tag}"):
                    self._notify(f"Scene: {tag}", "info", duration=4.0)
            self._last_scene_tags = tags

    def _update_info_panel(self, perception: ScenePerception):
        """Update the info panel with current context."""
        if self._info_panel is None:
            return

        lines = []

        # Object summary
        names = perception.object_names
        if names:
            lines.append(("Objects", ", ".join(names[:4])))

        # People
        if perception.people_count > 0:
            lines.append(("People", str(perception.people_count)))

        # Text
        if perception.has_text:
            text_preview = perception.all_text[:25]
            lines.append(("Text", f'"{text_preview}"'))

        # Scene
        if perception.scene_tags:
            lines.append(("Scene", ", ".join(perception.scene_tags[:2])))

        # Inference
        lines.append(("Latency", f"{perception.inference_ms:.0f}ms"))

        self._info_panel.set_content(lines)

    def _can_notify(self, key: str) -> bool:
        """Check if enough time has passed since the last notification with this key."""
        last = self._last_notification_times.get(key, 0)
        return (time.time() - last) >= self._cooldown

    def _notify(self, message: str, level: str = "info", duration: float = 5.0):
        """Push a notification to the HUD."""
        if self._notifications is not None:
            self._notifications.push(message, level=level, duration=duration)
        # Track cooldown by message prefix (first word as key)
        key = message.split(":")[0] if ":" in message else message[:20]
        self._last_notification_times[key] = time.time()

    # ─── Public API ───────────────────────────────────────────────────────

    def notify(self, message: str, level: str = "info", duration: float = 5.0):
        """Push a notification from external code."""
        self._notify(message, level, duration)

    def toggle_widget(self, widget_type: type):
        """Toggle a widget on/off."""
        widget = self.renderer.get(widget_type)
        if widget:
            widget.enabled = not widget.enabled

    def reset(self):
        """Reset brain state for a fresh start."""
        self._known_objects.clear()
        self._known_people_count = 0
        self._last_scene_tags.clear()
        self._last_notification_times.clear()
        self._frame_count = 0
        if self._notifications:
            self._notifications.clear()
