"""Widget-based HUD system for the AR glasses.

A flexible overlay framework where each HUD element is a Widget that
knows how to draw itself. The WidgetRenderer composes widgets onto a frame.

Widget types:
- ObjectLabels: floating labels on detected objects
- TrackedObjectLabels: labels with persistent track IDs and smooth animation
- HandSkeleton: hand landmark skeleton overlay with gesture label
- StatusBar: top bar with time, FPS, scene info
- NotificationStack: temporary messages that fade out
- InfoPanel: contextual info panel (pinned to a corner)
- SceneSummary: bottom bar with scene description
- Crosshair: center crosshair/focus indicator
"""

import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from src.perception.detector import Detection
from src.perception.scene import ScenePerception


# ─── Colors (BGR) ─────────────────────────────────────────────────────────────

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (150, 150, 150)
DARK_BG = (25, 25, 25)
ACCENT = (255, 170, 0)
GREEN = (0, 220, 0)
YELLOW = (0, 220, 220)
RED = (0, 0, 220)
CYAN = (220, 220, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

# Deterministic palette for object classes
CLASS_PALETTE = [
    (0, 220, 0), (220, 180, 0), (0, 140, 255), (200, 0, 200),
    (255, 100, 0), (0, 255, 255), (255, 0, 100), (100, 255, 100),
    (255, 200, 0), (0, 100, 255), (180, 0, 180), (0, 200, 150),
]


def class_color(name: str) -> tuple:
    return CLASS_PALETTE[hash(name) % len(CLASS_PALETTE)]


def draw_translucent_rect(frame: np.ndarray, pt1: tuple, pt2: tuple,
                           color: tuple = DARK_BG, alpha: float = 0.7):
    """Draw a semi-transparent filled rectangle."""
    overlay = frame.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ─── Base Widget ──────────────────────────────────────────────────────────────

class Widget(ABC):
    """Base class for HUD widgets."""

    def __init__(self, enabled: bool = True, z_order: int = 0):
        self.enabled = enabled
        self.z_order = z_order  # higher = drawn later (on top)

    @abstractmethod
    def draw(self, frame: np.ndarray, context: dict) -> np.ndarray:
        """Draw this widget onto the frame.

        Args:
            frame: the frame to draw on (modified in place).
            context: shared state dict with keys like 'perception', 'fps', etc.
        Returns:
            The frame (same reference, modified in place).
        """
        ...


# ─── Object Labels ────────────────────────────────────────────────────────────

class ObjectLabels(Widget):
    """Draws bounding boxes and labels on detected objects."""

    def __init__(self, show_confidence: bool = True, min_confidence: float = 0.3,
                 **kwargs):
        super().__init__(**kwargs)
        self.show_confidence = show_confidence
        self.min_confidence = min_confidence

    def draw(self, frame: np.ndarray, context: dict) -> np.ndarray:
        perception: Optional[ScenePerception] = context.get("perception")
        if not perception:
            return frame

        for det in perception.objects:
            if det.confidence < self.min_confidence:
                continue
            x1, y1, x2, y2 = det.bbox
            color = class_color(det.class_name)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = det.class_name
            if self.show_confidence:
                label += f" {det.confidence:.0%}"
            lw, lh = cv2.getTextSize(label, FONT, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        FONT, 0.5, BLACK, 1, cv2.LINE_AA)

        return frame


# ─── Status Bar ───────────────────────────────────────────────────────────────

class StatusBar(Widget):
    """Top status bar with time, FPS, and mode indicator."""

    def __init__(self, height: int = 45, **kwargs):
        super().__init__(**kwargs)
        self.height = height

    def draw(self, frame: np.ndarray, context: dict) -> np.ndarray:
        h, w = frame.shape[:2]
        draw_translucent_rect(frame, (0, 0), (w, self.height))

        # Time
        now = time.strftime("%H:%M")
        cv2.putText(frame, now, (12, 30), FONT_BOLD, 0.65, WHITE, 1, cv2.LINE_AA)

        # FPS
        fps = context.get("fps", 0.0)
        fps_text = f"{fps:.0f} FPS"
        cv2.putText(frame, fps_text, (w - 100, 30), FONT, 0.5, GRAY, 1, cv2.LINE_AA)

        # Mode
        mode = context.get("mode", "assistant")
        cv2.putText(frame, mode, (w // 2 - 30, 30), FONT, 0.5, ACCENT, 1, cv2.LINE_AA)

        # Object count
        perception: Optional[ScenePerception] = context.get("perception")
        if perception:
            count_text = f"{len(perception.objects)} objects"
            cv2.putText(frame, count_text, (120, 30), FONT, 0.45, GRAY, 1, cv2.LINE_AA)

        return frame


# ─── Scene Summary ────────────────────────────────────────────────────────────

class SceneSummary(Widget):
    """Bottom bar with scene description and tags."""

    def __init__(self, height: int = 40, **kwargs):
        super().__init__(**kwargs)
        self.height = height

    def draw(self, frame: np.ndarray, context: dict) -> np.ndarray:
        h, w = frame.shape[:2]
        y_start = h - self.height
        draw_translucent_rect(frame, (0, y_start), (w, h))

        perception: Optional[ScenePerception] = context.get("perception")
        if perception:
            summary = perception.summary()
            cv2.putText(frame, summary, (12, h - 14),
                        FONT, 0.5, WHITE, 1, cv2.LINE_AA)

            if perception.scene_tags:
                tags = " | ".join(perception.scene_tags)
                tw = cv2.getTextSize(tags, FONT, 0.4, 1)[0][0]
                cv2.putText(frame, tags, (w - tw - 12, h - 14),
                            FONT, 0.4, CYAN, 1, cv2.LINE_AA)

        return frame


# ─── Notification Stack ───────────────────────────────────────────────────────

@dataclass
class Notification:
    message: str
    level: str = "info"       # "info", "warning", "error"
    created_at: float = field(default_factory=time.time)
    duration: float = 5.0     # seconds to display

    @property
    def expired(self) -> bool:
        return (time.time() - self.created_at) > self.duration

    @property
    def age_ratio(self) -> float:
        """0.0 = just created, 1.0 = about to expire."""
        return min(1.0, (time.time() - self.created_at) / self.duration)


class NotificationStack(Widget):
    """Displays temporary notification messages that fade out."""

    def __init__(self, max_visible: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.max_visible = max_visible
        self._notifications: deque[Notification] = deque()

    def push(self, message: str, level: str = "info", duration: float = 5.0):
        """Add a notification."""
        self._notifications.append(Notification(
            message=message, level=level, duration=duration,
        ))

    def draw(self, frame: np.ndarray, context: dict) -> np.ndarray:
        # Prune expired
        while self._notifications and self._notifications[0].expired:
            self._notifications.popleft()

        h, w = frame.shape[:2]
        visible = list(self._notifications)[-self.max_visible:]
        y = 60  # below status bar

        for notif in visible:
            color = WHITE
            if notif.level == "warning":
                color = YELLOW
            elif notif.level == "error":
                color = RED

            # Fade out as it ages
            alpha = max(0.3, 1.0 - notif.age_ratio * 0.7)
            text_color = tuple(int(c * alpha) for c in color)

            lw, lh = cv2.getTextSize(notif.message, FONT, 0.5, 1)[0]
            draw_translucent_rect(frame, (w - lw - 24, y - 2),
                                   (w - 4, y + lh + 8), alpha=0.5)
            cv2.putText(frame, notif.message, (w - lw - 16, y + lh + 2),
                        FONT, 0.5, text_color, 1, cv2.LINE_AA)
            y += lh + 16

        return frame

    @property
    def count(self) -> int:
        return len(self._notifications)

    def clear(self):
        self._notifications.clear()


# ─── Info Panel ───────────────────────────────────────────────────────────────

class InfoPanel(Widget):
    """A pinned info panel showing key-value pairs. Useful for contextual data."""

    def __init__(self, position: str = "top-right", width: int = 250, **kwargs):
        super().__init__(**kwargs)
        self.position = position
        self.width = width
        self._lines: list[tuple[str, str]] = []

    def set_content(self, lines: list[tuple[str, str]]):
        """Set panel content as (label, value) pairs."""
        self._lines = lines

    def draw(self, frame: np.ndarray, context: dict) -> np.ndarray:
        if not self._lines:
            return frame

        h, w = frame.shape[:2]
        line_h = 22
        panel_h = len(self._lines) * line_h + 16
        pad = 8

        if self.position == "top-right":
            x1 = w - self.width - pad
            y1 = 55
        elif self.position == "top-left":
            x1 = pad
            y1 = 55
        elif self.position == "bottom-right":
            x1 = w - self.width - pad
            y1 = h - panel_h - 50
        else:  # bottom-left
            x1 = pad
            y1 = h - panel_h - 50

        x2 = x1 + self.width
        y2 = y1 + panel_h

        draw_translucent_rect(frame, (x1, y1), (x2, y2), alpha=0.65)

        cy = y1 + 18
        for label, value in self._lines:
            cv2.putText(frame, label, (x1 + 8, cy),
                        FONT, 0.4, GRAY, 1, cv2.LINE_AA)
            vw = cv2.getTextSize(value, FONT, 0.4, 1)[0][0]
            cv2.putText(frame, value, (x2 - vw - 8, cy),
                        FONT, 0.4, WHITE, 1, cv2.LINE_AA)
            cy += line_h

        return frame


# ─── Crosshair ───────────────────────────────────────────────────────────────

class Crosshair(Widget):
    """Center crosshair / focus indicator."""

    def __init__(self, size: int = 20, color: tuple = GREEN, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.color = color

    def draw(self, frame: np.ndarray, context: dict) -> np.ndarray:
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        s = self.size
        cv2.line(frame, (cx - s, cy), (cx - s // 3, cy), self.color, 1)
        cv2.line(frame, (cx + s // 3, cy), (cx + s, cy), self.color, 1)
        cv2.line(frame, (cx, cy - s), (cx, cy - s // 3), self.color, 1)
        cv2.line(frame, (cx, cy + s // 3), (cx, cy + s), self.color, 1)
        return frame


# ─── Tracked Object Labels ───────────────────────────────────────────────

class TrackedObjectLabels(Widget):
    """Draws bounding boxes with persistent track IDs and smooth animation."""

    def __init__(self, show_confidence: bool = True, show_id: bool = True,
                 min_confidence: float = 0.3, use_smooth: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.show_confidence = show_confidence
        self.show_id = show_id
        self.min_confidence = min_confidence
        self.use_smooth = use_smooth

    def draw(self, frame: np.ndarray, context: dict) -> np.ndarray:
        tracked = context.get("tracked_objects")
        if not tracked:
            return frame

        for obj in tracked:
            if obj.confidence < self.min_confidence:
                continue

            # Use smoothed bbox for stable rendering
            if self.use_smooth:
                x1, y1, x2, y2 = [int(v) for v in obj.smooth_bbox]
            else:
                x1, y1, x2, y2 = obj.bbox

            color = class_color(obj.class_name)

            # Thicker box for objects tracked for many frames
            thickness = 2 if obj.frames_seen < 10 else 3
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Build label
            parts = []
            if self.show_id:
                parts.append(f"#{obj.track_id}")
            parts.append(obj.class_name)
            if self.show_confidence:
                parts.append(f"{obj.confidence:.0%}")
            label = " ".join(parts)

            lw, lh = cv2.getTextSize(label, FONT, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        FONT, 0.5, BLACK, 1, cv2.LINE_AA)

        return frame


# ─── Hand Skeleton ────────────────────────────────────────────────────────

# MediaPipe hand connections for skeleton drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm
]


class HandSkeleton(Widget):
    """Draws hand landmark skeletons with gesture labels."""

    def __init__(self, show_gesture: bool = True, show_landmarks: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.show_gesture = show_gesture
        self.show_landmarks = show_landmarks

    def draw(self, frame: np.ndarray, context: dict) -> np.ndarray:
        hands = context.get("hands")
        if not hands:
            return frame

        h, w = frame.shape[:2]

        for hand in hands:
            if not hand.landmarks:
                continue

            # Convert normalized landmarks to pixel coords
            pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in hand.landmarks]

            # Draw connections
            for i, j in HAND_CONNECTIONS:
                if i < len(pts) and j < len(pts):
                    cv2.line(frame, pts[i], pts[j], CYAN, 2, cv2.LINE_AA)

            # Draw landmark points
            if self.show_landmarks:
                for k, pt in enumerate(pts):
                    # Fingertips get larger dots
                    radius = 5 if k in (4, 8, 12, 16, 20) else 3
                    color = ACCENT if k in (4, 8, 12, 16, 20) else GREEN
                    cv2.circle(frame, pt, radius, color, -1)

            # Gesture label
            if self.show_gesture and hand.gesture.value != "unknown":
                label = f"{hand.handedness} {hand.gesture.value}"
                x1, y1, _, _ = hand.bbox
                cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                            FONT_BOLD, 0.6, CYAN, 1, cv2.LINE_AA)

        return frame


# ─── Widget Renderer ──────────────────────────────────────────────────────────

class WidgetRenderer:
    """Composes multiple widgets onto a frame.

    Usage:
        renderer = WidgetRenderer()
        renderer.add(StatusBar())
        renderer.add(ObjectLabels())
        renderer.add(SceneSummary())

        frame = renderer.render(frame, {"perception": perception, "fps": 30})
    """

    def __init__(self):
        self._widgets: list[Widget] = []

    def add(self, widget: Widget):
        self._widgets.append(widget)
        self._widgets.sort(key=lambda w: w.z_order)

    def remove(self, widget_type: type):
        self._widgets = [w for w in self._widgets if not isinstance(w, widget_type)]

    def get(self, widget_type: type) -> Optional[Widget]:
        for w in self._widgets:
            if isinstance(w, widget_type):
                return w
        return None

    def render(self, frame: np.ndarray, context: dict) -> np.ndarray:
        """Render all enabled widgets onto the frame."""
        for widget in self._widgets:
            if widget.enabled:
                frame = widget.draw(frame, context)
        return frame

    @property
    def widgets(self) -> list[Widget]:
        return list(self._widgets)
