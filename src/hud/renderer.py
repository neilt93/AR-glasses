"""HUD renderer — draws overlays on camera frames for the AR display."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.perception.detector import Detection
from src.task.step_estimator import StepEstimate
from src.instructions.engine import Guidance


# Color palette (BGR)
COLOR_GREEN = (0, 220, 0)
COLOR_YELLOW = (0, 220, 220)
COLOR_RED = (0, 0, 220)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_DARK_BG = (30, 30, 30)
COLOR_ACCENT = (255, 160, 0)  # blue-ish accent

# Class-specific box colors
CLASS_COLORS = {
    "base_plate": (180, 120, 0),
    "bracket": (0, 200, 100),
    "screw": (0, 180, 255),
    "screwdriver": (200, 0, 200),
    "servo": (255, 100, 0),
    "hand": (100, 100, 255),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


class HUDRenderer:
    """Renders the AR HUD overlay onto a camera frame."""

    def __init__(
        self,
        task_name: str = "Servo Bracket Assembly",
        show_detections: bool = True,
        show_confidence: bool = True,
        top_bar_height: int = 50,
        bottom_bar_height: int = 80,
    ):
        self.task_name = task_name
        self.show_detections = show_detections
        self.show_confidence = show_confidence
        self.top_bar_height = top_bar_height
        self.bottom_bar_height = bottom_bar_height

    def render(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        estimate: Optional[StepEstimate] = None,
        guidance: Optional[Guidance] = None,
    ) -> np.ndarray:
        """Render the full HUD onto a copy of the frame."""
        display = frame.copy()
        h, w = display.shape[:2]

        if self.show_detections:
            self._draw_detections(display, detections)

        self._draw_top_bar(display, w, estimate)
        self._draw_bottom_bar(display, w, h, guidance)
        self._draw_progress_bar(display, w, estimate)

        return display

    def _draw_detections(self, frame: np.ndarray, detections: list[Detection]):
        """Draw bounding boxes and labels for detected objects."""
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = CLASS_COLORS.get(det.class_name, COLOR_GREEN)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{det.class_name}"
            if self.show_confidence:
                label += f" {det.confidence:.0%}"
            label_size, _ = cv2.getTextSize(label, FONT, 0.5, 1)
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - 6),
                (x1 + label_size[0] + 4, y1),
                color,
                -1,
            )
            cv2.putText(
                frame, label,
                (x1 + 2, y1 - 4),
                FONT, 0.5, COLOR_BLACK, 1, cv2.LINE_AA,
            )

    def _draw_top_bar(
        self, frame: np.ndarray, w: int, estimate: Optional[StepEstimate]
    ):
        """Draw the top status bar."""
        bar_h = self.top_bar_height
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Task name
        cv2.putText(
            frame, self.task_name,
            (10, 30),
            FONT_BOLD, 0.6, COLOR_WHITE, 1, cv2.LINE_AA,
        )

        if estimate is not None:
            # Step label
            step_text = estimate.step_label
            cv2.putText(
                frame, step_text,
                (w // 3, 30),
                FONT, 0.55, COLOR_ACCENT, 1, cv2.LINE_AA,
            )

            # Confidence chip
            conf_text = f"Conf: {estimate.confidence:.0%}"
            cv2.putText(
                frame, conf_text,
                (w - 130, 30),
                FONT, 0.45, COLOR_YELLOW, 1, cv2.LINE_AA,
            )

    def _draw_bottom_bar(
        self, frame: np.ndarray, w: int, h: int, guidance: Optional[Guidance]
    ):
        """Draw the bottom instruction/warning bar."""
        bar_h = self.bottom_bar_height
        y_start = h - bar_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_start), (w, h), COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        if guidance is None:
            cv2.putText(
                frame, "Waiting for task to start...",
                (10, y_start + 30),
                FONT, 0.55, COLOR_WHITE, 1, cv2.LINE_AA,
            )
            return

        # Main instruction
        cv2.putText(
            frame, guidance.instruction,
            (10, y_start + 28),
            FONT, 0.55, COLOR_WHITE, 1, cv2.LINE_AA,
        )

        # Next hint
        if guidance.next_hint:
            cv2.putText(
                frame, guidance.next_hint,
                (10, y_start + 52),
                FONT, 0.45, COLOR_ACCENT, 1, cv2.LINE_AA,
            )

        # Warnings
        if guidance.warnings:
            warn_text = " | ".join(guidance.warnings)
            cv2.putText(
                frame, warn_text,
                (10, y_start + 72),
                FONT, 0.4, COLOR_RED, 1, cv2.LINE_AA,
            )

    def _draw_progress_bar(
        self, frame: np.ndarray, w: int, estimate: Optional[StepEstimate]
    ):
        """Draw a thin progress bar below the top bar."""
        if estimate is None:
            return
        y = self.top_bar_height
        bar_h = 4
        progress_w = int(w * (estimate.progress_pct / 100.0))
        cv2.rectangle(frame, (0, y), (w, y + bar_h), COLOR_DARK_BG, -1)
        cv2.rectangle(frame, (0, y), (progress_w, y + bar_h), COLOR_GREEN, -1)
