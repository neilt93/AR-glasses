"""Hand pose detection using MediaPipe Hands.

Detects hands in the frame and classifies basic gestures:
- open: all fingers extended
- closed/fist: all fingers curled
- pointing: index finger extended, others curled
- pinch: thumb + index finger close together
- peace: index + middle extended

Also provides hand landmarks for spatial reasoning (e.g., "hand is near object X").

Usage:
    hands = HandDetector()
    results = hands.detect(frame)
    for hand in results:
        print(f"{hand.gesture} hand at {hand.center}")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class Gesture(Enum):
    """Recognized hand gestures."""
    UNKNOWN = "unknown"
    OPEN = "open"
    CLOSED = "closed"
    POINTING = "pointing"
    PINCH = "pinch"
    PEACE = "peace"
    THUMBS_UP = "thumbs_up"


@dataclass
class HandResult:
    """Detection result for a single hand."""
    landmarks: list[tuple[float, float, float]]  # 21 (x, y, z) normalized landmarks
    handedness: str  # "Left" or "Right"
    confidence: float
    gesture: Gesture = Gesture.UNKNOWN
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # pixel coords

    @property
    def center(self) -> tuple[float, float]:
        """Center of the hand in normalized coords (0-1)."""
        if not self.landmarks:
            return (0.5, 0.5)
        xs = [lm[0] for lm in self.landmarks]
        ys = [lm[1] for lm in self.landmarks]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    @property
    def center_px(self) -> tuple[int, int]:
        """Center of the hand in pixel coords (from bbox)."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def wrist(self) -> tuple[float, float, float]:
        """Wrist landmark (index 0)."""
        return self.landmarks[0] if self.landmarks else (0.0, 0.0, 0.0)

    @property
    def index_tip(self) -> tuple[float, float, float]:
        """Index fingertip (landmark 8)."""
        return self.landmarks[8] if len(self.landmarks) > 8 else (0.0, 0.0, 0.0)

    @property
    def thumb_tip(self) -> tuple[float, float, float]:
        """Thumb tip (landmark 4)."""
        return self.landmarks[4] if len(self.landmarks) > 4 else (0.0, 0.0, 0.0)


# MediaPipe hand landmark indices
WRIST = 0
THUMB_TIP = 4
THUMB_IP = 3
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_MCP = 13
RING_TIP = 16
PINKY_MCP = 17
PINKY_TIP = 20


def _finger_extended(landmarks: list[tuple[float, float, float]],
                     tip: int, mcp: int) -> bool:
    """Check if a finger is extended (tip is above MCP in y-axis)."""
    if len(landmarks) <= max(tip, mcp):
        return False
    # In image coords, y increases downward, so tip.y < mcp.y = extended
    return landmarks[tip][1] < landmarks[mcp][1]


def _thumb_extended(landmarks: list[tuple[float, float, float]],
                    handedness: str) -> bool:
    """Check if thumb is extended (away from palm)."""
    if len(landmarks) <= THUMB_TIP:
        return False
    # For right hand, thumb extends left (smaller x); for left hand, extends right
    if handedness == "Right":
        return landmarks[THUMB_TIP][0] < landmarks[THUMB_IP][0]
    else:
        return landmarks[THUMB_TIP][0] > landmarks[THUMB_IP][0]


def _pinch_distance(landmarks: list[tuple[float, float, float]]) -> float:
    """Distance between thumb tip and index tip (normalized coords)."""
    if len(landmarks) <= max(THUMB_TIP, INDEX_TIP):
        return 1.0
    t = landmarks[THUMB_TIP]
    i = landmarks[INDEX_TIP]
    return ((t[0] - i[0]) ** 2 + (t[1] - i[1]) ** 2) ** 0.5


def classify_gesture(landmarks: list[tuple[float, float, float]],
                     handedness: str = "Right") -> Gesture:
    """Classify hand gesture from 21 landmarks."""
    if len(landmarks) < 21:
        return Gesture.UNKNOWN

    thumb = _thumb_extended(landmarks, handedness)
    index = _finger_extended(landmarks, INDEX_TIP, INDEX_MCP)
    middle = _finger_extended(landmarks, MIDDLE_TIP, MIDDLE_MCP)
    ring = _finger_extended(landmarks, RING_TIP, RING_MCP)
    pinky = _finger_extended(landmarks, PINKY_TIP, PINKY_MCP)

    fingers = [thumb, index, middle, ring, pinky]
    extended_count = sum(fingers)

    # Check pinch first (thumb + index close together)
    if _pinch_distance(landmarks) < 0.05:
        return Gesture.PINCH

    # Thumbs up: only thumb extended
    if thumb and not index and not middle and not ring and not pinky:
        return Gesture.THUMBS_UP

    # Pointing: only index extended
    if index and not middle and not ring and not pinky:
        return Gesture.POINTING

    # Peace: index + middle
    if index and middle and not ring and not pinky:
        return Gesture.PEACE

    # Open hand: 4+ fingers extended
    if extended_count >= 4:
        return Gesture.OPEN

    # Closed fist: 0-1 fingers extended
    if extended_count <= 1:
        return Gesture.CLOSED

    return Gesture.UNKNOWN


class HandDetector:
    """Detects hands using MediaPipe.

    Falls back gracefully when MediaPipe is not installed.
    """

    def __init__(
        self,
        max_hands: int = 2,
        min_confidence: float = 0.5,
        min_tracking: float = 0.5,
    ):
        self._max_hands = max_hands
        self._min_confidence = min_confidence
        self._min_tracking = min_tracking
        self._mp_hands = None
        self._hands = None
        self._ready = False
        self._load()

    def _load(self):
        """Load MediaPipe Hands model."""
        try:
            import mediapipe as mp
            self._mp_hands = mp.solutions.hands
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self._max_hands,
                min_detection_confidence=self._min_confidence,
                min_tracking_confidence=self._min_tracking,
            )
            self._ready = True
            print(f"[hands] MediaPipe Hands loaded (max {self._max_hands} hands)")
        except ImportError:
            print("[hands] mediapipe not installed — hand detection unavailable")
            print("[hands] Install with: pip install mediapipe")
        except Exception as e:
            print(f"[hands] Failed to load hand detector: {e}")

    def detect(self, frame: np.ndarray) -> list[HandResult]:
        """Detect hands in the frame."""
        if not self._ready or self._hands is None:
            return []

        import cv2
        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            return []

        h, w = frame.shape[:2]
        hand_results = []

        for i, hand_lms in enumerate(results.multi_hand_landmarks):
            # Extract landmarks as (x, y, z) normalized
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]

            # Handedness
            handedness = "Right"
            if results.multi_handedness and i < len(results.multi_handedness):
                handedness = results.multi_handedness[i].classification[0].label
                confidence = results.multi_handedness[i].classification[0].score
            else:
                confidence = 0.5

            # Compute bounding box in pixel coords
            xs = [int(lm[0] * w) for lm in landmarks]
            ys = [int(lm[1] * h) for lm in landmarks]
            pad = 20
            bbox = (
                max(0, min(xs) - pad),
                max(0, min(ys) - pad),
                min(w, max(xs) + pad),
                min(h, max(ys) + pad),
            )

            # Classify gesture
            gesture = classify_gesture(landmarks, handedness)

            hand_results.append(HandResult(
                landmarks=landmarks,
                handedness=handedness,
                confidence=confidence,
                gesture=gesture,
                bbox=bbox,
            ))

        return hand_results

    def close(self):
        """Release MediaPipe resources."""
        if self._hands:
            self._hands.close()

    @property
    def is_ready(self) -> bool:
        return self._ready
