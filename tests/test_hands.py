"""Tests for hand pose detection (src/perception/hands.py)."""

import pytest
import numpy as np

from src.perception.hands import (
    HandResult, Gesture, classify_gesture,
    _finger_extended, _thumb_extended, _pinch_distance,
    WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP,
    INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP,
)


def _make_landmarks(positions: dict[int, tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """Build a 21-landmark list with specified positions, rest at (0.5, 0.5, 0)."""
    lms = [(0.5, 0.5, 0.0)] * 21
    for idx, pos in positions.items():
        lms[idx] = pos
    return lms


class TestGestureClassification:
    def test_open_hand(self):
        """All fingers extended → OPEN."""
        lms = _make_landmarks({
            WRIST: (0.5, 0.9, 0),
            # Thumb extended (right hand: tip.x < ip.x)
            3: (0.35, 0.6, 0),  # THUMB_IP
            THUMB_TIP: (0.3, 0.55, 0),
            # All fingers: tip.y < mcp.y
            INDEX_MCP: (0.4, 0.6, 0), INDEX_TIP: (0.4, 0.2, 0),
            MIDDLE_MCP: (0.5, 0.6, 0), MIDDLE_TIP: (0.5, 0.2, 0),
            RING_MCP: (0.6, 0.6, 0), RING_TIP: (0.6, 0.2, 0),
            PINKY_MCP: (0.7, 0.6, 0), PINKY_TIP: (0.7, 0.2, 0),
        })
        assert classify_gesture(lms, "Right") == Gesture.OPEN

    def test_closed_fist(self):
        """All fingers curled → CLOSED."""
        lms = _make_landmarks({
            WRIST: (0.5, 0.9, 0),
            3: (0.35, 0.6, 0),
            THUMB_TIP: (0.45, 0.65, 0),  # not extended (tip.x > ip.x for right)
            INDEX_MCP: (0.4, 0.6, 0), INDEX_TIP: (0.55, 0.7, 0),  # tip below mcp, away from thumb
            MIDDLE_MCP: (0.5, 0.6, 0), MIDDLE_TIP: (0.5, 0.7, 0),
            RING_MCP: (0.6, 0.6, 0), RING_TIP: (0.6, 0.7, 0),
            PINKY_MCP: (0.7, 0.6, 0), PINKY_TIP: (0.7, 0.7, 0),
        })
        assert classify_gesture(lms, "Right") == Gesture.CLOSED

    def test_pointing(self):
        """Only index extended → POINTING."""
        lms = _make_landmarks({
            WRIST: (0.5, 0.9, 0),
            3: (0.35, 0.6, 0),
            THUMB_TIP: (0.4, 0.65, 0),  # not extended
            INDEX_MCP: (0.4, 0.6, 0), INDEX_TIP: (0.4, 0.2, 0),  # extended
            MIDDLE_MCP: (0.5, 0.6, 0), MIDDLE_TIP: (0.5, 0.7, 0),
            RING_MCP: (0.6, 0.6, 0), RING_TIP: (0.6, 0.7, 0),
            PINKY_MCP: (0.7, 0.6, 0), PINKY_TIP: (0.7, 0.7, 0),
        })
        assert classify_gesture(lms, "Right") == Gesture.POINTING

    def test_peace(self):
        """Index + middle extended → PEACE."""
        lms = _make_landmarks({
            WRIST: (0.5, 0.9, 0),
            3: (0.35, 0.6, 0),
            THUMB_TIP: (0.4, 0.65, 0),
            INDEX_MCP: (0.4, 0.6, 0), INDEX_TIP: (0.4, 0.2, 0),
            MIDDLE_MCP: (0.5, 0.6, 0), MIDDLE_TIP: (0.5, 0.2, 0),
            RING_MCP: (0.6, 0.6, 0), RING_TIP: (0.6, 0.7, 0),
            PINKY_MCP: (0.7, 0.6, 0), PINKY_TIP: (0.7, 0.7, 0),
        })
        assert classify_gesture(lms, "Right") == Gesture.PEACE

    def test_pinch(self):
        """Thumb + index tips close together → PINCH."""
        lms = _make_landmarks({
            WRIST: (0.5, 0.9, 0),
            3: (0.35, 0.6, 0),
            THUMB_TIP: (0.4, 0.4, 0),
            INDEX_TIP: (0.41, 0.41, 0),  # very close to thumb
            INDEX_MCP: (0.4, 0.6, 0),
            MIDDLE_MCP: (0.5, 0.6, 0), MIDDLE_TIP: (0.5, 0.7, 0),
            RING_MCP: (0.6, 0.6, 0), RING_TIP: (0.6, 0.7, 0),
            PINKY_MCP: (0.7, 0.6, 0), PINKY_TIP: (0.7, 0.7, 0),
        })
        assert classify_gesture(lms, "Right") == Gesture.PINCH

    def test_thumbs_up(self):
        """Only thumb extended → THUMBS_UP."""
        lms = _make_landmarks({
            WRIST: (0.5, 0.9, 0),
            3: (0.35, 0.6, 0),
            THUMB_TIP: (0.3, 0.55, 0),  # extended for right hand
            INDEX_MCP: (0.4, 0.6, 0), INDEX_TIP: (0.4, 0.7, 0),
            MIDDLE_MCP: (0.5, 0.6, 0), MIDDLE_TIP: (0.5, 0.7, 0),
            RING_MCP: (0.6, 0.6, 0), RING_TIP: (0.6, 0.7, 0),
            PINKY_MCP: (0.7, 0.6, 0), PINKY_TIP: (0.7, 0.7, 0),
        })
        assert classify_gesture(lms, "Right") == Gesture.THUMBS_UP

    def test_too_few_landmarks(self):
        lms = [(0.5, 0.5, 0.0)] * 10  # only 10 landmarks
        assert classify_gesture(lms) == Gesture.UNKNOWN


class TestHelperFunctions:
    def test_finger_extended(self):
        lms = [(0.5, 0.5, 0)] * 21
        lms[INDEX_TIP] = (0.4, 0.2, 0)    # tip above
        lms[INDEX_MCP] = (0.4, 0.6, 0)     # mcp below
        assert _finger_extended(lms, INDEX_TIP, INDEX_MCP) is True

    def test_finger_curled(self):
        lms = [(0.5, 0.5, 0)] * 21
        lms[INDEX_TIP] = (0.4, 0.7, 0)    # tip below
        lms[INDEX_MCP] = (0.4, 0.6, 0)
        assert _finger_extended(lms, INDEX_TIP, INDEX_MCP) is False

    def test_thumb_extended_right(self):
        lms = [(0.5, 0.5, 0)] * 21
        lms[THUMB_TIP] = (0.3, 0.5, 0)  # tip left of ip (extended for right hand)
        lms[3] = (0.35, 0.5, 0)         # THUMB_IP
        assert _thumb_extended(lms, "Right") is True

    def test_thumb_not_extended_right(self):
        lms = [(0.5, 0.5, 0)] * 21
        lms[THUMB_TIP] = (0.4, 0.5, 0)
        lms[3] = (0.35, 0.5, 0)
        assert _thumb_extended(lms, "Right") is False

    def test_pinch_distance_close(self):
        lms = [(0.5, 0.5, 0)] * 21
        lms[THUMB_TIP] = (0.5, 0.5, 0)
        lms[INDEX_TIP] = (0.51, 0.51, 0)
        assert _pinch_distance(lms) < 0.05

    def test_pinch_distance_far(self):
        lms = [(0.5, 0.5, 0)] * 21
        lms[THUMB_TIP] = (0.2, 0.2, 0)
        lms[INDEX_TIP] = (0.8, 0.8, 0)
        assert _pinch_distance(lms) > 0.5


class TestHandResult:
    def test_center(self):
        lms = [(0.3, 0.4, 0), (0.5, 0.6, 0)] + [(0.5, 0.5, 0)] * 19
        hand = HandResult(landmarks=lms, handedness="Right", confidence=0.9)
        cx, cy = hand.center
        assert 0.4 < cx < 0.6
        assert 0.4 < cy < 0.6

    def test_wrist(self):
        lms = [(0.1, 0.2, 0.3)] + [(0.5, 0.5, 0)] * 20
        hand = HandResult(landmarks=lms, handedness="Right", confidence=0.9)
        assert hand.wrist == (0.1, 0.2, 0.3)

    def test_index_tip(self):
        lms = [(0.5, 0.5, 0)] * 21
        lms[8] = (0.7, 0.1, 0)
        hand = HandResult(landmarks=lms, handedness="Right", confidence=0.9)
        assert hand.index_tip == (0.7, 0.1, 0)

    def test_center_px(self):
        hand = HandResult(
            landmarks=[(0.5, 0.5, 0)] * 21,
            handedness="Right",
            confidence=0.9,
            bbox=(100, 200, 300, 400),
        )
        assert hand.center_px == (200, 300)


class TestHandDetector:
    def test_no_mediapipe_returns_empty(self):
        """Without MediaPipe, detector should return empty results."""
        from src.perception.hands import HandDetector
        detector = HandDetector.__new__(HandDetector)
        detector._ready = False
        detector._hands = None
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert detector.detect(frame) == []

    def test_is_ready_false_without_mediapipe(self):
        from src.perception.hands import HandDetector
        detector = HandDetector.__new__(HandDetector)
        detector._ready = False
        assert detector.is_ready is False
