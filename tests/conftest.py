"""Shared test fixtures for the AR Assembly Copilot test suite."""

import sys
import os
import pytest

# Ensure project root is on sys.path so `src.*` imports resolve
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.perception.detector import Detection
from src.state.scene_state import SceneState


def make_detection(class_name: str, bbox: tuple[int, int, int, int], confidence: float = 0.9) -> Detection:
    """Helper to create a Detection with auto class_id."""
    class_ids = {"base_plate": 0, "bracket": 1, "screw": 2, "screwdriver": 3, "servo": 4, "hand": 5}
    return Detection(
        class_name=class_name,
        class_id=class_ids.get(class_name, -1),
        bbox=bbox,
        confidence=confidence,
    )


def make_scene_state(**kwargs) -> SceneState:
    """Helper to create a SceneState with overrides."""
    return SceneState(**kwargs)


# ---------------------------------------------------------------------------
# Fixtures for commonly needed scene states
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_state() -> SceneState:
    return SceneState()


@pytest.fixture
def plate_only_state() -> SceneState:
    return SceneState(base_plate_visible=True)


@pytest.fixture
def bracket_held_state() -> SceneState:
    return SceneState(base_plate_visible=True, bracket_visible=True, hand_visible=True)


@pytest.fixture
def bracket_on_plate_state() -> SceneState:
    return SceneState(base_plate_visible=True, bracket_visible=True, bracket_on_plate=True)


@pytest.fixture
def first_screw_inserted_state() -> SceneState:
    return SceneState(
        base_plate_visible=True, bracket_visible=True, bracket_on_plate=True,
        screws_visible=1, screw_near_plate=True,
    )


@pytest.fixture
def first_screw_tightened_state() -> SceneState:
    return SceneState(
        base_plate_visible=True, bracket_visible=True, bracket_on_plate=True,
        screws_visible=1, screw_near_plate=True,
        screwdriver_visible=True, screwdriver_engaged=True,
    )


@pytest.fixture
def second_screw_inserted_state() -> SceneState:
    return SceneState(
        base_plate_visible=True, bracket_visible=True, bracket_on_plate=True,
        screws_visible=2, screw_near_plate=True,
    )


@pytest.fixture
def second_screw_tightened_state() -> SceneState:
    return SceneState(
        base_plate_visible=True, bracket_visible=True, bracket_on_plate=True,
        screws_visible=2, screw_near_plate=True,
        screwdriver_visible=True, screwdriver_engaged=True,
    )


@pytest.fixture
def verified_state() -> SceneState:
    return SceneState(
        base_plate_visible=True, bracket_visible=True, bracket_on_plate=True,
        screws_visible=2, screw_near_plate=True,
        screwdriver_visible=False, screwdriver_engaged=False,
    )
