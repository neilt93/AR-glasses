"""Tests for scene state extraction (src/state/scene_state.py)."""

import pytest
from src.state.scene_state import SceneState, SceneStateExtractor
from tests.conftest import make_detection


class TestSceneStateToDict:
    def test_default_state(self):
        s = SceneState()
        d = s.to_dict()
        assert d["base_plate_visible"] is False
        assert d["screws_visible"] == 0
        assert "raw_detections" not in d  # excluded from dict


class TestSceneStateExtractor:
    def setup_method(self):
        self.extractor = SceneStateExtractor(
            overlap_threshold=0.2,
            near_threshold=80.0,
            engage_threshold=40.0,
        )

    def test_empty_detections(self):
        state = self.extractor.extract([])
        assert state.base_plate_visible is False
        assert state.bracket_visible is False
        assert state.screws_visible == 0

    def test_single_plate(self):
        dets = [make_detection("base_plate", (100, 200, 400, 450))]
        state = self.extractor.extract(dets)
        assert state.base_plate_visible is True
        assert state.bracket_visible is False

    def test_presence_flags(self):
        dets = [
            make_detection("base_plate", (0, 0, 100, 100)),
            make_detection("bracket", (10, 10, 50, 50)),
            make_detection("screw", (20, 20, 30, 30)),
            make_detection("screw", (40, 40, 50, 50)),
            make_detection("screwdriver", (60, 60, 80, 80)),
            make_detection("servo", (70, 70, 90, 90)),
            make_detection("hand", (80, 80, 100, 100)),
        ]
        state = self.extractor.extract(dets)
        assert state.base_plate_visible is True
        assert state.bracket_visible is True
        assert state.screwdriver_visible is True
        assert state.servo_visible is True
        assert state.hand_visible is True
        assert state.screws_visible == 2

    def test_bracket_on_plate_when_overlapping(self):
        # Bracket bbox is inside the plate bbox → high overlap ratio
        plate = make_detection("base_plate", (0, 0, 200, 200))
        bracket = make_detection("bracket", (20, 20, 100, 100))
        state = self.extractor.extract([plate, bracket])
        assert state.bracket_on_plate is True

    def test_bracket_not_on_plate_when_far(self):
        plate = make_detection("base_plate", (0, 0, 100, 100))
        bracket = make_detection("bracket", (300, 300, 400, 400))
        state = self.extractor.extract([plate, bracket])
        assert state.bracket_on_plate is False

    def test_screw_near_plate(self):
        plate = make_detection("base_plate", (100, 100, 200, 200))    # center (150, 150)
        screw = make_detection("screw", (140, 140, 160, 160))          # center (150, 150)
        state = self.extractor.extract([plate, screw])
        assert state.screw_near_plate is True

    def test_screw_far_from_plate(self):
        plate = make_detection("base_plate", (0, 0, 50, 50))           # center (25, 25)
        screw = make_detection("screw", (400, 400, 420, 420))          # center (410, 410)
        state = self.extractor.extract([plate, screw])
        assert state.screw_near_plate is False

    def test_screwdriver_engaged(self):
        # Screwdriver very close to screw → engaged
        screw = make_detection("screw", (100, 100, 120, 120))          # center (110, 110)
        driver = make_detection("screwdriver", (105, 105, 125, 125))   # center (115, 115)
        state = self.extractor.extract([screw, driver])
        assert state.screwdriver_engaged is True
        assert state.screwdriver_near_screw is True

    def test_screwdriver_near_but_not_engaged(self):
        screw = make_detection("screw", (100, 100, 110, 110))          # center (105, 105)
        driver = make_detection("screwdriver", (150, 100, 200, 110))   # center (175, 105)
        state = self.extractor.extract([screw, driver])
        # distance = 70, which is < 80 (near) but > 40 (engage)
        assert state.screwdriver_near_screw is True
        assert state.screwdriver_engaged is False

    def test_raw_detections_stored(self):
        dets = [make_detection("hand", (0, 0, 50, 50))]
        state = self.extractor.extract(dets)
        assert len(state.raw_detections) == 1
        assert state.raw_detections[0].class_name == "hand"
