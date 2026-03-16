"""Scene state extraction — converts detections into structured task state."""

from dataclasses import dataclass, field
from typing import Optional

from src.perception.detector import Detection
from src.state.features import (
    bbox_overlap_ratio,
    distance_between,
    is_near,
    is_overlapping,
)


@dataclass
class SceneState:
    """Structured representation of what is happening in the scene."""
    base_plate_visible: bool = False
    bracket_visible: bool = False
    screwdriver_visible: bool = False
    servo_visible: bool = False
    screws_visible: int = 0
    hand_visible: bool = False

    bracket_on_plate: bool = False
    screw_near_plate: bool = False
    screwdriver_near_screw: bool = False
    screwdriver_engaged: bool = False

    raw_detections: list[Detection] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "base_plate_visible": self.base_plate_visible,
            "bracket_visible": self.bracket_visible,
            "screwdriver_visible": self.screwdriver_visible,
            "servo_visible": self.servo_visible,
            "screws_visible": self.screws_visible,
            "hand_visible": self.hand_visible,
            "bracket_on_plate": self.bracket_on_plate,
            "screw_near_plate": self.screw_near_plate,
            "screwdriver_near_screw": self.screwdriver_near_screw,
            "screwdriver_engaged": self.screwdriver_engaged,
        }


class SceneStateExtractor:
    """Extracts structured scene state from a list of detections."""

    def __init__(
        self,
        overlap_threshold: float = 0.2,
        near_threshold: float = 80.0,
        engage_threshold: float = 40.0,
    ):
        self.overlap_threshold = overlap_threshold
        self.near_threshold = near_threshold
        self.engage_threshold = engage_threshold

    def extract(self, detections: list[Detection]) -> SceneState:
        """Build a SceneState from raw detections."""
        state = SceneState(raw_detections=detections)

        # Index detections by class
        by_class: dict[str, list[Detection]] = {}
        for det in detections:
            by_class.setdefault(det.class_name, []).append(det)

        # Presence flags
        state.base_plate_visible = "base_plate" in by_class
        state.bracket_visible = "bracket" in by_class
        state.screwdriver_visible = "screwdriver" in by_class
        state.servo_visible = "servo" in by_class
        state.hand_visible = "hand" in by_class
        state.screws_visible = len(by_class.get("screw", []))

        # Spatial relationships
        plates = by_class.get("base_plate", [])
        brackets = by_class.get("bracket", [])
        screws = by_class.get("screw", [])
        drivers = by_class.get("screwdriver", [])

        # Bracket on plate?
        if plates and brackets:
            plate = plates[0]
            bracket = brackets[0]
            overlap = bbox_overlap_ratio(bracket.bbox, plate.bbox)
            state.bracket_on_plate = overlap > self.overlap_threshold

        # Screw near plate?
        if plates and screws:
            plate = plates[0]
            for screw in screws:
                if is_near(screw, plate, self.near_threshold):
                    state.screw_near_plate = True
                    break

        # Screwdriver near/engaged with screw?
        if drivers and screws:
            driver = drivers[0]
            for screw in screws:
                dist = distance_between(driver, screw)
                if dist < self.near_threshold:
                    state.screwdriver_near_screw = True
                if dist < self.engage_threshold:
                    state.screwdriver_engaged = True

        return state
