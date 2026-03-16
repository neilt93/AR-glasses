"""Spatial awareness — understands relationships between objects.

Computes spatial relationships between detected objects and provides
contextual insights like "phone is on the desk" or "cup near laptop".

Used by the brain to generate smarter notifications and by the
info panel to show spatial context.

Usage:
    spatial = SpatialAnalyzer()
    relations = spatial.analyze(perception)
    for rel in relations:
        print(f"{rel.subject} {rel.relation} {rel.object}")
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.perception.detector import Detection


class SpatialRelation(Enum):
    """Types of spatial relationships between objects."""
    NEAR = "near"
    ON_TOP = "on top of"
    BELOW = "below"
    LEFT_OF = "left of"
    RIGHT_OF = "right of"
    INSIDE = "inside"
    HOLDING = "holding"


@dataclass
class ObjectRelation:
    """A spatial relationship between two objects."""
    subject: str
    relation: SpatialRelation
    object: str
    confidence: float = 1.0

    def describe(self) -> str:
        return f"{self.subject} {self.relation.value} {self.object}"


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _overlap_ratio(inner: tuple[int, int, int, int],
                   outer: tuple[int, int, int, int]) -> float:
    """What fraction of inner is inside outer."""
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = _bbox_area(inner)
    return inter / area if area > 0 else 0.0


def _distance(a: tuple[int, int, int, int],
              b: tuple[int, int, int, int]) -> float:
    """Euclidean distance between bbox centers."""
    ca, cb = _bbox_center(a), _bbox_center(b)
    return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5


class SpatialAnalyzer:
    """Analyzes spatial relationships between detected objects."""

    def __init__(
        self,
        near_threshold: float = 150.0,
        overlap_threshold: float = 0.3,
    ):
        self._near_threshold = near_threshold
        self._overlap_threshold = overlap_threshold

    def analyze(self, detections: list[Detection]) -> list[ObjectRelation]:
        """Find all spatial relationships between objects."""
        if len(detections) < 2:
            return []

        relations = []
        for i, a in enumerate(detections):
            for j, b in enumerate(detections):
                if i >= j:
                    continue
                rels = self._relate(a, b)
                relations.extend(rels)

        return relations

    def _relate(self, a: Detection, b: Detection) -> list[ObjectRelation]:
        """Find relationships between two objects."""
        rels = []
        ca = _bbox_center(a.bbox)
        cb = _bbox_center(b.bbox)
        dist = _distance(a.bbox, b.bbox)

        # Check containment
        a_in_b = _overlap_ratio(a.bbox, b.bbox)
        b_in_a = _overlap_ratio(b.bbox, a.bbox)

        if a_in_b > self._overlap_threshold:
            rels.append(ObjectRelation(a.class_name, SpatialRelation.INSIDE, b.class_name))
        elif b_in_a > self._overlap_threshold:
            rels.append(ObjectRelation(b.class_name, SpatialRelation.INSIDE, a.class_name))

        # Check on-top / below (vertical stacking)
        if a.bbox[3] <= b.bbox[1] + 20 and abs(ca[0] - cb[0]) < 100:
            # a is above b (remember: y increases downward)
            rels.append(ObjectRelation(a.class_name, SpatialRelation.ON_TOP, b.class_name))
        elif b.bbox[3] <= a.bbox[1] + 20 and abs(ca[0] - cb[0]) < 100:
            rels.append(ObjectRelation(b.class_name, SpatialRelation.ON_TOP, a.class_name))

        # Check holding (hand near small object)
        if a.class_name == "person" and _bbox_area(b.bbox) < _bbox_area(a.bbox) * 0.3:
            if dist < self._near_threshold:
                rels.append(ObjectRelation("person", SpatialRelation.HOLDING, b.class_name))

        # Check nearness
        if dist < self._near_threshold and not rels:
            rels.append(ObjectRelation(a.class_name, SpatialRelation.NEAR, b.class_name))

        # Left/right (only if they're roughly at the same height)
        if abs(ca[1] - cb[1]) < 80 and not rels:
            if ca[0] < cb[0] - 50:
                rels.append(ObjectRelation(a.class_name, SpatialRelation.LEFT_OF, b.class_name))
            elif ca[0] > cb[0] + 50:
                rels.append(ObjectRelation(a.class_name, SpatialRelation.RIGHT_OF, b.class_name))

        return rels

    def describe_scene(self, detections: list[Detection], max_relations: int = 3) -> str:
        """Generate a natural language scene description from spatial relationships."""
        relations = self.analyze(detections)
        if not relations:
            return ""
        descriptions = [r.describe() for r in relations[:max_relations]]
        return "; ".join(descriptions)
