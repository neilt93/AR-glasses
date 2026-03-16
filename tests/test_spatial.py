"""Tests for spatial analyzer (src/assistant/spatial.py)."""

import pytest

from src.assistant.spatial import (
    SpatialAnalyzer, SpatialRelation, ObjectRelation,
    _bbox_center, _bbox_area, _overlap_ratio, _distance,
)
from src.perception.detector import Detection


class TestHelpers:
    def test_bbox_center(self):
        assert _bbox_center((0, 0, 100, 100)) == (50.0, 50.0)
        assert _bbox_center((10, 20, 30, 40)) == (20.0, 30.0)

    def test_bbox_area(self):
        assert _bbox_area((0, 0, 10, 10)) == 100
        assert _bbox_area((0, 0, 0, 0)) == 0

    def test_overlap_ratio_full(self):
        ratio = _overlap_ratio((10, 10, 50, 50), (0, 0, 100, 100))
        assert ratio == pytest.approx(1.0)

    def test_overlap_ratio_none(self):
        ratio = _overlap_ratio((0, 0, 10, 10), (50, 50, 100, 100))
        assert ratio == 0.0

    def test_distance(self):
        d = _distance((0, 0, 10, 10), (100, 0, 110, 10))
        assert d == pytest.approx(100.0)


class TestObjectRelation:
    def test_describe(self):
        rel = ObjectRelation("cup", SpatialRelation.NEAR, "laptop")
        assert rel.describe() == "cup near laptop"

    def test_describe_on_top(self):
        rel = ObjectRelation("book", SpatialRelation.ON_TOP, "desk")
        assert rel.describe() == "book on top of desk"


class TestSpatialAnalyzer:
    def test_empty_detections(self):
        sa = SpatialAnalyzer()
        assert sa.analyze([]) == []

    def test_single_detection(self):
        sa = SpatialAnalyzer()
        dets = [Detection("cup", 41, (10, 10, 50, 50), 0.9)]
        assert sa.analyze(dets) == []

    def test_near_objects(self):
        sa = SpatialAnalyzer(near_threshold=200)
        dets = [
            Detection("cup", 41, (100, 100, 150, 150), 0.9),
            Detection("laptop", 63, (160, 100, 300, 200), 0.85),
        ]
        rels = sa.analyze(dets)
        assert len(rels) >= 1
        rel_types = {r.relation for r in rels}
        assert SpatialRelation.NEAR in rel_types or SpatialRelation.LEFT_OF in rel_types

    def test_containment(self):
        sa = SpatialAnalyzer()
        dets = [
            Detection("cup", 41, (120, 120, 180, 180), 0.9),  # inside laptop bbox
            Detection("laptop", 63, (100, 100, 300, 300), 0.85),
        ]
        rels = sa.analyze(dets)
        inside_rels = [r for r in rels if r.relation == SpatialRelation.INSIDE]
        assert len(inside_rels) >= 1
        assert inside_rels[0].subject == "cup"

    def test_vertical_stacking(self):
        sa = SpatialAnalyzer()
        dets = [
            Detection("cup", 41, (100, 50, 200, 100), 0.9),    # above
            Detection("book", 73, (100, 110, 200, 200), 0.85),  # below
        ]
        rels = sa.analyze(dets)
        on_top = [r for r in rels if r.relation == SpatialRelation.ON_TOP]
        assert len(on_top) >= 1

    def test_left_right(self):
        sa = SpatialAnalyzer(near_threshold=50)  # reduce near threshold
        dets = [
            Detection("cup", 41, (50, 200, 100, 250), 0.9),     # left
            Detection("bottle", 39, (300, 200, 350, 250), 0.85), # right
        ]
        rels = sa.analyze(dets)
        lr_rels = [r for r in rels
                   if r.relation in (SpatialRelation.LEFT_OF, SpatialRelation.RIGHT_OF)]
        assert len(lr_rels) >= 1

    def test_describe_scene(self):
        sa = SpatialAnalyzer(near_threshold=200)
        dets = [
            Detection("cup", 41, (100, 100, 150, 150), 0.9),
            Detection("laptop", 63, (160, 100, 300, 200), 0.85),
        ]
        desc = sa.describe_scene(dets)
        assert len(desc) > 0

    def test_describe_scene_empty(self):
        sa = SpatialAnalyzer()
        assert sa.describe_scene([]) == ""

    def test_holding_detection(self):
        sa = SpatialAnalyzer(near_threshold=200)
        dets = [
            Detection("person", 0, (50, 50, 300, 400), 0.9),     # large person
            Detection("cell phone", 67, (150, 200, 190, 250), 0.8),  # small phone near person
        ]
        rels = sa.analyze(dets)
        holding = [r for r in rels if r.relation == SpatialRelation.HOLDING]
        assert len(holding) >= 1
