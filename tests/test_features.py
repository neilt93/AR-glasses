"""Tests for spatial feature extraction (src/state/features.py)."""

import pytest
from src.state.features import (
    bbox_center, bbox_area, bbox_iou, bbox_overlap_ratio,
    distance_between, is_near, is_overlapping,
)
from tests.conftest import make_detection


class TestBboxCenter:
    def test_simple(self):
        assert bbox_center((0, 0, 100, 100)) == (50.0, 50.0)

    def test_offset(self):
        assert bbox_center((10, 20, 30, 40)) == (20.0, 30.0)

    def test_zero_size(self):
        assert bbox_center((5, 5, 5, 5)) == (5.0, 5.0)


class TestBboxArea:
    def test_normal(self):
        assert bbox_area((0, 0, 10, 10)) == 100

    def test_zero_width(self):
        assert bbox_area((0, 0, 0, 10)) == 0

    def test_zero_height(self):
        assert bbox_area((0, 0, 10, 0)) == 0

    def test_inverted_returns_zero(self):
        # x2 < x1 → max(0, ...) clamps to 0
        assert bbox_area((10, 10, 5, 5)) == 0


class TestBboxIoU:
    def test_identical(self):
        box = (0, 0, 10, 10)
        assert bbox_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)

    def test_partial_overlap(self):
        iou = bbox_iou((0, 0, 10, 10), (5, 5, 15, 15))
        # intersection = 5*5 = 25, union = 100+100-25 = 175
        assert iou == pytest.approx(25 / 175)

    def test_one_inside_other(self):
        iou = bbox_iou((0, 0, 20, 20), (5, 5, 10, 10))
        # intersection = 5*5 = 25, union = 400 + 25 - 25 = 400
        assert iou == pytest.approx(25 / 400)

    def test_zero_area_boxes(self):
        assert bbox_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


class TestBboxOverlapRatio:
    def test_fully_inside(self):
        inner = (5, 5, 10, 10)  # area = 25
        outer = (0, 0, 20, 20)  # area = 400
        assert bbox_overlap_ratio(inner, outer) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert bbox_overlap_ratio((0, 0, 5, 5), (10, 10, 20, 20)) == pytest.approx(0.0)

    def test_half_overlap(self):
        inner = (0, 0, 10, 10)  # area = 100
        outer = (5, 0, 15, 10)  # overlap = 5*10 = 50
        assert bbox_overlap_ratio(inner, outer) == pytest.approx(0.5)

    def test_zero_area_inner(self):
        assert bbox_overlap_ratio((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0


class TestDistanceBetween:
    def test_same_position(self):
        d1 = make_detection("screw", (0, 0, 10, 10))
        d2 = make_detection("screw", (0, 0, 10, 10))
        assert distance_between(d1, d2) == pytest.approx(0.0)

    def test_known_distance(self):
        d1 = make_detection("screw", (0, 0, 10, 10))      # center (5, 5)
        d2 = make_detection("screwdriver", (6, 8, 16, 18)) # center (11, 13)
        # distance = sqrt(36 + 64) = 10
        assert distance_between(d1, d2) == pytest.approx(10.0)


class TestIsNear:
    def test_near(self):
        d1 = make_detection("screw", (0, 0, 10, 10))
        d2 = make_detection("screwdriver", (10, 0, 20, 10))
        assert is_near(d1, d2, threshold=20.0) is True

    def test_far(self):
        d1 = make_detection("screw", (0, 0, 10, 10))
        d2 = make_detection("screwdriver", (200, 200, 210, 210))
        assert is_near(d1, d2, threshold=80.0) is False


class TestIsOverlapping:
    def test_overlapping(self):
        d1 = make_detection("bracket", (0, 0, 100, 100))
        d2 = make_detection("base_plate", (0, 0, 100, 100))
        assert is_overlapping(d1, d2) is True

    def test_not_overlapping(self):
        d1 = make_detection("bracket", (0, 0, 10, 10))
        d2 = make_detection("base_plate", (100, 100, 200, 200))
        assert is_overlapping(d1, d2) is False
