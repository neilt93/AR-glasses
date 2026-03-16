"""Tests for depth estimation (src/perception/depth.py)."""

import pytest
import numpy as np

from src.perception.depth import DepthResult, DepthEstimator


class TestDepthResult:
    def _make_depth(self, h=480, w=640):
        """Create a gradient depth map (left=near, right=far)."""
        depth = np.linspace(0, 1, w, dtype=np.float32)
        depth = np.tile(depth, (h, 1))
        return DepthResult(depth_map=depth, raw_depth=depth.copy())

    def test_dimensions(self):
        result = self._make_depth()
        assert result.height == 480
        assert result.width == 640

    def test_min_max_depth(self):
        result = self._make_depth()
        assert result.min_depth == pytest.approx(0.0, abs=0.01)
        assert result.max_depth == pytest.approx(1.0, abs=0.01)

    def test_depth_at(self):
        result = self._make_depth()
        # Left edge should be near 0
        assert result.depth_at(0, 240) < 0.05
        # Right edge should be near 1
        assert result.depth_at(639, 240) > 0.95
        # Middle should be around 0.5
        assert 0.45 < result.depth_at(320, 240) < 0.55

    def test_depth_at_clamps(self):
        result = self._make_depth()
        # Out of bounds should clamp
        result.depth_at(-10, -10)  # should not raise
        result.depth_at(9999, 9999)  # should not raise

    def test_depth_in_bbox(self):
        result = self._make_depth()
        # Left half of image
        d = result.depth_in_bbox((0, 0, 320, 480))
        assert d < 0.4
        # Right half
        d = result.depth_in_bbox((320, 0, 640, 480))
        assert d > 0.6

    def test_depth_in_bbox_invalid(self):
        result = self._make_depth()
        d = result.depth_in_bbox((500, 500, 100, 100))  # inverted bbox
        assert d == 0.5

    def test_colormap(self):
        result = self._make_depth()
        colored = result.colormap()
        assert colored.shape == (480, 640, 3)
        assert colored.dtype == np.uint8

    def test_objects_by_depth(self):
        result = self._make_depth()
        bboxes = [
            (500, 200, 600, 300),  # right side (far)
            (50, 200, 150, 300),   # left side (near)
            (300, 200, 400, 300),  # middle
        ]
        names = ["far_obj", "near_obj", "mid_obj"]
        sorted_objs = result.objects_by_depth(bboxes, names)
        assert sorted_objs[0][0] == "near_obj"
        assert sorted_objs[-1][0] == "far_obj"

    def test_uniform_depth(self):
        depth = np.full((100, 100), 0.5, dtype=np.float32)
        result = DepthResult(depth_map=depth, raw_depth=depth.copy())
        assert result.min_depth == pytest.approx(0.5)
        assert result.max_depth == pytest.approx(0.5)


class TestDepthEstimator:
    def test_not_ready_without_torch(self):
        """Without torch, estimator should gracefully fail."""
        estimator = DepthEstimator.__new__(DepthEstimator)
        estimator._ready = False
        estimator._model = None
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = estimator.estimate(frame)
        assert result is None

    def test_is_ready_false(self):
        estimator = DepthEstimator.__new__(DepthEstimator)
        estimator._ready = False
        assert estimator.is_ready is False

    def test_close(self):
        estimator = DepthEstimator.__new__(DepthEstimator)
        estimator._ready = True
        estimator._model = "dummy"
        estimator.close()
        assert estimator.is_ready is False
        assert estimator._model is None
