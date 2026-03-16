"""Tests for object tracker (src/perception/tracker.py)."""

import pytest

from src.perception.tracker import ObjectTracker, TrackedObject, _iou, _centroid_distance
from src.perception.detector import Detection


class TestIoU:
    def test_no_overlap(self):
        assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_full_overlap(self):
        assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_partial_overlap(self):
        iou = _iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert 0.1 < iou < 0.2  # 25/175 ≈ 0.143

    def test_zero_area(self):
        assert _iou((0, 0, 0, 0), (0, 0, 10, 10)) == 0.0

    def test_contained(self):
        iou = _iou((0, 0, 100, 100), (25, 25, 75, 75))
        assert 0.2 < iou < 0.3  # 2500/10000 = 0.25


class TestCentroidDistance:
    def test_same_point(self):
        assert _centroid_distance((5.0, 5.0), (5.0, 5.0)) == 0.0

    def test_horizontal(self):
        assert _centroid_distance((0.0, 0.0), (3.0, 4.0)) == 5.0


class TestTrackedObject:
    def test_centroid(self):
        obj = TrackedObject(0, "cup", 41, (10, 20, 30, 40), 0.9)
        assert obj.centroid == (20.0, 30.0)

    def test_as_detection(self):
        obj = TrackedObject(5, "laptop", 63, (0, 0, 100, 100), 0.85)
        det = obj.as_detection()
        assert det.class_name == "laptop"
        assert det.bbox == (0, 0, 100, 100)


class TestObjectTracker:
    def test_register_new_objects(self):
        tracker = ObjectTracker()
        dets = [
            Detection("cup", 41, (10, 10, 50, 50), 0.9),
            Detection("laptop", 63, (100, 100, 300, 300), 0.85),
        ]
        tracked = tracker.update(dets)
        assert len(tracked) == 2
        assert tracked[0].track_id == 0
        assert tracked[1].track_id == 1

    def test_persistent_ids(self):
        tracker = ObjectTracker()
        dets1 = [Detection("cup", 41, (10, 10, 50, 50), 0.9)]
        t1 = tracker.update(dets1)
        tid = t1[0].track_id

        # Same object, slightly moved
        dets2 = [Detection("cup", 41, (12, 12, 52, 52), 0.88)]
        t2 = tracker.update(dets2)
        assert t2[0].track_id == tid

    def test_frames_seen_increments(self):
        tracker = ObjectTracker()
        dets = [Detection("cup", 41, (10, 10, 50, 50), 0.9)]
        tracker.update(dets)
        t = tracker.update([Detection("cup", 41, (11, 11, 51, 51), 0.9)])
        assert t[0].frames_seen == 2

    def test_object_disappears(self):
        tracker = ObjectTracker(max_disappeared=2)
        dets = [Detection("cup", 41, (10, 10, 50, 50), 0.9)]
        tracker.update(dets)

        # Object disappears
        tracker.update([])
        assert tracker.active_count == 1  # still alive (1 frame missing)

        tracker.update([])
        assert tracker.active_count == 1  # still alive (2 frames missing)

        tracker.update([])
        assert tracker.active_count == 0  # gone (exceeded max_disappeared)

    def test_new_object_gets_new_id(self):
        tracker = ObjectTracker()
        dets1 = [Detection("cup", 41, (10, 10, 50, 50), 0.9)]
        t1 = tracker.update(dets1)

        # Different object far away
        dets2 = [
            Detection("cup", 41, (10, 10, 50, 50), 0.9),
            Detection("laptop", 63, (500, 500, 800, 800), 0.85),
        ]
        t2 = tracker.update(dets2)
        assert len(t2) == 2
        ids = {t.track_id for t in t2}
        assert t1[0].track_id in ids  # cup kept its ID

    def test_class_match_required(self):
        """Different class at same location should NOT match the existing track."""
        tracker = ObjectTracker(max_disappeared=0)  # remove missing tracks immediately
        dets1 = [Detection("cup", 41, (10, 10, 50, 50), 0.9)]
        t1 = tracker.update(dets1)
        cup_id = t1[0].track_id

        # Same bbox but different class — cup disappears, bottle appears
        dets2 = [Detection("bottle", 39, (10, 10, 50, 50), 0.9)]
        t2 = tracker.update(dets2)
        # Bottle should get a new ID (cup was removed since max_disappeared=0)
        bottle_ids = [t.track_id for t in t2 if t.class_name == "bottle"]
        assert len(bottle_ids) == 1
        assert bottle_ids[0] != cup_id

    def test_smooth_bbox(self):
        tracker = ObjectTracker(smooth_factor=0.5)
        dets1 = [Detection("cup", 41, (0, 0, 100, 100), 0.9)]
        t1 = tracker.update(dets1)
        # First frame: smooth_bbox should be the actual bbox
        assert t1[0].smooth_bbox == (0.0, 0.0, 100.0, 100.0)

        # Second frame: object jumps to (50, 50, 150, 150)
        dets2 = [Detection("cup", 41, (50, 50, 150, 150), 0.9)]
        t2 = tracker.update(dets2)
        # Smooth bbox should be between old and new
        sx1, sy1, sx2, sy2 = t2[0].smooth_bbox
        assert 20 < sx1 < 30  # (0 * 0.5 + 50 * 0.5) = 25
        assert 120 < sx2 < 130

    def test_empty_update(self):
        tracker = ObjectTracker()
        result = tracker.update([])
        assert result == []

    def test_reset(self):
        tracker = ObjectTracker()
        tracker.update([Detection("cup", 41, (10, 10, 50, 50), 0.9)])
        assert tracker.active_count == 1
        tracker.reset()
        assert tracker.active_count == 0

    def test_entered_exited(self):
        tracker = ObjectTracker()
        t1 = tracker.update([
            Detection("cup", 41, (10, 10, 50, 50), 0.9),
        ])
        t2 = tracker.update([
            Detection("cup", 41, (12, 12, 52, 52), 0.9),
            Detection("laptop", 63, (200, 200, 400, 400), 0.85),
        ])
        entered = tracker.entered(t1, t2)
        assert len(entered) == 1
        assert entered[0].class_name == "laptop"

        exited = tracker.exited(t1, t2)
        assert len(exited) == 0

    def test_multiple_same_class(self):
        """Two objects of the same class should get different track IDs."""
        tracker = ObjectTracker()
        dets = [
            Detection("cup", 41, (10, 10, 50, 50), 0.9),
            Detection("cup", 41, (200, 200, 250, 250), 0.8),
        ]
        tracked = tracker.update(dets)
        assert len(tracked) == 2
        assert tracked[0].track_id != tracked[1].track_id
