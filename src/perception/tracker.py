"""Object tracker — persistent IDs across frames.

Assigns stable IDs to detected objects so they can be tracked over time.
Uses centroid-based matching with IoU fallback. No external dependencies.

Each tracked object gets a unique ID that persists as long as the object
remains visible. When objects disappear and reappear, they get new IDs.

Usage:
    tracker = ObjectTracker(max_disappeared=10)
    tracked = tracker.update(detections)
    for obj in tracked:
        print(f"Object {obj.track_id}: {obj.class_name} at {obj.bbox}")
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.perception.detector import Detection


@dataclass
class TrackedObject:
    """A detection with a persistent tracking ID."""
    track_id: int
    class_name: str
    class_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    frames_seen: int = 1
    frames_missing: int = 0

    # Smoothed bbox for stable rendering
    smooth_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def smooth_centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.smooth_bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def as_detection(self) -> Detection:
        """Convert back to a Detection (for widgets that expect it)."""
        return Detection(
            class_name=self.class_name,
            class_id=self.class_id,
            bbox=self.bbox,
            confidence=self.confidence,
        )


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Compute intersection-over-union of two bboxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _centroid_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two centroids."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class ObjectTracker:
    """Centroid + IoU tracker for persistent object IDs.

    Algorithm:
    1. For each existing track, compute IoU and centroid distance to all detections
    2. Match tracks to detections greedily (highest IoU first, with class matching)
    3. Unmatched detections become new tracks
    4. Unmatched tracks increment their missing counter
    5. Tracks missing for > max_disappeared frames are removed
    """

    def __init__(
        self,
        max_disappeared: int = 15,  # frames before track is deleted
        iou_threshold: float = 0.2,  # minimum IoU for matching
        max_distance: float = 150.0,  # max centroid distance for fallback matching
        smooth_factor: float = 0.3,  # EMA factor for bbox smoothing (0=no smooth, 1=instant)
    ):
        self._max_disappeared = max_disappeared
        self._iou_threshold = iou_threshold
        self._max_distance = max_distance
        self._smooth_factor = smooth_factor
        self._next_id = 0
        self._tracks: dict[int, TrackedObject] = {}

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        """Match new detections to existing tracks and return tracked objects.

        Returns tracked objects sorted by track_id.
        """
        if not detections:
            # Mark all tracks as missing
            to_remove = []
            for tid, track in self._tracks.items():
                track.frames_missing += 1
                if track.frames_missing > self._max_disappeared:
                    to_remove.append(tid)
            for tid in to_remove:
                del self._tracks[tid]
            return sorted(self._tracks.values(), key=lambda t: t.track_id)

        if not self._tracks:
            # No existing tracks — register all detections as new
            for det in detections:
                self._register(det)
            return sorted(self._tracks.values(), key=lambda t: t.track_id)

        # Build cost matrix: tracks × detections
        track_ids = list(self._tracks.keys())
        n_tracks = len(track_ids)
        n_dets = len(detections)

        # Compute IoU and distance for all pairs
        iou_matrix = np.zeros((n_tracks, n_dets))
        dist_matrix = np.zeros((n_tracks, n_dets))
        class_match = np.zeros((n_tracks, n_dets), dtype=bool)

        for i, tid in enumerate(track_ids):
            track = self._tracks[tid]
            for j, det in enumerate(detections):
                iou_matrix[i, j] = _iou(track.bbox, det.bbox)
                dist_matrix[i, j] = _centroid_distance(
                    track.centroid,
                    ((det.bbox[0] + det.bbox[2]) / 2.0,
                     (det.bbox[1] + det.bbox[3]) / 2.0),
                )
                class_match[i, j] = track.class_name == det.class_name

        # Greedy matching: prefer IoU, require class match
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        # Pass 1: Match by IoU (with class constraint)
        while True:
            # Mask already matched
            mask = np.ones_like(iou_matrix, dtype=bool)
            for i, tid in enumerate(track_ids):
                if tid in matched_tracks:
                    mask[i, :] = False
            for j in matched_dets:
                mask[:, j] = False

            valid = iou_matrix * mask * class_match
            if valid.max() < self._iou_threshold:
                break

            i, j = np.unravel_index(valid.argmax(), valid.shape)
            tid = track_ids[i]
            self._update_track(tid, detections[j])
            matched_tracks.add(tid)
            matched_dets.add(j)

        # Pass 2: Match remaining by centroid distance (with class constraint)
        for i, tid in enumerate(track_ids):
            if tid in matched_tracks:
                continue
            best_j = -1
            best_dist = self._max_distance
            for j, det in enumerate(detections):
                if j in matched_dets:
                    continue
                if not class_match[i, j]:
                    continue
                d = dist_matrix[i, j]
                if d < best_dist:
                    best_dist = d
                    best_j = j
            if best_j >= 0:
                self._update_track(tid, detections[best_j])
                matched_tracks.add(tid)
                matched_dets.add(best_j)

        # Register unmatched detections as new tracks
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._register(det)

        # Increment missing counter for unmatched tracks
        to_remove = []
        for tid in track_ids:
            if tid not in matched_tracks:
                self._tracks[tid].frames_missing += 1
                if self._tracks[tid].frames_missing > self._max_disappeared:
                    to_remove.append(tid)
        for tid in to_remove:
            del self._tracks[tid]

        return sorted(self._tracks.values(), key=lambda t: t.track_id)

    def _register(self, det: Detection):
        """Register a new tracked object."""
        bbox_f = tuple(float(v) for v in det.bbox)
        self._tracks[self._next_id] = TrackedObject(
            track_id=self._next_id,
            class_name=det.class_name,
            class_id=det.class_id,
            bbox=det.bbox,
            confidence=det.confidence,
            smooth_bbox=bbox_f,
        )
        self._next_id += 1

    def _update_track(self, tid: int, det: Detection):
        """Update an existing track with a new detection."""
        track = self._tracks[tid]
        track.bbox = det.bbox
        track.confidence = det.confidence
        track.frames_seen += 1
        track.frames_missing = 0

        # EMA smoothing on bbox
        a = self._smooth_factor
        old = track.smooth_bbox
        new = tuple(float(v) for v in det.bbox)
        track.smooth_bbox = tuple(
            a * n + (1 - a) * o for o, n in zip(old, new)
        )

    def reset(self):
        """Clear all tracks."""
        self._tracks.clear()
        self._next_id = 0

    @property
    def active_count(self) -> int:
        """Number of currently active tracks."""
        return len(self._tracks)

    @property
    def tracks(self) -> dict[int, TrackedObject]:
        """Current track dict (read-only view)."""
        return dict(self._tracks)

    def entered(self, prev: list[TrackedObject], curr: list[TrackedObject]) -> list[TrackedObject]:
        """Objects that appeared (new track IDs)."""
        prev_ids = {t.track_id for t in prev}
        return [t for t in curr if t.track_id not in prev_ids]

    def exited(self, prev: list[TrackedObject], curr: list[TrackedObject]) -> list[TrackedObject]:
        """Objects that disappeared (removed track IDs)."""
        curr_ids = {t.track_id for t in curr}
        return [t for t in prev if t.track_id not in curr_ids]
