"""Spatial feature extraction from raw detections."""

from src.perception.detector import Detection


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """Return the center point of a bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_area(bbox: tuple[int, int, int, int]) -> int:
    """Return the area of a bounding box."""
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Compute intersection-over-union of two bounding boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = bbox_area(a)
    area_b = bbox_area(b)
    union = area_a + area_b - inter
    if union == 0:
        return 0.0
    return inter / union


def bbox_overlap_ratio(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
    """What fraction of `inner` overlaps with `outer`."""
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    inner_area = bbox_area(inner)
    if inner_area == 0:
        return 0.0
    return inter / inner_area


def distance_between(a: Detection, b: Detection) -> float:
    """Euclidean distance between the centers of two detections."""
    ca = bbox_center(a.bbox)
    cb = bbox_center(b.bbox)
    return ((ca[0] - cb[0])**2 + (ca[1] - cb[1])**2) ** 0.5


def is_near(a: Detection, b: Detection, threshold: float = 80.0) -> bool:
    """Check if two detections are near each other."""
    return distance_between(a, b) < threshold


def is_overlapping(a: Detection, b: Detection, iou_threshold: float = 0.15) -> bool:
    """Check if two detections overlap significantly."""
    return bbox_iou(a.bbox, b.bbox) > iou_threshold
