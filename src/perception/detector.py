"""Object detection module — detects task-relevant objects using YOLO."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Detection:
    """A single detected object."""
    class_name: str
    class_id: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float


# Default classes for the servo-bracket assembly task
ASSEMBLY_CLASSES = [
    "base_plate",
    "bracket",
    "screw",
    "screwdriver",
    "servo",
    "hand",
]


class ObjectDetector:
    """Detects task-relevant objects in a frame.

    Uses Ultralytics YOLO when a model path is provided.
    Falls back to a mock detector for development without a trained model.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.4,
        classes: Optional[list[str]] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.classes = classes or ASSEMBLY_CLASSES
        self._model = None
        self._use_mock = model_path is None

        if model_path is not None:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load a YOLO model from disk."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            self._use_mock = False
        except ImportError:
            print("[detector] ultralytics not installed — using mock detector")
            self._use_mock = True
        except Exception as e:
            print(f"[detector] failed to load model: {e} — using mock detector")
            self._use_mock = True

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a single frame."""
        if self._use_mock:
            return self._mock_detect(frame)
        return self._yolo_detect(frame)

    def _yolo_detect(self, frame: np.ndarray) -> list[Detection]:
        """Run real YOLO inference."""
        results = self._model(frame, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                cls_name = result.names.get(cls_id, f"class_{cls_id}")
                detections.append(Detection(
                    class_name=cls_name,
                    class_id=cls_id,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                ))
        return detections

    def _mock_detect(self, frame: np.ndarray) -> list[Detection]:
        """Return mock detections for development/testing.

        Generates plausible fake detections so the rest of the pipeline
        can be developed and tested without a trained model.
        """
        h, w = frame.shape[:2]
        mock_objects = [
            Detection("base_plate", 0, (int(w*0.2), int(h*0.4), int(w*0.8), int(h*0.9)), 0.92),
            Detection("bracket", 1, (int(w*0.1), int(h*0.1), int(w*0.3), int(h*0.35)), 0.85),
            Detection("screwdriver", 3, (int(w*0.7), int(h*0.05), int(w*0.95), int(h*0.15)), 0.78),
            Detection("screw", 2, (int(w*0.45), int(h*0.15), int(w*0.55), int(h*0.25)), 0.65),
        ]
        return mock_objects
