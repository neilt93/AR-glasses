"""General-purpose scene understanding.

Combines object detection (YOLO/COCO), OCR, and scene classification
into a single perception output per frame. This replaces the assembly-
specific detector as the primary perception layer for the AR assistant.
"""

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from src.perception.detector import Detection


# COCO class names (80 classes) — what YOLOv8 pretrained model detects
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


@dataclass
class TextRegion:
    """A detected text region in the frame."""
    text: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float


@dataclass
class ScenePerception:
    """Complete perception output for a single frame."""
    # Objects detected in scene
    objects: list[Detection] = field(default_factory=list)

    # Text found in scene (OCR)
    text_regions: list[TextRegion] = field(default_factory=list)

    # People count
    people_count: int = 0

    # Dominant objects (most prominent by area)
    dominant_objects: list[str] = field(default_factory=list)

    # Scene tags (high-level descriptors derived from objects)
    scene_tags: list[str] = field(default_factory=list)

    # Inference time
    inference_ms: float = 0.0

    @property
    def object_names(self) -> list[str]:
        """Unique object class names detected."""
        return list(dict.fromkeys(d.class_name for d in self.objects))

    @property
    def has_text(self) -> bool:
        return len(self.text_regions) > 0

    @property
    def all_text(self) -> str:
        """All detected text joined."""
        return " ".join(r.text for r in self.text_regions)

    def objects_of_class(self, class_name: str) -> list[Detection]:
        return [d for d in self.objects if d.class_name == class_name]

    def summary(self) -> str:
        """One-line scene summary for the HUD."""
        parts = []
        if self.people_count:
            parts.append(f"{self.people_count} {'person' if self.people_count == 1 else 'people'}")
        non_person = [n for n in self.dominant_objects if n != "person"][:3]
        if non_person:
            parts.append(", ".join(non_person))
        if self.text_regions:
            parts.append(f"text: \"{self.text_regions[0].text[:30]}\"")
        return " | ".join(parts) if parts else "empty scene"


class SceneUnderstanding:
    """General-purpose scene perception using YOLO + OCR.

    Uses YOLOv8 pretrained on COCO for object detection.
    Uses OpenCV's EAST or Tesseract for OCR (optional).
    Falls back gracefully when models are unavailable.
    """

    def __init__(
        self,
        model_size: str = "yolov8n",  # nano for speed: n, s, m, l, x
        confidence: float = 0.35,
        enable_ocr: bool = False,
        device: str = "",  # "" = auto, "cpu", "cuda:0"
    ):
        self._confidence = confidence
        self._model_size = model_size
        self._enable_ocr = enable_ocr
        self._device = device
        self._yolo = None
        self._ocr_ready = False
        self._load_models()

    def _load_models(self):
        """Load detection models. Falls back gracefully."""
        # YOLO
        try:
            from ultralytics import YOLO
            self._yolo = YOLO(f"{self._model_size}.pt")
            if self._device:
                self._yolo.to(self._device)
            print(f"[scene] Loaded {self._model_size} ({len(COCO_CLASSES)} classes)")
        except ImportError:
            print("[scene] ultralytics not installed — object detection unavailable")
            print("[scene] Install with: pip install ultralytics")
        except Exception as e:
            print(f"[scene] Failed to load YOLO: {e}")

        # OCR
        if self._enable_ocr:
            try:
                self._east_net = cv2.dnn.readNet(
                    cv2.samples.findFile("frozen_east_text_detection.pb", False)
                )
                self._ocr_ready = True
                print("[scene] EAST text detector loaded")
            except Exception:
                # Try pytesseract as fallback
                try:
                    import pytesseract
                    self._ocr_ready = True
                    print("[scene] Using pytesseract for OCR")
                except ImportError:
                    print("[scene] OCR unavailable (install pytesseract for text detection)")

    def perceive(self, frame: np.ndarray) -> ScenePerception:
        """Run full scene perception on a frame."""
        import time
        t0 = time.time()

        perception = ScenePerception()

        # Object detection
        if self._yolo is not None:
            perception.objects = self._detect_objects(frame)
        else:
            perception.objects = self._mock_detect(frame)

        # People count
        perception.people_count = sum(
            1 for d in perception.objects if d.class_name == "person"
        )

        # Dominant objects (by bbox area)
        perception.dominant_objects = self._rank_by_area(perception.objects)

        # Scene tags
        perception.scene_tags = self._infer_scene_tags(perception)

        # OCR
        if self._enable_ocr and self._ocr_ready:
            perception.text_regions = self._detect_text(frame)

        perception.inference_ms = (time.time() - t0) * 1000.0
        return perception

    def _detect_objects(self, frame: np.ndarray) -> list[Detection]:
        """Run YOLO detection."""
        results = self._yolo(frame, conf=self._confidence, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
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
        """Mock detection for development without YOLO installed."""
        h, w = frame.shape[:2]
        return [
            Detection("laptop", 63, (int(w*0.3), int(h*0.3), int(w*0.7), int(h*0.7)), 0.88),
            Detection("cup", 41, (int(w*0.05), int(h*0.4), int(w*0.15), int(h*0.65)), 0.72),
            Detection("keyboard", 66, (int(w*0.25), int(h*0.7), int(w*0.75), int(h*0.9)), 0.81),
        ]

    def _rank_by_area(self, detections: list[Detection], top_n: int = 5) -> list[str]:
        """Rank detected objects by bounding box area, return unique names."""
        def bbox_area(d: Detection) -> int:
            x1, y1, x2, y2 = d.bbox
            return max(0, x2 - x1) * max(0, y2 - y1)

        sorted_dets = sorted(detections, key=bbox_area, reverse=True)
        seen = set()
        result = []
        for d in sorted_dets:
            if d.class_name not in seen:
                seen.add(d.class_name)
                result.append(d.class_name)
            if len(result) >= top_n:
                break
        return result

    def _infer_scene_tags(self, perception: ScenePerception) -> list[str]:
        """Derive high-level scene descriptors from detected objects."""
        names = set(d.class_name for d in perception.objects)
        tags = []

        # Workspace indicators
        workspace = {"laptop", "keyboard", "mouse", "monitor", "cell phone", "book"}
        if names & workspace:
            tags.append("workspace")

        # Kitchen/dining
        kitchen = {"cup", "bowl", "fork", "knife", "spoon", "bottle", "wine glass",
                   "microwave", "oven", "toaster", "sink", "refrigerator"}
        if names & kitchen:
            tags.append("kitchen/dining")

        # Outdoors
        outdoor = {"car", "truck", "bus", "bicycle", "motorcycle", "traffic light",
                   "stop sign", "fire hydrant", "bench"}
        if names & outdoor:
            tags.append("outdoors")

        # Social
        if perception.people_count > 1:
            tags.append("social")
        elif perception.people_count == 1:
            tags.append("person present")

        # Tools/maker
        tools = {"scissors", "knife"}
        if names & tools:
            tags.append("tools")

        return tags

    def _detect_text(self, frame: np.ndarray) -> list[TextRegion]:
        """Detect text in the frame using available OCR."""
        try:
            import pytesseract
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            data = pytesseract.image_to_data(
                gray, output_type=pytesseract.Output.DICT, config="--psm 11"
            )
            regions = []
            for i, text in enumerate(data["text"]):
                text = text.strip()
                conf = int(data["conf"][i])
                if text and conf > 40:
                    x = data["left"][i]
                    y = data["top"][i]
                    w = data["width"][i]
                    h = data["height"][i]
                    regions.append(TextRegion(
                        text=text,
                        bbox=(x, y, x + w, y + h),
                        confidence=conf / 100.0,
                    ))
            return regions
        except Exception:
            return []

    @property
    def is_ready(self) -> bool:
        """True if at least object detection is available."""
        return self._yolo is not None
