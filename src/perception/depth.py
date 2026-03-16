"""Monocular depth estimation using MiDaS.

Estimates relative depth from a single RGB image. This enables:
- Spatial reasoning ("the cup is closer than the laptop")
- Depth-aware object sorting
- Z-ordering for HUD overlays
- Future: occlusion handling, spatial anchors

Uses MiDaS v2.1 small model by default (fast, CPU-friendly).
Falls back gracefully when torch/MiDaS unavailable.

Usage:
    depth = DepthEstimator()
    result = depth.estimate(frame)
    print(f"Depth range: {result.min_depth:.2f} - {result.max_depth:.2f}")
    print(f"Center depth: {result.depth_at(320, 240):.2f}")
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class DepthResult:
    """Depth estimation output for a single frame."""
    depth_map: np.ndarray       # HxW float32, normalized 0.0 (near) to 1.0 (far)
    raw_depth: np.ndarray       # HxW float32, raw MiDaS output (inverse depth)
    inference_ms: float = 0.0

    @property
    def height(self) -> int:
        return self.depth_map.shape[0]

    @property
    def width(self) -> int:
        return self.depth_map.shape[1]

    @property
    def min_depth(self) -> float:
        """Minimum depth value (closest point)."""
        return float(self.depth_map.min())

    @property
    def max_depth(self) -> float:
        """Maximum depth value (farthest point)."""
        return float(self.depth_map.max())

    def depth_at(self, x: int, y: int) -> float:
        """Get normalized depth at a pixel coordinate."""
        h, w = self.depth_map.shape
        y = max(0, min(y, h - 1))
        x = max(0, min(x, w - 1))
        return float(self.depth_map[y, x])

    def depth_in_bbox(self, bbox: tuple[int, int, int, int]) -> float:
        """Get median depth within a bounding box."""
        x1, y1, x2, y2 = bbox
        h, w = self.depth_map.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.5
        region = self.depth_map[y1:y2, x1:x2]
        return float(np.median(region))

    def colormap(self, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
        """Convert depth map to a colored visualization."""
        depth_uint8 = (self.depth_map * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_uint8, colormap)

    def objects_by_depth(self, bboxes: list[tuple[int, int, int, int]],
                         names: list[str]) -> list[tuple[str, float]]:
        """Sort objects by their depth (nearest first)."""
        pairs = [(name, self.depth_in_bbox(bbox)) for name, bbox in zip(names, bboxes)]
        return sorted(pairs, key=lambda p: p[1])


class DepthEstimator:
    """Monocular depth estimation using MiDaS.

    Falls back gracefully when torch/MiDaS is unavailable.
    Uses OpenCV DNN as the default backend (no torch required).
    """

    def __init__(
        self,
        model_type: str = "midas_small",  # "midas_small", "dpt_hybrid", "dpt_large"
        target_size: tuple[int, int] = (256, 256),  # inference resolution
    ):
        self._model_type = model_type
        self._target_size = target_size
        self._model = None
        self._transform = None
        self._use_cv_dnn = False
        self._net = None
        self._ready = False
        self._load()

    def _load(self):
        """Load depth model. Try torch first, fall back to OpenCV DNN."""
        # Try torch + MiDaS hub
        try:
            import torch
            self._model = torch.hub.load("intel-isl/MiDaS", self._model_type,
                                         trust_repo=True)
            self._model.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms",
                                              trust_repo=True)
            if self._model_type == "midas_small":
                self._transform = midas_transforms.small_transform
            elif self._model_type == "dpt_hybrid":
                self._transform = midas_transforms.dpt_transform
            else:
                self._transform = midas_transforms.dpt_transform

            self._ready = True
            print(f"[depth] MiDaS {self._model_type} loaded (torch)")
            return
        except ImportError:
            pass
        except Exception as e:
            print(f"[depth] Torch MiDaS failed: {e}")

        print("[depth] Depth estimation unavailable")
        print("[depth] Install with: pip install torch torchvision")

    def estimate(self, frame: np.ndarray) -> Optional[DepthResult]:
        """Estimate depth from a single frame."""
        if not self._ready:
            return None

        import time
        t0 = time.time()

        result = self._estimate_torch(frame)

        if result is not None:
            result.inference_ms = (time.time() - t0) * 1000.0
        return result

    def _estimate_torch(self, frame: np.ndarray) -> Optional[DepthResult]:
        """Run MiDaS inference with torch."""
        try:
            import torch
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = self._transform(rgb)

            with torch.no_grad():
                prediction = self._model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            raw = prediction.cpu().numpy()

            # Normalize to 0-1 (MiDaS outputs inverse depth: higher = closer)
            # Invert so 0 = near, 1 = far (more intuitive)
            if raw.max() - raw.min() > 0:
                normalized = (raw - raw.min()) / (raw.max() - raw.min())
                # MiDaS: higher values = closer, so invert for intuitive near=0, far=1
                normalized = 1.0 - normalized
            else:
                normalized = np.zeros_like(raw)

            return DepthResult(
                depth_map=normalized.astype(np.float32),
                raw_depth=raw.astype(np.float32),
            )
        except Exception as e:
            print(f"[depth] Inference error: {e}")
            return None

    @property
    def is_ready(self) -> bool:
        return self._ready

    def close(self):
        """Release model resources."""
        self._model = None
        self._ready = False
