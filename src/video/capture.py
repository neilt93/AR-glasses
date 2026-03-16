"""Video capture module — reads frames from a webcam or video file."""

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class FrameData:
    """A single captured frame with metadata."""
    image: np.ndarray
    timestamp: float
    frame_number: int
    width: int
    height: int


class VideoCapture:
    """Captures frames from a camera or video file."""

    def __init__(
        self,
        source: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.source = source
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0

    def open(self) -> bool:
        """Open the video source. Returns True on success."""
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self._frame_count = 0
        return True

    def read(self) -> Optional[FrameData]:
        """Read the next frame. Returns None if capture failed."""
        if self._cap is None or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        # Resize to target resolution if needed
        h, w = frame.shape[:2]
        if w != self.target_width or h != self.target_height:
            frame = cv2.resize(frame, (self.target_width, self.target_height))
        self._frame_count += 1
        return FrameData(
            image=frame,
            timestamp=time.time(),
            frame_number=self._frame_count,
            width=self.target_width,
            height=self.target_height,
        )

    def release(self):
        """Release the video source."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.release()
