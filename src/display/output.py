"""Display output — renders content fullscreen on the AR glasses.

Handles window creation, positioning on the correct monitor,
and fullscreen management using OpenCV's windowing system.
"""

from typing import Optional

import cv2
import numpy as np

from src.display.monitor import MonitorInfo, find_glasses_or_fallback, print_monitor_info


class GlassesDisplay:
    """Manages rendering to the AR glasses as a fullscreen window.

    The RayNeo Air 4 Pro appears as a standard 1920x1080 external monitor.
    This class creates an OpenCV window positioned on that monitor and
    handles fullscreen toggling.
    """

    def __init__(
        self,
        window_name: str = "AR Glasses",
        target_monitor: Optional[MonitorInfo] = None,
        auto_detect: bool = True,
        debug_mode: bool = False,
    ):
        self.window_name = window_name
        self.debug_mode = debug_mode
        self._monitor: Optional[MonitorInfo] = None
        self._window_created = False

        if target_monitor is not None:
            self._monitor = target_monitor
        elif auto_detect:
            self._monitor = find_glasses_or_fallback()

    @property
    def monitor(self) -> Optional[MonitorInfo]:
        return self._monitor

    @property
    def resolution(self) -> tuple[int, int]:
        if self._monitor:
            return (self._monitor.width, self._monitor.height)
        return (1920, 1080)

    def open(self):
        """Create and position the display window on the glasses."""
        if self._monitor is None:
            self._monitor = find_glasses_or_fallback()

        if self.debug_mode:
            print_monitor_info()

        # Create the window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Move it to the glasses monitor
        x, y = self._monitor.x, self._monitor.y
        w, h = self._monitor.width, self._monitor.height
        cv2.moveWindow(self.window_name, x, y)
        cv2.resizeWindow(self.window_name, w, h)

        # Go fullscreen
        cv2.setWindowProperty(
            self.window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )

        self._window_created = True
        print(f"[display] Opened on monitor [{self._monitor.index}] "
              f"{self._monitor.name} ({w}x{h}) at ({x}, {y})")

    def show(self, frame: np.ndarray):
        """Display a frame on the glasses.

        The frame is resized to match the glasses resolution if needed.
        """
        if not self._window_created:
            self.open()

        target_w, target_h = self.resolution
        h, w = frame.shape[:2]
        if w != target_w or h != target_h:
            frame = cv2.resize(frame, (target_w, target_h))

        cv2.imshow(self.window_name, frame)

    def close(self):
        """Close the display window."""
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
