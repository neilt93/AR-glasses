"""Test script — detect monitors and render a test pattern to the AR glasses.

Usage:
    python scripts/test_glasses.py
    python scripts/test_glasses.py --monitor 1    # force a specific monitor index
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from src.display.monitor import detect_monitors, find_glasses, print_monitor_info, MonitorInfo
from src.display.output import GlassesDisplay


def draw_test_pattern(width: int, height: int, frame_num: int) -> np.ndarray:
    """Draw a test pattern with calibration info."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Border rectangle (verify full frame is visible)
    cv2.rectangle(frame, (2, 2), (width - 3, height - 3), (0, 255, 0), 2)

    # Crosshair at center
    cx, cy = width // 2, height // 2
    cv2.line(frame, (cx - 40, cy), (cx + 40, cy), (255, 255, 255), 1)
    cv2.line(frame, (cx, cy - 40), (cx, cy + 40), (255, 255, 255), 1)

    # Corner markers
    for x, y in [(20, 20), (width - 20, 20), (20, height - 20), (width - 20, height - 20)]:
        cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)

    # Title
    cv2.putText(frame, "AR Glasses Test Pattern", (cx - 200, 60),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{width}x{height}", (cx - 60, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 1, cv2.LINE_AA)

    # Color bars
    colors = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0),
        (0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 255, 255),
    ]
    bar_w = width // len(colors)
    bar_y = height // 2 - 30
    for i, color in enumerate(colors):
        x1 = i * bar_w
        cv2.rectangle(frame, (x1, bar_y), (x1 + bar_w, bar_y + 60), color, -1)

    # Animated element (proves frames are updating)
    angle = (frame_num * 3) % 360
    pulse_x = int(cx + 150 * np.cos(np.radians(angle)))
    pulse_y = int(cy + 80 + 50 * np.sin(np.radians(angle)))
    cv2.circle(frame, (pulse_x, pulse_y), 15, (0, 200, 255), -1)

    # Instructions
    cv2.putText(frame, "Press 'q' to quit", (20, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Frame: {frame_num}", (width - 200, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Test AR glasses display output")
    parser.add_argument("--monitor", "-m", type=int, default=None,
                        help="Force output to a specific monitor index")
    args = parser.parse_args()

    # Detect monitors
    monitors = detect_monitors()
    print_monitor_info(monitors)
    print()

    if not monitors:
        print("No monitors detected. Check your display connections.")
        return

    # Select target monitor
    target = None
    if args.monitor is not None:
        for mon in monitors:
            if mon.index == args.monitor:
                target = mon
                break
        if target is None:
            print(f"Monitor index {args.monitor} not found.")
            return
        print(f"Forcing output to monitor [{target.index}] {target.name}")
    else:
        target = find_glasses(monitors)
        if target:
            print(f"AR glasses detected: [{target.index}] {target.name}")
        else:
            print("AR glasses not detected. Showing on primary monitor.")
            target = next((m for m in monitors if m.is_primary), monitors[0])

    # Open display
    display = GlassesDisplay(
        window_name="AR Glasses Test",
        target_monitor=target,
    )
    display.open()

    print(f"\nRendering test pattern at {target.width}x{target.height}...")
    print("Press 'q' to quit.\n")

    frame_num = 0
    while True:
        pattern = draw_test_pattern(target.width, target.height, frame_num)
        display.show(pattern)
        frame_num += 1

        key = cv2.waitKey(16) & 0xFF  # ~60fps
        if key == ord("q"):
            break

    display.close()
    print("Done.")


if __name__ == "__main__":
    main()
