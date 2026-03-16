"""Test general scene understanding with the webcam.

Runs YOLO object detection on the webcam feed and displays detections
with labels. Tests the general perception pipeline independent of the
assembly-specific code.

Usage:
    python scripts/test_scene.py
    python scripts/test_scene.py --camera 1
    python scripts/test_scene.py --glasses     # output to AR glasses
    python scripts/test_scene.py --model yolov8s  # use small model (more accurate)
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from src.perception.scene import SceneUnderstanding, ScenePerception
from src.display.monitor import find_glasses_or_fallback
from src.display.output import GlassesDisplay


# Colors for different object classes (cycle through)
PALETTE = [
    (0, 220, 0), (220, 180, 0), (0, 140, 255), (200, 0, 200),
    (255, 100, 0), (0, 255, 255), (255, 0, 100), (100, 255, 100),
    (255, 200, 0), (0, 100, 255), (180, 0, 180), (0, 200, 150),
]

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


def color_for_class(class_name: str) -> tuple:
    """Deterministic color for a class name."""
    return PALETTE[hash(class_name) % len(PALETTE)]


def draw_scene(frame: np.ndarray, perception: ScenePerception) -> np.ndarray:
    """Draw scene understanding overlay on a frame."""
    display = frame.copy()
    h, w = display.shape[:2]

    # Draw object detections
    for det in perception.objects:
        x1, y1, x2, y2 = det.bbox
        color = color_for_class(det.class_name)

        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        label = f"{det.class_name} {det.confidence:.0%}"
        label_size, _ = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.rectangle(display, (x1, y1 - label_size[1] - 8),
                      (x1 + label_size[0] + 4, y1), color, -1)
        cv2.putText(display, label, (x1 + 2, y1 - 4),
                    FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw text regions
    for tr in perception.text_regions:
        x1, y1, x2, y2 = tr.bbox
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.putText(display, tr.text, (x1, y1 - 4),
                    FONT, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    # Top bar: scene summary
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

    summary = perception.summary()
    cv2.putText(display, summary, (10, 32),
                FONT_BOLD, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Scene tags on the right
    if perception.scene_tags:
        tags_text = " | ".join(perception.scene_tags)
        tag_size, _ = cv2.getTextSize(tags_text, FONT, 0.45, 1)
        cv2.putText(display, tags_text, (w - tag_size[0] - 10, 32),
                    FONT, 0.45, (0, 200, 200), 1, cv2.LINE_AA)

    # Bottom bar: stats
    overlay = display.copy()
    cv2.rectangle(overlay, (0, h - 35), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

    stats = (f"Objects: {len(perception.objects)} | "
             f"Classes: {len(perception.object_names)} | "
             f"{perception.inference_ms:.0f}ms")
    cv2.putText(display, stats, (10, h - 12),
                FONT, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.putText(display, "q: quit", (w - 80, h - 12),
                FONT, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

    return display


def main():
    parser = argparse.ArgumentParser(description="Test general scene understanding")
    parser.add_argument("--camera", "-c", type=int, default=0)
    parser.add_argument("--model", "-m", type=str, default="yolov8n",
                        help="YOLO model size: yolov8n (fast), yolov8s, yolov8m (accurate)")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--glasses", "-g", action="store_true")
    parser.add_argument("--ocr", action="store_true", help="Enable text detection (needs pytesseract)")
    args = parser.parse_args()

    # Init perception
    print(f"Loading {args.model}...")
    scene = SceneUnderstanding(
        model_size=args.model,
        confidence=args.confidence,
        enable_ocr=args.ocr,
    )

    # Init camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"Camera {args.camera} opened")

    # Init display
    glasses_display = None
    window_name = "Scene Understanding"
    if args.glasses:
        glasses_display = GlassesDisplay(window_name=window_name)
        glasses_display.open()
    else:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

    print("Running — press 'q' to quit\n")

    fps_counter = 0
    fps_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perceive
            perception = scene.perceive(frame)

            # Draw overlay
            display = draw_scene(frame, perception)

            # Show
            if glasses_display is not None:
                glasses_display.show(display)
            else:
                cv2.imshow(window_name, display)

            # FPS tracking
            fps_counter += 1
            elapsed = time.time() - fps_time
            if elapsed >= 2.0:
                fps = fps_counter / elapsed
                fps_counter = 0
                fps_time = time.time()
                obj_names = ", ".join(perception.object_names[:5]) or "none"
                print(f"[{fps:.1f} FPS] {len(perception.objects)} objects: {obj_names}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        if glasses_display is not None:
            glasses_display.close()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
