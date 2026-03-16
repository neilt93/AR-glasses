"""AR Smart Assistant — main application loop.

Webcam → Scene Understanding → Object Tracking → Hand Detection
    → Assistant Brain → Widget HUD → Voice Output → Glasses Display

Usage:
    python -m src.assistant.app
    python -m src.assistant.app --glasses
    python -m src.assistant.app --model yolov8s --ocr
    python -m src.assistant.app --voice            # enable spoken feedback
    python -m src.assistant.app --hands            # enable hand tracking
    python -m src.assistant.app --no-tracking      # disable object tracking
"""

import argparse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np

from src.perception.scene import SceneUnderstanding
from src.display.output import GlassesDisplay
from src.hud.widgets import (
    WidgetRenderer, ObjectLabels, Crosshair, TrackedObjectLabels,
    HandSkeleton,
)
from src.assistant.brain import AssistantBrain


def main():
    parser = argparse.ArgumentParser(description="AR Smart Assistant")
    parser.add_argument("--camera", "-c", type=int, default=0)
    parser.add_argument("--model", "-m", type=str, default="yolov8n",
                        help="YOLO model: yolov8n (fast) / yolov8s / yolov8m (accurate)")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--glasses", "-g", action="store_true",
                        help="Output to AR glasses")
    parser.add_argument("--ocr", action="store_true",
                        help="Enable text detection")
    parser.add_argument("--voice", "-v", action="store_true",
                        help="Enable voice output (spoken notifications)")
    parser.add_argument("--hands", action="store_true",
                        help="Enable hand pose detection")
    parser.add_argument("--no-tracking", action="store_true",
                        help="Disable object tracking (use per-frame detection only)")
    parser.add_argument("--no-labels", action="store_true",
                        help="Hide object bounding boxes")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    # ── Init perception ───────────────────────────────────────────────────
    print(f"[assistant] Loading {args.model}...")
    scene = SceneUnderstanding(
        model_size=args.model,
        confidence=args.confidence,
        enable_ocr=args.ocr,
    )

    # ── Init brain + HUD ──────────────────────────────────────────────────
    enable_tracking = not args.no_tracking
    brain = AssistantBrain(
        notification_cooldown=10.0,
        enable_tracking=enable_tracking,
        enable_hands=args.hands,
        enable_voice=args.voice,
    )
    if args.no_labels:
        if enable_tracking:
            brain.toggle_widget(TrackedObjectLabels)
        else:
            brain.toggle_widget(ObjectLabels)

    # ── Init camera ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[assistant] ERROR: Could not open camera {args.camera}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    print(f"[assistant] Camera {args.camera} opened ({args.width}x{args.height})")

    # ── Init display ──────────────────────────────────────────────────────
    window_name = "AR Assistant"
    glasses = None
    if args.glasses:
        glasses = GlassesDisplay(window_name=window_name)
        glasses.open()
    else:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, args.width, args.height)

    # ── Welcome ───────────────────────────────────────────────────────────
    features = []
    if enable_tracking:
        features.append("tracking")
    if args.hands:
        features.append("hands")
    if args.voice:
        features.append("voice")
    if args.ocr:
        features.append("OCR")

    feature_str = f" [{', '.join(features)}]" if features else ""
    brain.notify(f"AR Assistant ready{feature_str}", "info", duration=3.0)
    brain.speak("AR Assistant ready")

    controls = "[q] quit  [h] labels  [c] crosshair  [i] info"
    if args.hands:
        controls += "  [k] hand skeleton"
    print(f"[assistant] Running — {controls}")

    # ── Main loop ─────────────────────────────────────────────────────────
    fps_counter = 0
    fps_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[assistant] Camera read failed")
                break

            # 1. Perceive
            perception = scene.perceive(frame)

            # 2. Think (includes tracking + hand detection)
            brain.think(perception, frame=frame)

            # 3. Render HUD
            context = brain.get_render_context(perception, fps=fps)
            display = brain.renderer.render(frame, context)

            # 4. Output
            if glasses is not None:
                glasses.show(display)
            else:
                cv2.imshow(window_name, display)

            # FPS
            fps_counter += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                fps_counter = 0
                fps_time = time.time()

            # ── Keyboard ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("h"):
                if enable_tracking:
                    brain.toggle_widget(TrackedObjectLabels)
                else:
                    brain.toggle_widget(ObjectLabels)
            elif key == ord("c"):
                brain.toggle_widget(Crosshair)
            elif key == ord("i"):
                from src.hud.widgets import InfoPanel
                brain.toggle_widget(InfoPanel)
            elif key == ord("k") and args.hands:
                brain.toggle_widget(HandSkeleton)

    finally:
        brain.shutdown()
        cap.release()
        if glasses is not None:
            glasses.close()
        cv2.destroyAllWindows()
        print("[assistant] Stopped")


if __name__ == "__main__":
    main()
