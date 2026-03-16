#!/usr/bin/env python3
"""Data collection script — captures and saves frames for annotation."""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from src.video.capture import VideoCapture


def main():
    parser = argparse.ArgumentParser(description="Collect training images")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera index")
    parser.add_argument("--output", "-o", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--prefix", "-p", type=str, default="frame", help="Filename prefix")
    parser.add_argument("--interval", "-i", type=float, default=0.0,
                        help="Auto-capture interval in seconds (0 = manual only)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    cap = VideoCapture(source=args.camera, width=640, height=480)

    if not cap.open():
        print("Could not open camera")
        return

    print("Data collection mode")
    print("  SPACE = capture frame")
    print("  q     = quit")
    if args.interval > 0:
        print(f"  Auto-capture every {args.interval}s")

    count = 0
    last_auto = time.time()

    while True:
        frame_data = cap.read()
        if frame_data is None:
            break

        display = frame_data.image.copy()
        cv2.putText(display, f"Captured: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Data Collection", display)

        key = cv2.waitKey(1) & 0xFF
        save = False

        if key == ord(" "):
            save = True
        elif key == ord("q"):
            break

        if args.interval > 0 and (time.time() - last_auto) >= args.interval:
            save = True
            last_auto = time.time()

        if save:
            filename = f"{args.prefix}_{count:05d}.jpg"
            filepath = os.path.join(args.output, filename)
            cv2.imwrite(filepath, frame_data.image)
            count += 1
            print(f"Saved: {filepath}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected {count} images in {args.output}")


if __name__ == "__main__":
    main()
