#!/usr/bin/env python3
"""Simple annotation helper — view images and record bounding box labels.

This is a minimal tool for bootstrapping annotations. For serious annotation
work, use a dedicated tool like Label Studio, CVAT, or Roboflow.

Usage:
    python scripts/annotate_data.py --input data/raw --output data/labels
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

CLASSES = ["base_plate", "bracket", "screw", "screwdriver", "servo", "hand"]


def main():
    parser = argparse.ArgumentParser(description="Simple annotation viewer")
    parser.add_argument("--input", "-i", type=str, default="data/raw")
    parser.add_argument("--output", "-o", type=str, default="data/labels")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    images = sorted([
        f for f in os.listdir(args.input)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not images:
        print(f"No images found in {args.input}")
        return

    print(f"Found {len(images)} images")
    print("Controls:")
    print("  LEFT/RIGHT = navigate images")
    print("  q = quit")
    print()
    print("For full annotation, use Label Studio or Roboflow.")
    print("This script is just for quick visual review.")

    idx = 0
    while True:
        img_path = os.path.join(args.input, images[idx])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read: {img_path}")
            idx = (idx + 1) % len(images)
            continue

        display = img.copy()
        cv2.putText(display, f"[{idx+1}/{len(images)}] {images[idx]}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.imshow("Annotation Viewer", display)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        elif key == 83 or key == ord("d"):  # RIGHT arrow or 'd'
            idx = (idx + 1) % len(images)
        elif key == 81 or key == ord("a"):  # LEFT arrow or 'a'
            idx = (idx - 1) % len(images)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
