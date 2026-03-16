#!/usr/bin/env python3
"""Replay a session log file and print step-by-step summary."""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def replay(log_path: str, show_all: bool = False):
    """Read a JSONL log file and print a summary."""
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    with open(log_path, "r") as f:
        lines = f.readlines()

    print(f"Replaying {len(lines)} frames from: {log_path}")
    print("-" * 60)

    prev_step = None
    for line in lines:
        record = json.loads(line)
        step = record.get("step_label", "")
        changed = record.get("changed", False)

        if changed or show_all:
            t = record.get("t", 0)
            frame = record.get("frame", 0)
            conf = record.get("confidence", 0)
            instr = record.get("instruction", "")
            warnings = record.get("warnings", [])
            ms = record.get("inference_ms", 0)

            print(f"[{t:7.2f}s] Frame {frame:5d} | {step}")
            print(f"           Instruction: {instr}")
            if warnings:
                print(f"           Warnings: {', '.join(warnings)}")
            print(f"           Confidence: {conf:.0%}  Latency: {ms:.1f}ms")
            print()

        prev_step = step

    print("-" * 60)
    print("Replay complete.")


def main():
    parser = argparse.ArgumentParser(description="Replay a session log")
    parser.add_argument("--log", "-l", required=True, help="Path to .jsonl log file")
    parser.add_argument("--all", "-a", action="store_true", help="Show all frames, not just transitions")
    args = parser.parse_args()
    replay(args.log, show_all=args.all)


if __name__ == "__main__":
    main()
