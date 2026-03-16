#!/usr/bin/env python3
"""Launch the assembly copilot in demo mode."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.orchestrator import Pipeline, PipelineConfig


def main():
    config = PipelineConfig(
        fullscreen=True,
        persistence_frames=8,
        stall_threshold=200,
        detection_confidence=0.5,
        task_name="Servo Bracket Assembly",
    )
    print("=== AR Assembly Copilot — Demo Mode ===")
    print("Press 'q' to quit, 'r' to reset task")
    print()

    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
