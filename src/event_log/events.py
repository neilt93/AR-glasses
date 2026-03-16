"""Event logger — logs per-frame pipeline data to JSONL files."""

import json
import os
import time
from datetime import datetime
from typing import Optional

from src.perception.detector import Detection
from src.state.scene_state import SceneState
from src.task.step_estimator import StepEstimate
from src.instructions.engine import Guidance


class EventLogger:
    """Logs pipeline events to a JSONL file for debugging and replay."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._filepath = os.path.join(log_dir, f"session_{timestamp}.jsonl")
        self._file = open(self._filepath, "w")
        self._start_time = time.time()

    def log(
        self,
        frame_number: int,
        detections: list[Detection],
        state: SceneState,
        estimate: StepEstimate,
        guidance: Guidance,
        inference_ms: float = 0.0,
    ):
        """Log a single frame's pipeline output."""
        record = {
            "t": time.time() - self._start_time,
            "frame": frame_number,
            "detections": [
                {
                    "class": d.class_name,
                    "bbox": list(d.bbox),
                    "conf": round(d.confidence, 3),
                }
                for d in detections
            ],
            "state": state.to_dict(),
            "step": estimate.step.name,
            "step_num": estimate.step_number,
            "step_label": estimate.step_label,
            "confidence": round(estimate.confidence, 3),
            "changed": estimate.changed,
            "progress_pct": round(estimate.progress_pct, 1),
            "instruction": guidance.instruction,
            "warnings": guidance.warnings,
            "inference_ms": round(inference_ms, 1),
        }
        self._file.write(json.dumps(record) + "\n")

    def flush(self):
        self._file.flush()

    def close(self):
        self._file.close()

    @property
    def filepath(self) -> str:
        return self._filepath

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
