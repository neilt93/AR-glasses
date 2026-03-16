"""Core data structures for run recording and trajectory storage."""

import json
import os
import uuid
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class FrameRecord:
    """A single frame's complete pipeline state."""
    t: int                          # frame index within the run
    timestamp: float                # wall-clock time
    scene_state: dict               # structured scene state
    step_estimate: int              # current step (AssemblyStep int value)
    step_confidence: float          # FSM transition confidence
    detections: list[dict]          # [{class, bbox, conf}, ...]
    hand_positions: dict            # placeholder for future hand tracking
    anomaly_score: float = 0.0      # from Layer 3
    duration_z_score: float = 0.0   # from Layer 1
    failure_risk: float = 0.0       # from Layer 4
    inference_ms: float = 0.0


@dataclass
class RunRecord:
    """A complete run trajectory."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    user_id: str = "default"
    task_id: str = "servo_bracket_assembly"
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    outcome: str = "in_progress"    # "success" | "failure" | "abandoned"
    failure_step: Optional[int] = None
    failure_frame: Optional[int] = None
    total_frames: int = 0
    step_durations: dict = field(default_factory=dict)  # step_num -> frame count
    frames: list[FrameRecord] = field(default_factory=list)

    def add_frame(self, frame: FrameRecord):
        self.frames.append(frame)
        self.total_frames = len(self.frames)

    def finish(self, outcome: str, failure_step: Optional[int] = None):
        self.end_time = time.time()
        self.outcome = outcome
        self.failure_step = failure_step
        if failure_step is not None and self.frames:
            self.failure_frame = len(self.frames) - 1
        self._compute_step_durations()

    def _compute_step_durations(self):
        """Count frames spent in each step."""
        durations: dict[int, int] = {}
        for frame in self.frames:
            step = frame.step_estimate
            durations[step] = durations.get(step, 0) + 1
        self.step_durations = {str(k): v for k, v in durations.items()}

    def to_dict(self) -> dict:
        """Serialize to a dict (without frame images)."""
        return {
            "run_id": self.run_id,
            "user_id": self.user_id,
            "task_id": self.task_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "outcome": self.outcome,
            "failure_step": self.failure_step,
            "failure_frame": self.failure_frame,
            "total_frames": self.total_frames,
            "step_durations": self.step_durations,
            "frames": [asdict(f) for f in self.frames],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunRecord":
        frames = [FrameRecord(**f) for f in data.pop("frames", [])]
        record = cls(**{k: v for k, v in data.items() if k != "frames"})
        record.frames = frames
        return record


class RunStore:
    """Persists run records to a JSONL file."""

    def __init__(self, store_dir: str = "data/runs"):
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)
        self._index_path = os.path.join(store_dir, "index.jsonl")

    def save(self, run: RunRecord):
        """Save a complete run record."""
        # Save full run to its own file
        run_path = os.path.join(self.store_dir, f"{run.run_id}.json")
        with open(run_path, "w") as f:
            json.dump(run.to_dict(), f)

        # Append summary to index (without frames, for fast loading)
        summary = {
            "run_id": run.run_id,
            "user_id": run.user_id,
            "task_id": run.task_id,
            "start_time": run.start_time,
            "end_time": run.end_time,
            "outcome": run.outcome,
            "failure_step": run.failure_step,
            "total_frames": run.total_frames,
            "step_durations": run.step_durations,
        }
        with open(self._index_path, "a") as f:
            f.write(json.dumps(summary) + "\n")

    def load(self, run_id: str) -> Optional[RunRecord]:
        """Load a full run record by ID."""
        run_path = os.path.join(self.store_dir, f"{run_id}.json")
        if not os.path.exists(run_path):
            return None
        with open(run_path, "r") as f:
            data = json.load(f)
        return RunRecord.from_dict(data)

    def load_index(self) -> list[dict]:
        """Load all run summaries (without frames)."""
        if not os.path.exists(self._index_path):
            return []
        summaries = []
        with open(self._index_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    summaries.append(json.loads(line))
        return summaries

    def load_all(self, outcome: Optional[str] = None) -> list[RunRecord]:
        """Load all complete runs, optionally filtered by outcome."""
        index = self.load_index()
        runs = []
        for summary in index:
            if outcome and summary.get("outcome") != outcome:
                continue
            run = self.load(summary["run_id"])
            if run is not None:
                runs.append(run)
        return runs

    def count(self, outcome: Optional[str] = None) -> int:
        index = self.load_index()
        if outcome is None:
            return len(index)
        return sum(1 for s in index if s.get("outcome") == outcome)
