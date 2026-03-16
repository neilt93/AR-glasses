"""Layer 3: Scene state anomaly detection.

For each step, maintains a distribution of 'normal' scene states from
successful runs. At inference, computes how far the current state is
from the learned distribution to detect anomalies.
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


# Feature keys extracted from scene state dicts
BINARY_FEATURES = [
    "base_plate_visible",
    "bracket_visible",
    "screwdriver_visible",
    "servo_visible",
    "hand_visible",
    "bracket_on_plate",
    "screw_near_plate",
    "screwdriver_near_screw",
    "screwdriver_engaged",
]

CONTINUOUS_FEATURES = [
    "screws_visible",
]

ALL_FEATURES = BINARY_FEATURES + CONTINUOUS_FEATURES


def state_to_vector(state: dict) -> np.ndarray:
    """Convert a scene state dict to a fixed-length feature vector."""
    vec = []
    for key in ALL_FEATURES:
        val = state.get(key, 0)
        vec.append(float(val) if isinstance(val, (int, float, bool)) else 0.0)
    return np.array(vec, dtype=np.float64)


class AnomalyDetector:
    """Detects anomalous scene states per step using distance-based methods.

    Supports three modes:
    1. Gaussian: fit mean/cov per step, compute Mahalanobis distance.
    2. KNN: store reference states, compute distance to k nearest.
    3. Buffer: maintain a rolling buffer of recent successful states.
    """

    def __init__(
        self,
        method: str = "knn",
        k: int = 5,
        buffer_size: int = 200,
        anomaly_threshold: float = 3.0,
    ):
        self.method = method
        self.k = k
        self.buffer_size = buffer_size
        self.anomaly_threshold = anomaly_threshold

        # Per-step reference data
        # step -> list of feature vectors
        self._references: dict[int, list[np.ndarray]] = {}
        # step -> (mean, cov_inv) for Gaussian mode
        self._gaussian_params: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # Rolling buffer for online updates
        self._buffers: dict[int, deque] = {}

    def fit(self, successful_runs: list):
        """Fit from successful runs.

        Each run should have frames with scene_state and step_estimate.
        """
        per_step_states: dict[int, list[np.ndarray]] = {}

        for run in successful_runs:
            frames = run.get("frames", []) if isinstance(run, dict) else run.frames
            for frame in frames:
                state = frame.get("scene_state") if isinstance(frame, dict) else frame.scene_state
                step = frame.get("step_estimate") if isinstance(frame, dict) else frame.step_estimate
                if state is None or step is None:
                    continue
                vec = state_to_vector(state)
                per_step_states.setdefault(step, []).append(vec)

        for step, vecs in per_step_states.items():
            self._references[step] = vecs
            self._buffers[step] = deque(vecs[-self.buffer_size:], maxlen=self.buffer_size)

            # Fit Gaussian if enough data
            if len(vecs) >= 10:
                arr = np.array(vecs)
                mean = arr.mean(axis=0)
                cov = np.cov(arr, rowvar=False)
                # Regularize covariance
                cov += np.eye(cov.shape[0]) * 1e-4
                try:
                    cov_inv = np.linalg.inv(cov)
                    self._gaussian_params[step] = (mean, cov_inv)
                except np.linalg.LinAlgError:
                    pass

    def score(self, step: int, state: dict) -> float:
        """Compute anomaly score for a scene state at a given step.

        Returns a normalized score where higher = more anomalous.
        Score of 0.0 means not enough data to judge.
        """
        vec = state_to_vector(state)

        if self.method == "gaussian":
            return self._gaussian_score(step, vec)
        elif self.method == "knn":
            return self._knn_score(step, vec)
        else:
            return self._knn_score(step, vec)

    def _gaussian_score(self, step: int, vec: np.ndarray) -> float:
        """Mahalanobis distance from the learned Gaussian."""
        params = self._gaussian_params.get(step)
        if params is None:
            return 0.0
        mean, cov_inv = params
        diff = vec - mean
        dist = float(np.sqrt(diff @ cov_inv @ diff))
        return dist

    def _knn_score(self, step: int, vec: np.ndarray) -> float:
        """Average distance to k nearest reference states."""
        refs = list(self._buffers.get(step, []))
        if len(refs) < self.k:
            refs = self._references.get(step, [])
        if len(refs) < 2:
            return 0.0

        distances = [float(np.linalg.norm(vec - ref)) for ref in refs]
        distances.sort()
        k = min(self.k, len(distances))
        avg_dist = sum(distances[:k]) / k
        return avg_dist

    def is_anomalous(self, step: int, state: dict) -> bool:
        """Quick check: is the current state anomalous?"""
        return self.score(step, state) > self.anomaly_threshold

    def add_to_buffer(self, step: int, state: dict):
        """Add a successful state to the rolling buffer (online update)."""
        vec = state_to_vector(state)
        if step not in self._buffers:
            self._buffers[step] = deque(maxlen=self.buffer_size)
        self._buffers[step].append(vec)

    def refit_from_buffer(self):
        """Refit Gaussian params from current buffers."""
        for step, buf in self._buffers.items():
            vecs = list(buf)
            if len(vecs) >= 10:
                arr = np.array(vecs)
                mean = arr.mean(axis=0)
                cov = np.cov(arr, rowvar=False)
                cov += np.eye(cov.shape[0]) * 1e-4
                try:
                    cov_inv = np.linalg.inv(cov)
                    self._gaussian_params[step] = (mean, cov_inv)
                except np.linalg.LinAlgError:
                    pass
