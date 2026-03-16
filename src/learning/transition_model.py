"""Layer 2: Transition probability model.

Learns P(next_step | current_step) from observed runs.
Maintains a belief distribution over steps using Bayesian updates.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class TransitionModel:
    """Learns step transition probabilities and maintains belief state.

    The transition matrix captures nominal sequences and common deviations.
    At inference, a belief vector tracks the probability distribution over
    all steps given the observation history.
    """

    def __init__(
        self,
        num_steps: int = 10,
        smoothing: float = 0.01,
        observation_noise: float = 0.2,
    ):
        self.num_steps = num_steps
        self.smoothing = smoothing
        self.observation_noise = observation_noise

        # Transition counts: counts[i][j] = # times step i -> step j
        self._counts = np.full((num_steps, num_steps), smoothing)
        # Transition probability matrix (normalized counts)
        self._transition_matrix = np.ones((num_steps, num_steps)) / num_steps
        # Current belief vector
        self._belief = np.zeros(num_steps)
        self._belief[0] = 1.0  # start at step 0

    def fit(self, runs: list):
        """Fit transition matrix from completed runs.

        Each run should have a list of frames with step_estimate fields.
        """
        self._counts = np.full((self.num_steps, self.num_steps), self.smoothing)

        for run in runs:
            frames = run.get("frames", []) if isinstance(run, dict) else run.frames
            prev_step = None
            for frame in frames:
                step = frame.get("step_estimate") if isinstance(frame, dict) else frame.step_estimate
                if step is None:
                    continue
                if prev_step is not None and prev_step != step:
                    if 0 <= prev_step < self.num_steps and 0 <= step < self.num_steps:
                        self._counts[prev_step][step] += 1
                prev_step = step

        self._normalize()

    def _normalize(self):
        """Normalize counts into a probability matrix."""
        row_sums = self._counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        self._transition_matrix = self._counts / row_sums

    def update_counts(self, from_step: int, to_step: int):
        """Incrementally update transition counts (online)."""
        if 0 <= from_step < self.num_steps and 0 <= to_step < self.num_steps:
            self._counts[from_step][to_step] += 1
            self._normalize()

    def predict_next(self, current_step: int) -> np.ndarray:
        """Return probability distribution over next steps."""
        if 0 <= current_step < self.num_steps:
            return self._transition_matrix[current_step].copy()
        return np.ones(self.num_steps) / self.num_steps

    def update_belief(self, observed_step: int, step_confidence: float = 1.0) -> np.ndarray:
        """Bayesian belief update given an observation.

        belief_t+1 ∝ P(obs | step) * P(step | prev) * belief_t

        Args:
            observed_step: The step detected by the FSM.
            step_confidence: Confidence in the observation (0-1).

        Returns:
            Updated belief vector.
        """
        # Prediction step: propagate belief through transition matrix
        predicted = np.zeros(self.num_steps)
        for i in range(self.num_steps):
            predicted += self._belief[i] * self._transition_matrix[i]

        # Observation likelihood: peaked at observed_step, spread by noise
        likelihood = np.full(self.num_steps, self.observation_noise / self.num_steps)
        if 0 <= observed_step < self.num_steps:
            likelihood[observed_step] = step_confidence * (1.0 - self.observation_noise) + \
                                         self.observation_noise / self.num_steps

        # Bayes update
        self._belief = predicted * likelihood
        total = self._belief.sum()
        if total > 0:
            self._belief /= total
        else:
            self._belief = np.ones(self.num_steps) / self.num_steps

        return self._belief.copy()

    def reset_belief(self):
        """Reset belief to initial state."""
        self._belief = np.zeros(self.num_steps)
        self._belief[0] = 1.0

    @property
    def belief(self) -> np.ndarray:
        return self._belief.copy()

    @property
    def belief_entropy(self) -> float:
        """Shannon entropy of the belief distribution. High = uncertain."""
        b = self._belief[self._belief > 0]
        return -float(np.sum(b * np.log2(b)))

    @property
    def max_belief_step(self) -> int:
        """Step with highest belief probability."""
        return int(np.argmax(self._belief))

    @property
    def max_belief_confidence(self) -> float:
        """Confidence of the most likely step."""
        return float(np.max(self._belief))

    def is_uncertain(self, entropy_threshold: float = 2.0) -> bool:
        """True if the belief distribution is highly uncertain."""
        return self.belief_entropy > entropy_threshold

    def is_bimodal(self, gap_threshold: float = 0.3) -> bool:
        """True if belief is split between two likely steps."""
        sorted_beliefs = np.sort(self._belief)[::-1]
        if len(sorted_beliefs) < 2:
            return False
        return sorted_beliefs[1] > gap_threshold * sorted_beliefs[0]

    def transition_matrix_to_dict(self) -> dict:
        """Serialize transition matrix for storage."""
        return {
            "counts": self._counts.tolist(),
            "matrix": self._transition_matrix.tolist(),
        }

    def load_from_dict(self, data: dict):
        """Load from serialized form."""
        if "counts" in data:
            self._counts = np.array(data["counts"])
            self._normalize()
