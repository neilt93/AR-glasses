"""Layer 1: Step duration models.

For each step k, fit a Gaussian over observed durations from successful runs.
At inference, compute z-score of current duration to detect stalls.
Supports per-user personalization via EMA updates.
"""

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StepStats:
    """Running statistics for a single step's duration."""
    mean: float = 0.0
    std: float = 1.0
    count: int = 0
    min_observed: float = float("inf")
    max_observed: float = 0.0


class StepDurationModel:
    """Learns and evaluates expected duration per assembly step.

    Trained from successful runs. At inference, produces z-scores
    indicating how far the current duration deviates from normal.
    """

    def __init__(
        self,
        soft_warning_z: float = 2.0,
        hard_warning_z: float = 3.0,
        ema_alpha: float = 0.1,
        min_std: float = 5.0,
    ):
        self.soft_z = soft_warning_z
        self.hard_z = hard_warning_z
        self.ema_alpha = ema_alpha
        self.min_std = min_std  # floor to prevent division issues early on

        # step_num -> StepStats
        self._global_stats: dict[int, StepStats] = {}
        # user_id -> step_num -> StepStats
        self._user_stats: dict[str, dict[int, StepStats]] = {}

    def fit(self, successful_runs: list[dict]):
        """Fit from a list of run summaries (with step_durations dicts).

        Each run should have:
            step_durations: {step_num_str: frame_count, ...}
            user_id: str
        """
        # Collect durations per step
        per_step: dict[int, list[float]] = {}
        per_user_step: dict[str, dict[int, list[float]]] = {}

        for run in successful_runs:
            user = run.get("user_id", "default")
            durations = run.get("step_durations", {})
            for step_str, dur in durations.items():
                step = int(step_str)
                per_step.setdefault(step, []).append(float(dur))
                per_user_step.setdefault(user, {}).setdefault(step, []).append(float(dur))

        # Fit global stats
        for step, durs in per_step.items():
            self._global_stats[step] = self._compute_stats(durs)

        # Fit per-user stats
        for user, steps in per_user_step.items():
            self._user_stats[user] = {}
            for step, durs in steps.items():
                self._user_stats[user][step] = self._compute_stats(durs)

    def _compute_stats(self, durations: list[float]) -> StepStats:
        n = len(durations)
        if n == 0:
            return StepStats()
        mean = sum(durations) / n
        if n > 1:
            variance = sum((d - mean) ** 2 for d in durations) / (n - 1)
            std = max(math.sqrt(variance), self.min_std)
        else:
            std = self.min_std
        return StepStats(
            mean=mean,
            std=std,
            count=n,
            min_observed=min(durations),
            max_observed=max(durations),
        )

    def z_score(
        self, step: int, current_duration: float, user_id: Optional[str] = None
    ) -> float:
        """Compute z-score for the current step duration.

        Uses per-user stats if available, falls back to global.
        """
        stats = self._get_stats(step, user_id)
        if stats.count < 2:
            return 0.0  # not enough data
        return (current_duration - stats.mean) / stats.std

    def warning_level(
        self, step: int, current_duration: float, user_id: Optional[str] = None
    ) -> str:
        """Return warning level: 'none', 'soft', or 'hard'."""
        z = self.z_score(step, current_duration, user_id)
        if z > self.hard_z:
            return "hard"
        elif z > self.soft_z:
            return "soft"
        return "none"

    def update_online(self, step: int, observed_duration: float, user_id: str = "default"):
        """EMA update after a completed run — no full refit needed."""
        # Global
        stats = self._global_stats.get(step, StepStats())
        stats = self._ema_update(stats, observed_duration)
        self._global_stats[step] = stats

        # Per-user
        if user_id not in self._user_stats:
            self._user_stats[user_id] = {}
        user_stats = self._user_stats[user_id].get(step, StepStats())
        user_stats = self._ema_update(user_stats, observed_duration)
        self._user_stats[user_id][step] = user_stats

    def _ema_update(self, stats: StepStats, new_val: float) -> StepStats:
        """Exponential moving average update of mean and std."""
        alpha = self.ema_alpha
        if stats.count == 0:
            stats.mean = new_val
            stats.std = self.min_std
            stats.count = 1
        else:
            stats.mean = alpha * new_val + (1 - alpha) * stats.mean
            deviation = abs(new_val - stats.mean)
            stats.std = alpha * deviation + (1 - alpha) * stats.std
            stats.std = max(stats.std, self.min_std)
            stats.count += 1
        stats.min_observed = min(stats.min_observed, new_val)
        stats.max_observed = max(stats.max_observed, new_val)
        return stats

    def _get_stats(self, step: int, user_id: Optional[str]) -> StepStats:
        """Get stats for a step, preferring per-user if available."""
        if user_id and user_id in self._user_stats:
            user_step_stats = self._user_stats[user_id].get(step)
            if user_step_stats and user_step_stats.count >= 5:
                return user_step_stats
        return self._global_stats.get(step, StepStats())

    def get_expected_duration(self, step: int, user_id: Optional[str] = None) -> float:
        """Return expected duration in frames for a step."""
        return self._get_stats(step, user_id).mean

    def serialize(self) -> dict:
        """Serialize model state for persistence."""
        return {
            "global": {
                str(k): {"mean": v.mean, "std": v.std, "count": v.count,
                          "min": v.min_observed, "max": v.max_observed}
                for k, v in self._global_stats.items()
            },
            "users": {
                uid: {
                    str(k): {"mean": v.mean, "std": v.std, "count": v.count,
                              "min": v.min_observed, "max": v.max_observed}
                    for k, v in steps.items()
                }
                for uid, steps in self._user_stats.items()
            },
        }

    def deserialize(self, data: dict):
        """Load model state from serialized form."""
        for step_str, vals in data.get("global", {}).items():
            self._global_stats[int(step_str)] = StepStats(
                mean=vals["mean"], std=vals["std"], count=vals["count"],
                min_observed=vals.get("min", float("inf")),
                max_observed=vals.get("max", 0.0),
            )
        for uid, steps in data.get("users", {}).items():
            self._user_stats[uid] = {}
            for step_str, vals in steps.items():
                self._user_stats[uid][int(step_str)] = StepStats(
                    mean=vals["mean"], std=vals["std"], count=vals["count"],
                    min_observed=vals.get("min", float("inf")),
                    max_observed=vals.get("max", 0.0),
                )
