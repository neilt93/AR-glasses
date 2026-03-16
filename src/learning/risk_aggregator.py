"""Risk aggregator and interrupt decision logic.

Combines signals from all learning layers into a single risk score
and decides whether to surface a warning on the HUD.

Pipeline per frame:
    detections → scene_state →
      [duration_model, transition_model, anomaly_detector, failure_classifier]
      → aggregate_risk_score →
      interrupt_decision →
      HUD update
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from src.learning.duration_model import StepDurationModel
from src.learning.transition_model import TransitionModel
from src.learning.anomaly_detector import AnomalyDetector
from src.learning.failure_classifier import FailureClassifier, FailurePrediction


@dataclass
class RiskAssessment:
    """Combined risk assessment from all layers."""
    # Individual scores
    duration_z: float = 0.0
    duration_warning: str = "none"
    transition_entropy: float = 0.0
    transition_uncertain: bool = False
    transition_bimodal: bool = False
    anomaly_score: float = 0.0
    anomaly_flagged: bool = False
    failure_risk: float = 0.0

    # Aggregate
    combined_risk: float = 0.0
    should_interrupt: bool = False
    interrupt_type: str = "none"  # "none", "soft", "hard"
    interrupt_message: str = ""


@dataclass
class InterruptConfig:
    """Tunable parameters for interrupt decision logic."""
    # Risk thresholds
    soft_threshold: float = 0.4
    hard_threshold: float = 0.7

    # Cooldown: minimum seconds between interrupts
    cooldown_seconds: float = 3.0

    # Layer weights for combining scores
    weight_duration: float = 0.25
    weight_transition: float = 0.15
    weight_anomaly: float = 0.25
    weight_failure: float = 0.35

    # Per-layer enable flags
    use_duration: bool = True
    use_transition: bool = True
    use_anomaly: bool = True
    use_failure: bool = True


class RiskAggregator:
    """Aggregates risk signals and decides when to interrupt the user."""

    def __init__(self, config: Optional[InterruptConfig] = None):
        self.config = config or InterruptConfig()
        self._last_interrupt_time: float = 0.0
        self._interrupt_count: int = 0

    def assess(
        self,
        duration_model: StepDurationModel,
        transition_model: TransitionModel,
        anomaly_detector: AnomalyDetector,
        failure_classifier: FailureClassifier,
        current_step: int,
        current_duration: float,
        scene_state: dict,
        recent_frames: list[dict],
        step_regressions: int = 0,
        user_id: Optional[str] = None,
    ) -> RiskAssessment:
        """Run all layers and produce a combined risk assessment."""
        assessment = RiskAssessment()

        # Layer 1: Duration
        if self.config.use_duration:
            assessment.duration_z = duration_model.z_score(
                current_step, current_duration, user_id
            )
            assessment.duration_warning = duration_model.warning_level(
                current_step, current_duration, user_id
            )

        # Layer 2: Transition
        if self.config.use_transition:
            assessment.transition_entropy = transition_model.belief_entropy
            assessment.transition_uncertain = transition_model.is_uncertain()
            assessment.transition_bimodal = transition_model.is_bimodal()

        # Layer 3: Anomaly
        if self.config.use_anomaly:
            assessment.anomaly_score = anomaly_detector.score(current_step, scene_state)
            assessment.anomaly_flagged = anomaly_detector.is_anomalous(current_step, scene_state)

        # Layer 4: Failure classifier
        if self.config.use_failure:
            expected_dur = duration_model.get_expected_duration(current_step, user_id)
            prediction = failure_classifier.predict(
                recent_frames=recent_frames,
                current_step=current_step,
                expected_duration=expected_dur,
                current_duration=current_duration,
                anomaly_score=assessment.anomaly_score,
                step_regressions=step_regressions,
            )
            assessment.failure_risk = prediction.risk

        # Combine into aggregate risk
        assessment.combined_risk = self._combine(assessment)

        # Interrupt decision
        self._decide_interrupt(assessment)

        return assessment

    def _combine(self, a: RiskAssessment) -> float:
        """Weighted combination of layer signals into [0, 1] risk score."""
        cfg = self.config
        score = 0.0
        total_weight = 0.0

        if cfg.use_duration:
            # Normalize z-score to 0-1 range
            dur_signal = min(1.0, max(0.0, a.duration_z / 4.0))
            score += cfg.weight_duration * dur_signal
            total_weight += cfg.weight_duration

        if cfg.use_transition:
            # Normalize entropy (max ~3.3 for 10 steps)
            trans_signal = min(1.0, a.transition_entropy / 3.0)
            if a.transition_bimodal:
                trans_signal = max(trans_signal, 0.6)
            score += cfg.weight_transition * trans_signal
            total_weight += cfg.weight_transition

        if cfg.use_anomaly:
            # Normalize anomaly score (threshold-relative)
            anom_signal = min(1.0, a.anomaly_score / 6.0)
            score += cfg.weight_anomaly * anom_signal
            total_weight += cfg.weight_anomaly

        if cfg.use_failure:
            score += cfg.weight_failure * a.failure_risk
            total_weight += cfg.weight_failure

        if total_weight > 0:
            score /= total_weight

        return min(1.0, max(0.0, score))

    def _decide_interrupt(self, assessment: RiskAssessment):
        """Decide whether to surface an interrupt based on risk and cooldown."""
        now = time.time()
        on_cooldown = (now - self._last_interrupt_time) < self.config.cooldown_seconds

        risk = assessment.combined_risk

        if risk >= self.config.hard_threshold and not on_cooldown:
            assessment.should_interrupt = True
            assessment.interrupt_type = "hard"
            assessment.interrupt_message = self._build_message(assessment, "hard")
            self._last_interrupt_time = now
            self._interrupt_count += 1
        elif risk >= self.config.soft_threshold and not on_cooldown:
            assessment.should_interrupt = True
            assessment.interrupt_type = "soft"
            assessment.interrupt_message = self._build_message(assessment, "soft")
            self._last_interrupt_time = now
            self._interrupt_count += 1
        else:
            assessment.should_interrupt = False
            assessment.interrupt_type = "none"
            assessment.interrupt_message = ""

    def _build_message(self, a: RiskAssessment, level: str) -> str:
        """Build a human-readable interrupt message from the assessment."""
        parts = []

        if a.duration_warning == "hard":
            parts.append("Step is taking much longer than expected.")
        elif a.duration_warning == "soft":
            parts.append("Step is taking longer than usual.")

        if a.anomaly_flagged:
            parts.append("Current state looks unusual for this step.")

        if a.transition_bimodal:
            parts.append("Uncertain which step you're on.")

        if a.failure_risk > 0.6:
            parts.append("High risk of failure detected.")
        elif a.failure_risk > 0.3:
            parts.append("Elevated failure risk.")

        if not parts:
            if level == "hard":
                parts.append("Something may be wrong — check your work.")
            else:
                parts.append("Possible issue detected.")

        return " ".join(parts)

    @property
    def interrupt_count(self) -> int:
        return self._interrupt_count

    def reset(self):
        self._last_interrupt_time = 0.0
        self._interrupt_count = 0
