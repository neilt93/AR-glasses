"""Layer 4: Failure precursor classifier.

Learns to predict P(failure ahead) from a window of recent scene states.
Starts with gradient-boosted trees on hand-engineered features.
Upgradeable to LSTM over raw feature sequences.
"""

import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.learning.anomaly_detector import state_to_vector, ALL_FEATURES


# Lookback windows (in frames) for extracting failure precursor features
LOOKBACK_WINDOWS = [10, 20, 30, 50]


@dataclass
class FailurePrediction:
    """Output of the failure classifier."""
    risk: float             # P(failure ahead), 0.0 - 1.0
    top_features: list      # most contributing features (if available)
    confidence: float       # confidence in the prediction


def extract_features(
    recent_frames: list[dict],
    current_step: int,
    expected_duration: float,
    current_duration: float,
    anomaly_score: float,
    step_regressions: int,
) -> np.ndarray:
    """Extract hand-engineered features for the failure classifier.

    Args:
        recent_frames: last N frames with scene_state, step_estimate, etc.
        current_step: current step number.
        expected_duration: expected duration from duration model.
        current_duration: actual frames in current step so far.
        anomaly_score: from Layer 3.
        step_regressions: number of backward step transitions this run.

    Returns:
        Feature vector for the classifier.
    """
    features = []

    # Feature 1: current step
    features.append(float(current_step))

    # Feature 2: duration ratio (current / expected)
    if expected_duration > 0:
        features.append(current_duration / expected_duration)
    else:
        features.append(0.0)

    # Feature 3: anomaly score
    features.append(anomaly_score)

    # Feature 4: step regressions count
    features.append(float(step_regressions))

    # Feature 5-13: current scene state features
    if recent_frames:
        latest = recent_frames[-1]
        state = latest.get("scene_state", {}) if isinstance(latest, dict) else latest.scene_state
        if isinstance(state, dict):
            vec = state_to_vector(state)
            features.extend(vec.tolist())
        else:
            features.extend([0.0] * len(ALL_FEATURES))
    else:
        features.extend([0.0] * len(ALL_FEATURES))

    # Feature 14+: temporal features from lookback windows
    for window in LOOKBACK_WINDOWS:
        if len(recent_frames) >= window:
            # Step stability: how many unique steps in the window
            steps_in_window = set()
            for f in recent_frames[-window:]:
                s = f.get("step_estimate") if isinstance(f, dict) else f.step_estimate
                steps_in_window.add(s)
            features.append(float(len(steps_in_window)))

            # Object presence stability: fraction of frames where key objects present
            plate_count = 0
            bracket_count = 0
            driver_count = 0
            for f in recent_frames[-window:]:
                state = f.get("scene_state", {}) if isinstance(f, dict) else f.scene_state
                if isinstance(state, dict):
                    plate_count += int(state.get("base_plate_visible", False))
                    bracket_count += int(state.get("bracket_visible", False))
                    driver_count += int(state.get("screwdriver_visible", False))
            features.append(plate_count / window)
            features.append(bracket_count / window)
            features.append(driver_count / window)
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float64)


class FailureClassifier:
    """Predicts P(failure ahead) from recent scene state history.

    Uses gradient boosted trees (sklearn) when available.
    Falls back to a simple heuristic scorer without sklearn.
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model = None
        self._feature_names: list[str] = []
        self._trained = False

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the classifier on labeled data.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Binary labels (1 = failure, 0 = success).
        """
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            self._model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )
            self._model.fit(X, y)
            self._trained = True
        except ImportError:
            print("[failure_classifier] sklearn not available — using heuristic fallback")
            self._trained = False

    def predict(
        self,
        recent_frames: list[dict],
        current_step: int,
        expected_duration: float,
        current_duration: float,
        anomaly_score: float,
        step_regressions: int,
    ) -> FailurePrediction:
        """Predict failure risk from current state."""
        features = extract_features(
            recent_frames, current_step, expected_duration,
            current_duration, anomaly_score, step_regressions,
        )

        if self._trained and self._model is not None:
            return self._model_predict(features)
        return self._heuristic_predict(features, anomaly_score, current_duration, expected_duration)

    def _model_predict(self, features: np.ndarray) -> FailurePrediction:
        """Use the trained model."""
        X = features.reshape(1, -1)
        risk = float(self._model.predict_proba(X)[0][1])

        top_features = []
        if hasattr(self._model, "feature_importances_"):
            importances = self._model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:3]
            top_features = [
                {"index": int(i), "importance": float(importances[i])}
                for i in top_idx
            ]

        return FailurePrediction(
            risk=risk,
            top_features=top_features,
            confidence=0.8 if self._model is not None else 0.3,
        )

    def _heuristic_predict(
        self,
        features: np.ndarray,
        anomaly_score: float,
        current_duration: float,
        expected_duration: float,
    ) -> FailurePrediction:
        """Heuristic fallback when no trained model is available.

        Combines duration deviation and anomaly score into a risk estimate.
        """
        risk = 0.0

        # Duration factor
        if expected_duration > 0:
            ratio = current_duration / expected_duration
            if ratio > 2.0:
                risk += 0.4
            elif ratio > 1.5:
                risk += 0.2

        # Anomaly factor
        if anomaly_score > 5.0:
            risk += 0.4
        elif anomaly_score > 3.0:
            risk += 0.2

        risk = min(risk, 1.0)
        return FailurePrediction(risk=risk, top_features=[], confidence=0.3)

    def save(self, path: str):
        """Save trained model to disk."""
        if self._model is not None:
            with open(path, "wb") as f:
                pickle.dump(self._model, f)

    def load(self, path: str):
        """Load trained model from disk."""
        if os.path.exists(path):
            with open(path, "rb") as f:
                self._model = pickle.load(f)
            self._trained = True

    @property
    def is_trained(self) -> bool:
        return self._trained
