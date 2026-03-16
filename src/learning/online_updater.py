"""Layer 5: Online adaptation.

Manages incremental updates to all learning layers after each run completes.
Handles different update cadences per layer:
- Duration model: EMA update after every run
- Transition model: count-based update after every run
- Anomaly detector: buffer update per run, periodic refit
- Failure classifier: batch retrain on schedule
"""

import json
import os
import time
from typing import Optional

from src.data.records import RunRecord, RunStore
from src.learning.duration_model import StepDurationModel
from src.learning.transition_model import TransitionModel
from src.learning.anomaly_detector import AnomalyDetector
from src.learning.failure_classifier import FailureClassifier


class OnlineUpdater:
    """Coordinates online updates across all learning layers."""

    def __init__(
        self,
        duration_model: StepDurationModel,
        transition_model: TransitionModel,
        anomaly_detector: AnomalyDetector,
        failure_classifier: FailureClassifier,
        run_store: RunStore,
        model_dir: str = "models/learned",
        anomaly_refit_every: int = 10,    # refit anomaly Gaussian every N runs
        classifier_retrain_every: int = 20,  # retrain classifier every N runs
    ):
        self.duration = duration_model
        self.transition = transition_model
        self.anomaly = anomaly_detector
        self.classifier = failure_classifier
        self.run_store = run_store
        self.model_dir = model_dir

        self.anomaly_refit_every = anomaly_refit_every
        self.classifier_retrain_every = classifier_retrain_every
        self._runs_since_anomaly_refit = 0
        self._runs_since_classifier_retrain = 0

        os.makedirs(model_dir, exist_ok=True)

    def on_run_complete(self, run: RunRecord):
        """Called after each run finishes. Updates all layers appropriately."""
        # Save the run
        self.run_store.save(run)

        # Layer 1: Duration model — EMA update (every run)
        self._update_duration(run)

        # Layer 2: Transition model — count update (every run)
        self._update_transitions(run)

        # Layer 3: Anomaly detector — buffer update (every run), refit periodically
        self._update_anomaly(run)

        # Layer 4: Failure classifier — retrain periodically
        self._runs_since_classifier_retrain += 1
        if self._runs_since_classifier_retrain >= self.classifier_retrain_every:
            self._retrain_classifier()
            self._runs_since_classifier_retrain = 0

        # Persist model states
        self._save_models()

    def _update_duration(self, run: RunRecord):
        """EMA update of step duration statistics."""
        if run.outcome != "success":
            return
        for step_str, dur in run.step_durations.items():
            step = int(step_str)
            self.duration.update_online(step, float(dur), user_id=run.user_id)

    def _update_transitions(self, run: RunRecord):
        """Add observed transitions to the count matrix."""
        prev_step = None
        for frame in run.frames:
            step = frame.step_estimate
            if prev_step is not None and prev_step != step:
                self.transition.update_counts(prev_step, step)
            prev_step = step

    def _update_anomaly(self, run: RunRecord):
        """Add successful states to buffer, periodically refit Gaussian."""
        if run.outcome == "success":
            for frame in run.frames:
                state = frame.scene_state
                if isinstance(state, dict):
                    self.anomaly.add_to_buffer(frame.step_estimate, state)

        self._runs_since_anomaly_refit += 1
        if self._runs_since_anomaly_refit >= self.anomaly_refit_every:
            self.anomaly.refit_from_buffer()
            self._runs_since_anomaly_refit = 0

    def _retrain_classifier(self):
        """Full retrain of the failure classifier from all stored runs."""
        from src.learning.failure_classifier import extract_features, LOOKBACK_WINDOWS

        all_runs = self.run_store.load_all()
        if len(all_runs) < 20:
            return  # not enough data

        success_runs = [r for r in all_runs if r.outcome == "success"]
        failure_runs = [r for r in all_runs if r.outcome == "failure"]
        if len(failure_runs) < 5:
            return  # need some failures

        X_list = []
        y_list = []

        # Positive samples: states near failure points
        for run in failure_runs:
            if run.failure_frame is None:
                continue
            for lookback in LOOKBACK_WINDOWS:
                idx = run.failure_frame - lookback
                if 0 <= idx < len(run.frames):
                    frame = run.frames[idx]
                    recent = [
                        {"scene_state": f.scene_state, "step_estimate": f.step_estimate}
                        for f in run.frames[max(0, idx - 50):idx + 1]
                    ]
                    features = extract_features(
                        recent,
                        current_step=frame.step_estimate,
                        expected_duration=self.duration.get_expected_duration(frame.step_estimate),
                        current_duration=float(run.step_durations.get(str(frame.step_estimate), 0)),
                        anomaly_score=frame.anomaly_score,
                        step_regressions=0,
                    )
                    X_list.append(features)
                    y_list.append(1)

        # Negative samples: random states from successful runs
        import random
        for run in success_runs:
            if len(run.frames) < 50:
                continue
            for _ in range(min(4, len(run.frames) // 50)):
                idx = random.randint(50, len(run.frames) - 1)
                frame = run.frames[idx]
                recent = [
                    {"scene_state": f.scene_state, "step_estimate": f.step_estimate}
                    for f in run.frames[max(0, idx - 50):idx + 1]
                ]
                features = extract_features(
                    recent,
                    current_step=frame.step_estimate,
                    expected_duration=self.duration.get_expected_duration(frame.step_estimate),
                    current_duration=float(run.step_durations.get(str(frame.step_estimate), 0)),
                    anomaly_score=0.0,
                    step_regressions=0,
                )
                X_list.append(features)
                y_list.append(0)

        if len(X_list) < 20:
            return

        import numpy as np
        X = np.array(X_list)
        y = np.array(y_list)
        self.classifier.train(X, y)
        self.classifier.save(os.path.join(self.model_dir, "failure_classifier.pkl"))

    def _save_models(self):
        """Persist all model states to disk."""
        # Duration model
        duration_data = self.duration.serialize()
        with open(os.path.join(self.model_dir, "duration_model.json"), "w") as f:
            json.dump(duration_data, f)

        # Transition model
        trans_data = self.transition.transition_matrix_to_dict()
        with open(os.path.join(self.model_dir, "transition_model.json"), "w") as f:
            json.dump(trans_data, f)

    def load_models(self):
        """Load all persisted model states."""
        # Duration
        path = os.path.join(self.model_dir, "duration_model.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                self.duration.deserialize(json.load(f))

        # Transition
        path = os.path.join(self.model_dir, "transition_model.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                self.transition.load_from_dict(json.load(f))

        # Failure classifier
        path = os.path.join(self.model_dir, "failure_classifier.pkl")
        if os.path.exists(path):
            self.classifier.load(path)

    def initial_fit(self):
        """Initial fit of all models from stored runs."""
        all_runs = self.run_store.load_all()
        if not all_runs:
            return

        success_runs = [r for r in all_runs if r.outcome == "success"]

        # Duration model
        summaries = [
            {"step_durations": r.step_durations, "user_id": r.user_id}
            for r in success_runs
        ]
        if summaries:
            self.duration.fit(summaries)

        # Transition model
        run_dicts = [r.to_dict() for r in all_runs]
        if run_dicts:
            self.transition.fit(run_dicts)

        # Anomaly detector
        success_dicts = [r.to_dict() for r in success_runs]
        if success_dicts:
            self.anomaly.fit(success_dicts)

        # Failure classifier (only if enough data)
        if len(all_runs) >= 20:
            self._retrain_classifier()
