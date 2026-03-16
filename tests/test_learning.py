"""Tests for learning layers (duration model, transition model, anomaly detector, failure classifier, risk aggregator)."""

import math
import pytest
import numpy as np

from src.learning.duration_model import StepDurationModel, StepStats
from src.learning.transition_model import TransitionModel
from src.learning.anomaly_detector import AnomalyDetector, state_to_vector, ALL_FEATURES
from src.learning.failure_classifier import FailureClassifier, FailurePrediction, extract_features
from src.learning.risk_aggregator import RiskAggregator, RiskAssessment, InterruptConfig


# ─── Duration Model ───────────────────────────────────────────────────────────

class TestStepDurationModel:
    def _make_runs(self):
        """Create a few fake successful runs with step_durations."""
        return [
            {"user_id": "alice", "step_durations": {"1": 30, "2": 50, "3": 40}},
            {"user_id": "alice", "step_durations": {"1": 32, "2": 48, "3": 42}},
            {"user_id": "bob",   "step_durations": {"1": 28, "2": 55, "3": 38}},
            {"user_id": "bob",   "step_durations": {"1": 35, "2": 52, "3": 45}},
        ]

    def test_fit_populates_stats(self):
        model = StepDurationModel()
        model.fit(self._make_runs())
        stats = model._global_stats
        assert 1 in stats
        assert stats[1].count == 4
        assert stats[1].mean > 0

    def test_z_score_zero_when_no_data(self):
        model = StepDurationModel()
        assert model.z_score(1, 100.0) == 0.0

    def test_z_score_normal_duration(self):
        model = StepDurationModel(min_std=1.0)
        model.fit(self._make_runs())
        # Step 1 mean ≈ 31.25 → a duration of 31 should have z ≈ 0
        z = model.z_score(1, 31.25)
        assert abs(z) < 0.5

    def test_z_score_high_for_long_duration(self):
        model = StepDurationModel(min_std=1.0)
        model.fit(self._make_runs())
        z = model.z_score(1, 200.0)
        assert z > 3.0

    def test_warning_levels(self):
        model = StepDurationModel(soft_warning_z=2.0, hard_warning_z=3.0, min_std=1.0)
        model.fit(self._make_runs())
        assert model.warning_level(1, 31.0) == "none"
        assert model.warning_level(1, 200.0) == "hard"

    def test_per_user_stats(self):
        model = StepDurationModel()
        # Need >= 5 runs for per-user to be used
        runs = [{"user_id": "alice", "step_durations": {"1": 30 + i}} for i in range(6)]
        model.fit(runs)
        assert "alice" in model._user_stats
        assert 1 in model._user_stats["alice"]

    def test_online_update(self):
        model = StepDurationModel()
        model.update_online(1, 50.0, "user1")
        assert model._global_stats[1].count == 1
        assert model._global_stats[1].mean == 50.0

    def test_serialize_deserialize(self):
        model = StepDurationModel()
        model.fit(self._make_runs())
        data = model.serialize()

        model2 = StepDurationModel()
        model2.deserialize(data)
        assert model2._global_stats[1].mean == pytest.approx(model._global_stats[1].mean)
        assert model2._global_stats[1].count == model._global_stats[1].count

    def test_get_expected_duration(self):
        model = StepDurationModel()
        model.fit(self._make_runs())
        dur = model.get_expected_duration(1)
        assert dur > 0


# ─── Transition Model ─────────────────────────────────────────────────────────

class TestTransitionModel:
    def test_initial_belief(self):
        tm = TransitionModel(num_steps=10)
        assert tm.belief[0] == pytest.approx(1.0)
        assert tm.max_belief_step == 0

    def test_predict_next_uniform_without_training(self):
        tm = TransitionModel(num_steps=5)
        probs = tm.predict_next(0)
        assert len(probs) == 5
        assert sum(probs) == pytest.approx(1.0)

    def test_fit_learns_transitions(self):
        tm = TransitionModel(num_steps=5, smoothing=0.001)
        runs = [
            {"frames": [
                {"step_estimate": 0}, {"step_estimate": 1},
                {"step_estimate": 2}, {"step_estimate": 3},
            ]},
            {"frames": [
                {"step_estimate": 0}, {"step_estimate": 1},
                {"step_estimate": 2}, {"step_estimate": 3},
            ]},
        ]
        tm.fit(runs)
        probs = tm.predict_next(0)
        # Should strongly predict 0→1
        assert probs[1] > 0.5

    def test_update_belief(self):
        tm = TransitionModel(num_steps=5)
        belief = tm.update_belief(0, step_confidence=1.0)
        assert belief[0] > 0.5  # should still be mostly at step 0

    def test_belief_entropy_concentrated(self):
        tm = TransitionModel(num_steps=5)
        # Concentrated belief → low entropy
        assert tm.belief_entropy < 1.0

    def test_is_uncertain(self):
        tm = TransitionModel(num_steps=10)
        # Spread belief uniformly
        tm._belief = np.ones(10) / 10
        assert tm.is_uncertain() is True

    def test_is_bimodal(self):
        tm = TransitionModel(num_steps=5)
        tm._belief = np.array([0.45, 0.45, 0.05, 0.03, 0.02])
        assert tm.is_bimodal() == True

    def test_not_bimodal_when_peaked(self):
        tm = TransitionModel(num_steps=5)
        tm._belief = np.array([0.9, 0.05, 0.02, 0.02, 0.01])
        assert tm.is_bimodal() == False

    def test_reset_belief(self):
        tm = TransitionModel(num_steps=5)
        tm.update_belief(3)
        tm.reset_belief()
        assert tm.belief[0] == pytest.approx(1.0)

    def test_update_counts_online(self):
        tm = TransitionModel(num_steps=5, smoothing=0.01)
        old_prob = tm.predict_next(0)[1]
        tm.update_counts(0, 1)
        new_prob = tm.predict_next(0)[1]
        assert new_prob > old_prob

    def test_serialization(self):
        tm = TransitionModel(num_steps=3)
        tm.update_counts(0, 1)
        data = tm.transition_matrix_to_dict()
        assert "counts" in data
        assert "matrix" in data

        tm2 = TransitionModel(num_steps=3)
        tm2.load_from_dict(data)
        np.testing.assert_array_almost_equal(tm.transition_matrix, tm2.transition_matrix)


# ─── Anomaly Detector ─────────────────────────────────────────────────────────

class TestAnomalyDetector:
    def _normal_state(self):
        return {
            "base_plate_visible": True, "bracket_visible": True,
            "screwdriver_visible": True, "servo_visible": False,
            "hand_visible": True, "bracket_on_plate": True,
            "screw_near_plate": True, "screwdriver_near_screw": True,
            "screwdriver_engaged": True, "screws_visible": 2,
        }

    def _abnormal_state(self):
        return {
            "base_plate_visible": False, "bracket_visible": False,
            "screwdriver_visible": False, "servo_visible": True,
            "hand_visible": False, "bracket_on_plate": False,
            "screw_near_plate": False, "screwdriver_near_screw": False,
            "screwdriver_engaged": False, "screws_visible": 0,
        }

    def test_state_to_vector_length(self):
        vec = state_to_vector(self._normal_state())
        assert len(vec) == len(ALL_FEATURES)

    def test_score_zero_with_no_data(self):
        ad = AnomalyDetector()
        score = ad.score(1, self._normal_state())
        assert score == 0.0

    def test_fit_and_score_normal(self):
        ad = AnomalyDetector(method="knn", k=3)
        runs = [{"frames": [
            {"step_estimate": 1, "scene_state": self._normal_state()}
            for _ in range(20)
        ]}]
        ad.fit(runs)
        score = ad.score(1, self._normal_state())
        # Normal state should have low score
        assert score < ad.anomaly_threshold

    def test_fit_and_score_anomalous(self):
        ad = AnomalyDetector(method="knn", k=3, anomaly_threshold=1.0)
        runs = [{"frames": [
            {"step_estimate": 1, "scene_state": self._normal_state()}
            for _ in range(20)
        ]}]
        ad.fit(runs)
        score = ad.score(1, self._abnormal_state())
        # Abnormal state should have higher score than normal
        normal_score = ad.score(1, self._normal_state())
        assert score > normal_score

    def test_is_anomalous(self):
        ad = AnomalyDetector(anomaly_threshold=0.5)
        runs = [{"frames": [
            {"step_estimate": 1, "scene_state": self._normal_state()}
            for _ in range(20)
        ]}]
        ad.fit(runs)
        # Normal state should not be anomalous
        assert ad.is_anomalous(1, self._normal_state()) is False

    def test_add_to_buffer(self):
        ad = AnomalyDetector(buffer_size=10)
        ad.add_to_buffer(1, self._normal_state())
        sizes = ad.buffer_sizes()
        assert sizes[1] == 1

    def test_gaussian_mode(self):
        ad = AnomalyDetector(method="gaussian")
        runs = [{"frames": [
            {"step_estimate": 1, "scene_state": self._normal_state()}
            for _ in range(20)
        ]}]
        ad.fit(runs)
        score = ad.score(1, self._normal_state())
        assert isinstance(score, float)


# ─── Failure Classifier ───────────────────────────────────────────────────────

class TestFailureClassifier:
    def test_heuristic_low_risk(self):
        fc = FailureClassifier()
        pred = fc.predict(
            recent_frames=[],
            current_step=3,
            expected_duration=100.0,
            current_duration=50.0,
            anomaly_score=0.0,
            step_regressions=0,
        )
        assert pred.risk == 0.0
        assert pred.confidence == 0.3  # heuristic confidence

    def test_heuristic_high_duration(self):
        fc = FailureClassifier()
        pred = fc.predict(
            recent_frames=[],
            current_step=3,
            expected_duration=50.0,
            current_duration=150.0,  # 3x expected
            anomaly_score=0.0,
            step_regressions=0,
        )
        assert pred.risk > 0.0

    def test_heuristic_high_anomaly(self):
        fc = FailureClassifier()
        pred = fc.predict(
            recent_frames=[],
            current_step=3,
            expected_duration=100.0,
            current_duration=50.0,
            anomaly_score=6.0,
            step_regressions=0,
        )
        assert pred.risk > 0.0

    def test_heuristic_step_regressions(self):
        fc = FailureClassifier()
        pred = fc.predict(
            recent_frames=[],
            current_step=3,
            expected_duration=100.0,
            current_duration=50.0,
            anomaly_score=0.0,
            step_regressions=3,
        )
        assert pred.risk > 0.0

    def test_heuristic_combined_caps_at_1(self):
        fc = FailureClassifier()
        pred = fc.predict(
            recent_frames=[],
            current_step=3,
            expected_duration=10.0,
            current_duration=100.0,  # 10x
            anomaly_score=10.0,
            step_regressions=5,
        )
        assert pred.risk <= 1.0

    def test_extract_features_shape(self):
        feats = extract_features(
            recent_frames=[],
            current_step=1,
            expected_duration=100.0,
            current_duration=50.0,
            anomaly_score=1.0,
            step_regressions=0,
        )
        # 4 base + len(ALL_FEATURES) + 4 * len(LOOKBACK_WINDOWS) temporal
        expected_len = 4 + len(ALL_FEATURES) + 4 * 4
        assert len(feats) == expected_len

    def test_not_trained_by_default(self):
        fc = FailureClassifier()
        assert fc.is_trained is False


# ─── Risk Aggregator ──────────────────────────────────────────────────────────

class TestRiskAggregator:
    def test_combine_all_zero(self):
        agg = RiskAggregator()
        assessment = RiskAssessment()
        result = agg._combine(assessment)
        assert result == pytest.approx(0.0)

    def test_combine_max_risk(self):
        agg = RiskAggregator()
        assessment = RiskAssessment(
            duration_z=4.0,
            transition_entropy=3.0,
            anomaly_score=6.0,
            failure_risk=1.0,
        )
        result = agg._combine(assessment)
        assert 0.0 <= result <= 1.0

    def test_interrupt_not_on_cooldown(self):
        config = InterruptConfig(soft_threshold=0.3, hard_threshold=0.7, cooldown_seconds=0.0)
        agg = RiskAggregator(config=config)
        assessment = RiskAssessment(combined_risk=0.5)
        agg._decide_interrupt(assessment)
        assert assessment.should_interrupt is True
        assert assessment.interrupt_type == "soft"

    def test_hard_interrupt(self):
        config = InterruptConfig(soft_threshold=0.3, hard_threshold=0.7, cooldown_seconds=0.0)
        agg = RiskAggregator(config=config)
        assessment = RiskAssessment(combined_risk=0.8)
        agg._decide_interrupt(assessment)
        assert assessment.interrupt_type == "hard"

    def test_no_interrupt_below_threshold(self):
        config = InterruptConfig(soft_threshold=0.5, cooldown_seconds=0.0)
        agg = RiskAggregator(config=config)
        assessment = RiskAssessment(combined_risk=0.1)
        agg._decide_interrupt(assessment)
        assert assessment.should_interrupt is False

    def test_cooldown_blocks_interrupt(self):
        config = InterruptConfig(soft_threshold=0.3, cooldown_seconds=999.0)
        agg = RiskAggregator(config=config)
        # First interrupt goes through
        a1 = RiskAssessment(combined_risk=0.5)
        agg._decide_interrupt(a1)
        assert a1.should_interrupt is True
        # Second blocked by cooldown
        a2 = RiskAssessment(combined_risk=0.5)
        agg._decide_interrupt(a2)
        assert a2.should_interrupt is False

    def test_build_message_duration_warning(self):
        agg = RiskAggregator()
        a = RiskAssessment(duration_warning="hard")
        msg = agg._build_message(a, "hard")
        assert "longer than expected" in msg

    def test_build_message_fallback(self):
        agg = RiskAggregator()
        a = RiskAssessment()
        msg = agg._build_message(a, "soft")
        assert "Possible issue" in msg

    def test_reset(self):
        agg = RiskAggregator()
        agg._interrupt_count = 5
        agg.reset()
        assert agg.interrupt_count == 0

    def test_disable_layers(self):
        config = InterruptConfig(
            use_duration=False, use_transition=False,
            use_anomaly=False, use_failure=False,
        )
        agg = RiskAggregator(config=config)
        assessment = RiskAssessment(
            duration_z=10.0, transition_entropy=5.0,
            anomaly_score=10.0, failure_risk=1.0,
        )
        result = agg._combine(assessment)
        assert result == pytest.approx(0.0)
