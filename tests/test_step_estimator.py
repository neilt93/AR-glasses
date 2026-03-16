"""Tests for the step estimator (src/task/step_estimator.py)."""

import pytest
from src.task.step_estimator import StepEstimator, StepEstimate, STEP_LABELS, TOTAL_STEPS
from src.task.task_fsm import AssemblyStep
from src.state.scene_state import SceneState
from tests.conftest import make_scene_state


class TestStepEstimator:
    def test_initial_estimate(self):
        est = StepEstimator(persistence_frames=1)
        # First update auto-advances past NOT_STARTED
        result = est.update(SceneState())
        assert result.step == AssemblyStep.PLACE_BASE_PLATE
        assert result.changed is True
        assert result.step_label == "Step 1: Place base plate"

    def test_progress_percentage(self):
        est = StepEstimator(persistence_frames=1)
        est.update(SceneState())  # → PLACE_BASE_PLATE (step 1)
        result = est.update(make_scene_state(base_plate_visible=True))  # → PICK_UP_BRACKET (step 2)
        assert result.step_number == 2
        assert result.progress_pct == pytest.approx(25.0)

    def test_completed_is_100_percent(self):
        est = StepEstimator(persistence_frames=1)
        est._fsm.current_step = AssemblyStep.VERIFY_ASSEMBLY
        state = make_scene_state(
            base_plate_visible=True, bracket_visible=True, bracket_on_plate=True,
            screws_visible=2, screwdriver_engaged=False,
        )
        result = est.update(state)
        assert result.step == AssemblyStep.COMPLETED
        # step 9 / 8 total = 112.5 → clamped to 100
        assert result.progress_pct == pytest.approx(100.0)

    def test_total_steps_constant(self):
        assert TOTAL_STEPS == 8

    def test_all_steps_have_labels(self):
        for step in AssemblyStep:
            assert step in STEP_LABELS

    def test_reset(self):
        est = StepEstimator(persistence_frames=1)
        est.update(SceneState())  # advance
        est.reset()
        assert est._fsm.current_step == AssemblyStep.NOT_STARTED
