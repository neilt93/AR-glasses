"""Tests for the instruction engine (src/instructions/engine.py)."""

import pytest
from src.instructions.engine import InstructionEngine, Guidance
from src.instructions.templates import INSTRUCTIONS, WARNINGS
from src.task.task_fsm import AssemblyStep
from src.task.step_estimator import StepEstimate
from src.state.scene_state import SceneState
from tests.conftest import make_scene_state


def _make_estimate(step: AssemblyStep, changed: bool = False) -> StepEstimate:
    return StepEstimate(
        step=step,
        step_label="",
        step_number=int(step),
        total_steps=8,
        changed=changed,
        confidence=1.0,
        progress_pct=0.0,
    )


class TestInstructionEngine:
    def test_basic_instruction(self):
        engine = InstructionEngine()
        est = _make_estimate(AssemblyStep.PLACE_BASE_PLATE, changed=True)
        guidance = engine.generate(est, SceneState())
        assert guidance.instruction == INSTRUCTIONS[AssemblyStep.PLACE_BASE_PLATE]

    def test_warning_bracket_not_on_plate(self):
        engine = InstructionEngine()
        est = _make_estimate(AssemblyStep.INSERT_FIRST_SCREW)
        state = make_scene_state(bracket_on_plate=False)
        guidance = engine.generate(est, state)
        assert WARNINGS["bracket_not_on_plate"] in guidance.warnings

    def test_no_warning_when_bracket_on_plate(self):
        engine = InstructionEngine()
        est = _make_estimate(AssemblyStep.INSERT_FIRST_SCREW)
        state = make_scene_state(bracket_on_plate=True)
        guidance = engine.generate(est, state)
        assert WARNINGS["bracket_not_on_plate"] not in guidance.warnings

    def test_warning_missing_screwdriver(self):
        engine = InstructionEngine()
        est = _make_estimate(AssemblyStep.TIGHTEN_FIRST_SCREW)
        state = make_scene_state(bracket_on_plate=True, screwdriver_visible=False)
        guidance = engine.generate(est, state)
        assert WARNINGS["missing_screwdriver"] in guidance.warnings

    def test_stall_warning(self):
        engine = InstructionEngine(stall_threshold_frames=3)
        est = _make_estimate(AssemblyStep.ALIGN_BRACKET, changed=False)
        state = SceneState()
        # Feed 4 frames without a step change → stall
        for _ in range(4):
            guidance = engine.generate(est, state)
        assert WARNINGS["stalled"] in guidance.warnings

    def test_stall_resets_on_step_change(self):
        engine = InstructionEngine(stall_threshold_frames=3)
        est_no_change = _make_estimate(AssemblyStep.ALIGN_BRACKET, changed=False)
        est_changed = _make_estimate(AssemblyStep.INSERT_FIRST_SCREW, changed=True)
        state = SceneState()

        # Build up stall counter
        for _ in range(2):
            engine.generate(est_no_change, state)

        # Step change resets counter
        engine.generate(est_changed, state)

        # 3 more frames without change — should NOT stall yet (counter reset)
        for _ in range(3):
            guidance = engine.generate(est_no_change, state)
        assert WARNINGS["stalled"] not in guidance.warnings

    def test_multiple_warnings(self):
        engine = InstructionEngine()
        est = _make_estimate(AssemblyStep.TIGHTEN_FIRST_SCREW)
        state = make_scene_state(bracket_on_plate=False, screwdriver_visible=False)
        guidance = engine.generate(est, state)
        assert WARNINGS["bracket_not_on_plate"] in guidance.warnings
        assert WARNINGS["missing_screwdriver"] in guidance.warnings

    def test_reset(self):
        engine = InstructionEngine(stall_threshold_frames=2)
        est = _make_estimate(AssemblyStep.ALIGN_BRACKET, changed=False)
        for _ in range(3):
            engine.generate(est, SceneState())
        engine.reset()
        guidance = engine.generate(est, SceneState())
        assert WARNINGS["stalled"] not in guidance.warnings
