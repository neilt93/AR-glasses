"""Tests for the assembly task FSM (src/task/task_fsm.py)."""

import pytest
from src.task.task_fsm import AssemblyStep, TaskFSM, FSMResult
from src.state.scene_state import SceneState
from tests.conftest import make_scene_state


class TestAssemblyStepEnum:
    def test_ordering(self):
        assert AssemblyStep.NOT_STARTED < AssemblyStep.PLACE_BASE_PLATE
        assert AssemblyStep.COMPLETED == 9

    def test_all_steps_present(self):
        assert len(AssemblyStep) == 10


class TestTaskFSM:
    def test_initial_state(self):
        fsm = TaskFSM(persistence_frames=1)
        assert fsm.current_step == AssemblyStep.NOT_STARTED

    def test_auto_advance_from_not_started(self):
        """NOT_STARTED → PLACE_BASE_PLATE happens unconditionally."""
        fsm = TaskFSM(persistence_frames=1)
        result = fsm.update(SceneState())
        assert result.changed is True
        assert result.current_step == AssemblyStep.PLACE_BASE_PLATE

    def test_debouncing(self):
        """Transition only fires after persistence_frames consecutive matches."""
        fsm = TaskFSM(persistence_frames=3)
        # First: auto-advance past NOT_STARTED (takes 3 frames)
        for _ in range(3):
            fsm.update(SceneState())
        assert fsm.current_step == AssemblyStep.PLACE_BASE_PLATE

        state = make_scene_state(base_plate_visible=True)
        # Frames 1 and 2: not enough persistence
        r1 = fsm.update(state)
        assert r1.changed is False
        r2 = fsm.update(state)
        assert r2.changed is False
        # Frame 3: persistence met
        r3 = fsm.update(state)
        assert r3.changed is True
        assert r3.current_step == AssemblyStep.PICK_UP_BRACKET

    def test_debounce_resets_on_condition_fail(self):
        fsm = TaskFSM(persistence_frames=3)
        # Get to PLACE_BASE_PLATE
        for _ in range(3):
            fsm.update(SceneState())

        state_ok = make_scene_state(base_plate_visible=True)
        state_bad = SceneState()

        # 2 good frames, then 1 bad
        fsm.update(state_ok)
        fsm.update(state_ok)
        fsm.update(state_bad)  # resets candidate count
        # Need 3 more good frames now
        fsm.update(state_ok)
        fsm.update(state_ok)
        assert fsm.current_step == AssemblyStep.PLACE_BASE_PLATE
        fsm.update(state_ok)
        assert fsm.current_step == AssemblyStep.PICK_UP_BRACKET

    def test_confidence_ramps_up(self):
        fsm = TaskFSM(persistence_frames=4)
        for _ in range(4):
            fsm.update(SceneState())
        assert fsm.current_step == AssemblyStep.PLACE_BASE_PLATE

        state = make_scene_state(base_plate_visible=True)
        r1 = fsm.update(state)
        assert r1.confidence == pytest.approx(0.25)
        r2 = fsm.update(state)
        assert r2.confidence == pytest.approx(0.50)
        r3 = fsm.update(state)
        assert r3.confidence == pytest.approx(0.75)
        r4 = fsm.update(state)
        assert r4.changed is True
        assert r4.confidence == pytest.approx(1.0)

    def test_full_assembly_sequence(
        self, plate_only_state, bracket_held_state, bracket_on_plate_state,
        first_screw_inserted_state, first_screw_tightened_state,
        second_screw_inserted_state, second_screw_tightened_state, verified_state,
    ):
        """Walk through the entire 9-step assembly with persistence=1."""
        fsm = TaskFSM(persistence_frames=1)

        # NOT_STARTED → PLACE_BASE_PLATE (auto)
        r = fsm.update(SceneState())
        assert r.current_step == AssemblyStep.PLACE_BASE_PLATE

        # PLACE_BASE_PLATE → PICK_UP_BRACKET
        r = fsm.update(plate_only_state)
        assert r.current_step == AssemblyStep.PICK_UP_BRACKET

        # PICK_UP_BRACKET → ALIGN_BRACKET
        r = fsm.update(bracket_held_state)
        assert r.current_step == AssemblyStep.ALIGN_BRACKET

        # ALIGN_BRACKET → INSERT_FIRST_SCREW
        r = fsm.update(bracket_on_plate_state)
        assert r.current_step == AssemblyStep.INSERT_FIRST_SCREW

        # INSERT_FIRST_SCREW → TIGHTEN_FIRST_SCREW
        r = fsm.update(first_screw_inserted_state)
        assert r.current_step == AssemblyStep.TIGHTEN_FIRST_SCREW

        # TIGHTEN_FIRST_SCREW → INSERT_SECOND_SCREW
        r = fsm.update(first_screw_tightened_state)
        assert r.current_step == AssemblyStep.INSERT_SECOND_SCREW

        # INSERT_SECOND_SCREW → TIGHTEN_SECOND_SCREW
        r = fsm.update(second_screw_inserted_state)
        assert r.current_step == AssemblyStep.TIGHTEN_SECOND_SCREW

        # TIGHTEN_SECOND_SCREW → VERIFY_ASSEMBLY
        r = fsm.update(second_screw_tightened_state)
        assert r.current_step == AssemblyStep.VERIFY_ASSEMBLY

        # VERIFY_ASSEMBLY → COMPLETED
        r = fsm.update(verified_state)
        assert r.current_step == AssemblyStep.COMPLETED

    def test_no_transition_at_completed(self, verified_state):
        """Once COMPLETED, no further transitions happen."""
        fsm = TaskFSM(persistence_frames=1)
        fsm.current_step = AssemblyStep.COMPLETED
        r = fsm.update(verified_state)
        assert r.changed is False
        assert r.current_step == AssemblyStep.COMPLETED

    def test_reset(self):
        fsm = TaskFSM(persistence_frames=1)
        fsm.current_step = AssemblyStep.TIGHTEN_FIRST_SCREW
        fsm.reset()
        assert fsm.current_step == AssemblyStep.NOT_STARTED

    def test_persistence_frames_zero(self):
        """persistence_frames=0 should advance immediately."""
        fsm = TaskFSM(persistence_frames=0)
        # With 0 persistence, every condition match advances immediately
        # But the logic requires _candidate_count >= persistence_frames (0)
        # and _candidate_step is set, so it should advance
        r = fsm.update(SceneState())
        assert r.current_step == AssemblyStep.PLACE_BASE_PLATE
