"""Finite-state machine for tracking assembly task progress."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from src.state.scene_state import SceneState


class AssemblyStep(IntEnum):
    """Steps in the servo bracket assembly task."""
    NOT_STARTED = 0
    PLACE_BASE_PLATE = 1
    PICK_UP_BRACKET = 2
    ALIGN_BRACKET = 3
    INSERT_FIRST_SCREW = 4
    TIGHTEN_FIRST_SCREW = 5
    INSERT_SECOND_SCREW = 6
    TIGHTEN_SECOND_SCREW = 7
    VERIFY_ASSEMBLY = 8
    COMPLETED = 9


# Transition rules: (current_step) -> list of (condition_fn, next_step)
# Each condition_fn takes a SceneState and returns bool.

def _can_start(s: SceneState) -> bool:
    return True

def _plate_placed(s: SceneState) -> bool:
    return s.base_plate_visible

def _bracket_picked(s: SceneState) -> bool:
    return s.bracket_visible and s.hand_visible

def _bracket_aligned(s: SceneState) -> bool:
    return s.bracket_on_plate

def _first_screw_inserted(s: SceneState) -> bool:
    return s.bracket_on_plate and s.screw_near_plate and s.screws_visible >= 1

def _first_screw_tightened(s: SceneState) -> bool:
    return s.bracket_on_plate and s.screwdriver_engaged

def _second_screw_inserted(s: SceneState) -> bool:
    return s.bracket_on_plate and s.screws_visible >= 2

def _second_screw_tightened(s: SceneState) -> bool:
    return s.bracket_on_plate and s.screwdriver_engaged and s.screws_visible >= 2

def _assembly_verified(s: SceneState) -> bool:
    return s.bracket_on_plate and s.screws_visible >= 2 and not s.screwdriver_engaged


TRANSITIONS: dict[AssemblyStep, list[tuple]] = {
    AssemblyStep.NOT_STARTED: [(_can_start, AssemblyStep.PLACE_BASE_PLATE)],
    AssemblyStep.PLACE_BASE_PLATE: [(_plate_placed, AssemblyStep.PICK_UP_BRACKET)],
    AssemblyStep.PICK_UP_BRACKET: [(_bracket_picked, AssemblyStep.ALIGN_BRACKET)],
    AssemblyStep.ALIGN_BRACKET: [(_bracket_aligned, AssemblyStep.INSERT_FIRST_SCREW)],
    AssemblyStep.INSERT_FIRST_SCREW: [(_first_screw_inserted, AssemblyStep.TIGHTEN_FIRST_SCREW)],
    AssemblyStep.TIGHTEN_FIRST_SCREW: [(_first_screw_tightened, AssemblyStep.INSERT_SECOND_SCREW)],
    AssemblyStep.INSERT_SECOND_SCREW: [(_second_screw_inserted, AssemblyStep.TIGHTEN_SECOND_SCREW)],
    AssemblyStep.TIGHTEN_SECOND_SCREW: [(_second_screw_tightened, AssemblyStep.VERIFY_ASSEMBLY)],
    AssemblyStep.VERIFY_ASSEMBLY: [(_assembly_verified, AssemblyStep.COMPLETED)],
}


@dataclass
class FSMResult:
    """Result of an FSM update."""
    current_step: AssemblyStep
    changed: bool
    confidence: float


class TaskFSM:
    """Finite-state machine that tracks which assembly step the user is on."""

    def __init__(self, persistence_frames: int = 5):
        self.current_step = AssemblyStep.NOT_STARTED
        self.persistence_frames = persistence_frames
        self._candidate_step: Optional[AssemblyStep] = None
        self._candidate_count = 0

    def update(self, state: SceneState) -> FSMResult:
        """Evaluate transitions and potentially advance the step.

        A transition only fires after the condition is met for
        `persistence_frames` consecutive frames (debouncing).
        """
        transitions = TRANSITIONS.get(self.current_step, [])
        next_step = None
        for condition_fn, target_step in transitions:
            if condition_fn(state):
                next_step = target_step
                break

        if next_step is not None and next_step == self._candidate_step:
            self._candidate_count += 1
        elif next_step is not None:
            self._candidate_step = next_step
            self._candidate_count = 1
        else:
            self._candidate_step = None
            self._candidate_count = 0

        changed = False
        if (
            self._candidate_step is not None
            and self._candidate_count >= self.persistence_frames
        ):
            self.current_step = self._candidate_step
            self._candidate_step = None
            self._candidate_count = 0
            changed = True

        if changed:
            confidence = 1.0
        elif self.persistence_frames > 0:
            confidence = min(1.0, self._candidate_count / self.persistence_frames)
        else:
            confidence = 1.0
        return FSMResult(
            current_step=self.current_step,
            changed=changed,
            confidence=confidence,
        )

    def reset(self):
        """Reset the FSM to the initial state."""
        self.current_step = AssemblyStep.NOT_STARTED
        self._candidate_step = None
        self._candidate_count = 0
