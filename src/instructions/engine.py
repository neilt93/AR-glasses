"""Instruction engine — generates guidance based on current step and state."""

from dataclasses import dataclass, field

from src.state.scene_state import SceneState
from src.task.task_fsm import AssemblyStep
from src.task.step_estimator import StepEstimate
from src.instructions.templates import INSTRUCTIONS, NEXT_STEP_HINTS, WARNINGS


@dataclass
class Guidance:
    """Output of the instruction engine for a single frame."""
    instruction: str
    next_hint: str
    warnings: list[str] = field(default_factory=list)


class InstructionEngine:
    """Produces human-readable guidance from step estimates and scene state."""

    def __init__(self, stall_threshold_frames: int = 150):
        self.stall_threshold = stall_threshold_frames
        self._frames_since_change = 0

    def reset(self):
        """Reset state for a new run."""
        self._frames_since_change = 0

    def generate(self, estimate: StepEstimate, state: SceneState) -> Guidance:
        """Generate guidance for the current frame."""
        step = estimate.step

        instruction = INSTRUCTIONS.get(step, "Continue working.")
        next_hint = NEXT_STEP_HINTS.get(step, "")

        if estimate.changed:
            self._frames_since_change = 0
        else:
            self._frames_since_change += 1

        warnings = self._check_warnings(step, state)
        return Guidance(
            instruction=instruction,
            next_hint=next_hint,
            warnings=warnings,
        )

    def _check_warnings(self, step: AssemblyStep, state: SceneState) -> list[str]:
        """Check for common error conditions."""
        warns = []

        # Bracket should be on plate for screw steps
        if step in (
            AssemblyStep.INSERT_FIRST_SCREW,
            AssemblyStep.TIGHTEN_FIRST_SCREW,
            AssemblyStep.INSERT_SECOND_SCREW,
            AssemblyStep.TIGHTEN_SECOND_SCREW,
        ) and not state.bracket_on_plate:
            warns.append(WARNINGS["bracket_not_on_plate"])

        # Screwdriver should be visible for tightening steps
        if step in (
            AssemblyStep.TIGHTEN_FIRST_SCREW,
            AssemblyStep.TIGHTEN_SECOND_SCREW,
        ) and not state.screwdriver_visible:
            warns.append(WARNINGS["missing_screwdriver"])

        # Stall detection
        if self._frames_since_change > self.stall_threshold:
            warns.append(WARNINGS["stalled"])

        return warns
