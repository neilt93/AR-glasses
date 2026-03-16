"""Step estimator — wraps the FSM and provides higher-level step info."""

from dataclasses import dataclass

from src.state.scene_state import SceneState
from src.task.task_fsm import AssemblyStep, TaskFSM, FSMResult


STEP_LABELS = {
    AssemblyStep.NOT_STARTED: "Not started",
    AssemblyStep.PLACE_BASE_PLATE: "Step 1: Place base plate",
    AssemblyStep.PICK_UP_BRACKET: "Step 2: Pick up bracket",
    AssemblyStep.ALIGN_BRACKET: "Step 3: Align bracket",
    AssemblyStep.INSERT_FIRST_SCREW: "Step 4: Insert first screw",
    AssemblyStep.TIGHTEN_FIRST_SCREW: "Step 5: Tighten first screw",
    AssemblyStep.INSERT_SECOND_SCREW: "Step 6: Insert second screw",
    AssemblyStep.TIGHTEN_SECOND_SCREW: "Step 7: Tighten second screw",
    AssemblyStep.VERIFY_ASSEMBLY: "Step 8: Verify assembly",
    AssemblyStep.COMPLETED: "Completed!",
}

TOTAL_STEPS = 8  # Excludes NOT_STARTED and COMPLETED


@dataclass
class StepEstimate:
    """High-level step estimation output."""
    step: AssemblyStep
    step_label: str
    step_number: int
    total_steps: int
    changed: bool
    confidence: float
    progress_pct: float


class StepEstimator:
    """Estimates the current assembly step from scene state."""

    def __init__(self, persistence_frames: int = 5):
        self._fsm = TaskFSM(persistence_frames=persistence_frames)

    def update(self, state: SceneState) -> StepEstimate:
        result: FSMResult = self._fsm.update(state)
        step = result.current_step
        step_num = int(step)
        progress = min(100.0, (step_num / TOTAL_STEPS) * 100.0)

        return StepEstimate(
            step=step,
            step_label=STEP_LABELS.get(step, "Unknown"),
            step_number=step_num,
            total_steps=TOTAL_STEPS,
            changed=result.changed,
            confidence=result.confidence,
            progress_pct=progress,
        )

    def reset(self):
        self._fsm.reset()
