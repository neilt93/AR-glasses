"""Instruction templates for each assembly step."""

from src.task.task_fsm import AssemblyStep


INSTRUCTIONS: dict[AssemblyStep, str] = {
    AssemblyStep.NOT_STARTED: "Ready to begin. Place the base plate on the table.",
    AssemblyStep.PLACE_BASE_PLATE: "Place the base plate flat on the work surface.",
    AssemblyStep.PICK_UP_BRACKET: "Pick up the servo bracket from the parts area.",
    AssemblyStep.ALIGN_BRACKET: "Align the bracket holes with the base plate holes.",
    AssemblyStep.INSERT_FIRST_SCREW: "Insert a screw through the first bracket hole.",
    AssemblyStep.TIGHTEN_FIRST_SCREW: "Use the screwdriver to tighten the first screw.",
    AssemblyStep.INSERT_SECOND_SCREW: "Insert a screw through the second bracket hole.",
    AssemblyStep.TIGHTEN_SECOND_SCREW: "Tighten the second screw with the screwdriver.",
    AssemblyStep.VERIFY_ASSEMBLY: "Check that the bracket is firmly attached.",
    AssemblyStep.COMPLETED: "Assembly complete! All steps finished.",
}

NEXT_STEP_HINTS: dict[AssemblyStep, str] = {
    AssemblyStep.NOT_STARTED: "First: place the base plate.",
    AssemblyStep.PLACE_BASE_PLATE: "Next: pick up the bracket.",
    AssemblyStep.PICK_UP_BRACKET: "Next: align bracket to base plate.",
    AssemblyStep.ALIGN_BRACKET: "Next: insert the first screw.",
    AssemblyStep.INSERT_FIRST_SCREW: "Next: tighten with screwdriver.",
    AssemblyStep.TIGHTEN_FIRST_SCREW: "Next: insert the second screw.",
    AssemblyStep.INSERT_SECOND_SCREW: "Next: tighten the second screw.",
    AssemblyStep.TIGHTEN_SECOND_SCREW: "Next: verify the assembly.",
    AssemblyStep.VERIFY_ASSEMBLY: "Almost done — verify and finish.",
    AssemblyStep.COMPLETED: "",
}

WARNINGS: dict[str, str] = {
    "bracket_not_on_plate": "Warning: bracket is not on the base plate.",
    "missing_screwdriver": "Warning: screwdriver not detected.",
    "stalled": "Hint: no progress detected for a while.",
    "wrong_orientation": "Warning: bracket may be upside down.",
}
