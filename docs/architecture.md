# System Architecture

## Overview

```
Wearable/Table Camera
        |
        v
  Video Capture (Module A)
        |
        v
  Object Detector (Module B)   -- YOLO-based, 4-8 classes
        |
        v
  Scene State Extractor (Module C) -- rule-based spatial reasoning
        |
        v
  Task Step Estimator (Module D) -- finite-state machine
        |
        v
  Instruction Engine (Module E) -- templated guidance
        |
        v
  HUD Renderer (Module F) -- OpenCV overlay
        |
        v
  RayNeo Air 4 Display
```

## Modules

### Module A — Video Capture (`src/video/capture.py`)
- Reads frames from a webcam or video file.
- Targets 15-30 FPS.
- Resizes frames to a working resolution (e.g., 640x480).
- Timestamps each frame.

### Module B — Object Detector (`src/perception/detector.py`)
- Runs a YOLO model on each frame (or every k-th frame).
- Detects task-relevant objects: base plate, bracket, screw, screwdriver,
  servo/motor.
- Outputs: list of `Detection(class, bbox, confidence)`.

### Module C — Scene State (`src/state/scene_state.py`, `src/state/features.py`)
- Converts raw detections into structured state.
- Uses spatial rules: overlap, proximity, containment.
- Outputs a `SceneState` dict describing what is present and how objects
  relate.

### Module D — Step Estimator (`src/task/task_fsm.py`, `src/task/step_estimator.py`)
- Finite-state machine driven by scene state transitions.
- Tracks current step, previous step, and confidence.
- Requires state persistence for N frames before transitioning.

### Module E — Instruction Engine (`src/instructions/engine.py`, `src/instructions/templates.py`)
- Maps current step to a human-readable instruction.
- Generates warnings for detected errors.
- Deterministic template lookup for v1.

### Module F — HUD Renderer (`src/hud/renderer.py`)
- Draws detection boxes on the camera feed.
- Renders a top bar (task name, step, confidence).
- Renders a bottom strip (next instruction, warnings).
- Outputs a composite frame for display.

### Pipeline Orchestrator (`src/pipeline/orchestrator.py`)
- Wires all modules together in the main loop.
- Manages frame timing, inference scheduling, and HUD updates.

### Smoothing Utilities (`src/utils/smoothing.py`)
- EMA filters for detection confidences.
- Cooldown timers for step transitions.
- Flicker suppression for HUD text.

### Event Logger (`src/logging/events.py`)
- Logs per-frame data: detections, state, step, instruction, latency.
- Saves to JSONL files in `logs/`.

## Data Flow (per frame)
1. `VideoCapture.read()` → raw frame
2. `ObjectDetector.detect(frame)` → detections
3. `SceneState.update(detections)` → structured state
4. `StepEstimator.update(state)` → current step + confidence
5. `InstructionEngine.get(step, state)` → instruction text + warnings
6. `HUDRenderer.render(frame, detections, step, instruction, warnings)` → display frame
7. `EventLogger.log(...)` → disk

## Configuration
- `configs/base.yaml` — default settings.
- `configs/demo_config.yaml` — locked-down demo settings.
- `configs/detector.yaml` — model paths and thresholds.
