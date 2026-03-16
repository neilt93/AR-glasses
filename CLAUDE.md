# AR Assembly Copilot

Real-time AR guidance system for servo bracket assembly tasks. Uses computer vision (YOLO) to detect objects, track task progress via FSM, and render live instructions on RayNeo Air 4 AR glasses.

## Quick Reference

```bash
# Install
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Run main app
python src/app.py --config configs/base.yaml --camera 0

# Run demo (locked settings)
python scripts/run_demo.py

# Collect training data
python scripts/collect_data.py --camera 0 --output data/raw

# Train learning models from run data
python scripts/train_models.py --runs-dir data/runs --model-dir models/learned

# Replay a session log
python scripts/replay_log.py --log logs/<session>.jsonl
```

Runtime keys: `q` quit, `r` reset task, `s` mark success, `f` mark failure.

## Architecture

Per-frame pipeline: **VideoCapture → ObjectDetector → SceneStateExtractor → TaskFSM/StepEstimator → InstructionEngine → LearningLayers → HUDRenderer → EventLogger**

Each stage lives in its own `src/` submodule. `Pipeline` in `src/pipeline/orchestrator.py` ties them together.

### Key Modules

| Module | Purpose |
|--------|---------|
| `src/perception/detector.py` | YOLO object detection (mock fallback when unavailable) |
| `src/state/scene_state.py` | Spatial rules: overlap, proximity, containment |
| `src/task/task_fsm.py` | 9-step FSM with debounced transitions |
| `src/instructions/engine.py` | Template-based guidance + error detection |
| `src/learning/` | 5-layer risk system: duration, transition, anomaly, failure, risk aggregator |
| `src/hud/renderer.py` | OpenCV HUD overlay with color palettes |
| `src/data/records.py` | RunRecord/FrameRecord dataclasses, RunStore persistence |
| `src/event_log/events.py` | JSONL session logging |

## Code Conventions

- **Python 3.10+** — uses `int | str` union syntax, walrus operator
- **Dataclass-heavy** — all data structures use `@dataclass`
- **PascalCase** classes, **snake_case** functions, **UPPER_SNAKE_CASE** constants, **`_prefix`** private attrs
- **Config-driven** — `PipelineConfig` dataclass loaded from YAML (`configs/base.yaml`), CLI overrides
- **No test suite currently** — validate changes by running the demo or replaying logs

## Configuration

- `configs/base.yaml` — all pipeline parameters with defaults
- `configs/demo_config.yaml` — locked demo settings (fullscreen, higher thresholds)
- `configs/detector.yaml` — model paths (optional)

Key config areas: video (camera, FPS, resolution), detector (confidence, model path), FSM (persistence_frames, stall_threshold), display, logging, learning (enable, user_id, model_dir).

## Dependencies

Core: `opencv-python`, `numpy`, `pyyaml`. Optional: `ultralytics` (YOLO), `scikit-learn` (failure classifier), `torch`/`torchvision` (training).
