# AR Smart Assistant

Smart AR assistant for RayNeo Air 4 Pro glasses. Goal: recreate Apple Vision Pro at a fraction of the cost. Hobbyist stage now, industry later.

**V1 setup:** PC + webcam (input) → processing → RayNeo Air 4 Pro glasses (USB-C display output, 1920x1080)

## Quick Reference

```bash
# Install
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# ── Smart Assistant (main app) ──
python -m src.assistant.app                  # webcam → screen
python -m src.assistant.app --glasses        # webcam → AR glasses
python -m src.assistant.app --model yolov8s  # more accurate model
python -m src.assistant.app --ocr            # enable text detection
python -m src.assistant.app --voice          # enable spoken notifications
python -m src.assistant.app --hands          # enable hand tracking
python -m src.assistant.app --no-tracking    # disable object tracking

# ── Display testing ──
python scripts/test_glasses.py               # test pattern on glasses
python scripts/test_scene.py                 # scene understanding demo

# ── Assembly copilot (original app) ──
python src/app.py --config configs/base.yaml --camera 0
python src/app.py --glasses                  # output to glasses
python scripts/run_demo.py
```

Assistant keys: `q` quit, `h` toggle labels, `c` crosshair, `i` info panel, `k` hand skeleton (if --hands).

## Architecture

### Smart Assistant Pipeline
**Webcam → SceneUnderstanding (YOLO COCO) → ObjectTracker → HandDetector → AssistantBrain → WidgetRenderer → VoiceEngine → GlassesDisplay**

### Key Modules

| Module | Purpose |
|--------|---------|
| `src/assistant/brain.py` | Decision layer: tracks scene, triggers notifications + voice |
| `src/assistant/voice.py` | Async TTS engine (pyttsx3, background thread, dedup) |
| `src/assistant/app.py` | Main assistant entry point |
| `src/perception/scene.py` | General scene understanding: YOLO 80-class + OCR + scene tags |
| `src/perception/tracker.py` | Object tracker: persistent IDs, IoU+centroid matching, smooth bbox |
| `src/perception/hands.py` | Hand pose detection (MediaPipe): 7 gestures, 21 landmarks |
| `src/perception/detector.py` | Assembly-specific YOLO detector (mock fallback) |
| `src/hud/widgets.py` | Composable widget HUD: ObjectLabels, TrackedObjectLabels, HandSkeleton, StatusBar, Notifications, InfoPanel, Crosshair |
| `src/hud/renderer.py` | Legacy assembly-specific HUD |
| `src/display/monitor.py` | Windows monitor detection, RayNeo auto-detect |
| `src/display/output.py` | GlassesDisplay: fullscreen output on AR glasses |
| `src/pipeline/orchestrator.py` | Assembly copilot pipeline (with --glasses support) |
| `src/learning/` | 5-layer risk system (assembly-specific) |

## Code Conventions

- **Python 3.10+** — uses `int | str` union syntax, walrus operator
- **Dataclass-heavy** — all data structures use `@dataclass`
- **PascalCase** classes, **snake_case** functions, **UPPER_SNAKE_CASE** constants, **`_prefix`** private attrs
- **247 tests** — `python -m pytest tests/ -v`

## Dependencies

Core: `opencv-python`, `numpy`, `pyyaml`, `ultralytics` (YOLOv8). Optional: `pyttsx3` (voice), `mediapipe` (hands), `pytesseract` (OCR), `scikit-learn` (failure classifier).
