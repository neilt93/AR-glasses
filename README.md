# AR Smart Assistant

A real-time scene-understanding assistant for RayNeo Air 4 Pro glasses. A PC
runs the perception pipeline on a webcam feed and streams an annotated HUD out
to the glasses over USB-C (1920x1080). The goal is Vision Pro-style spatial
awareness on inexpensive hobbyist hardware.

## What it does

- **Object detection** with YOLOv8 (COCO 80 classes), plus a persistent-ID
  object tracker (IoU + centroid matching, smoothed boxes).
- **Hand tracking** with MediaPipe: 21 landmarks, several gesture classes.
- **Monocular depth** with MiDaS for coarse spatial reasoning and depth sorting.
- **Spatial reasoning**: near / on-top / inside / holding / left-right relations.
- **Proactive suggestions + voice**: offline TTS (pyttsx3) for notifications
  such as low light, leaving the workspace, or a focus-mode nudge.
- **Composable HUD**: object labels, hand skeleton, depth overlay, status bar,
  crosshair, info panel, rendered to screen or to the glasses.

## Run

```bash
pip install -r requirements.txt

python -m src.assistant.app                  # webcam -> screen
python -m src.assistant.app --glasses        # webcam -> AR glasses
python -m src.assistant.app --model yolov8s  # more accurate detector
python -m src.assistant.app --voice --hands --depth   # enable optional stages
```

Keys while running: `q` quit, `h` labels, `c` crosshair, `i` info,
`k` hands, `d` depth overlay.

Optional stages need extra packages (kept out of the core install):
`pyttsx3` (voice), `mediapipe` (hands), `torch`/`torchvision` (depth/MiDaS),
`pytesseract` (OCR). See `requirements.txt`.

```bash
python -m pytest tests/ -v
```

## Layout

| Path | Purpose |
|------|---------|
| `src/assistant/` | Decision layer: scene state, notifications, voice, suggestions |
| `src/perception/` | YOLO scene, tracker, MediaPipe hands, MiDaS depth |
| `src/hud/` | Composable HUD widgets and renderer |
| `src/display/` | Windows monitor detection, RayNeo output |
| `scripts/` | Demo, glasses/scene test harnesses, data collection |

## Limitations

- V1 is tethered: a PC does the compute, the glasses are a display, not
  standalone hardware.
- Depth is monocular MiDaS, so it is relative and approximate, not metric.
- Detection is limited to the pretrained COCO classes.
- Display output and RayNeo auto-detect are geared to Windows.
- Python 3.10+ (uses `int | str` unions and the walrus operator).
