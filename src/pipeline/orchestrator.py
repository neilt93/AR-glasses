"""Pipeline orchestrator — wires all modules into the main inference loop."""

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.video.capture import VideoCapture, FrameData
from src.perception.detector import ObjectDetector
from src.state.scene_state import SceneStateExtractor
from src.task.step_estimator import StepEstimator
from src.instructions.engine import InstructionEngine
from src.hud.renderer import HUDRenderer
from src.logging.events import EventLogger
from src.utils.smoothing import ConfidenceBuffer


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""
    # Video
    camera_source: int | str = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30

    # Detector
    model_path: Optional[str] = None
    detection_confidence: float = 0.4
    detect_every_n: int = 1  # Run detector every N frames

    # FSM
    persistence_frames: int = 5
    stall_threshold: int = 150

    # Display
    task_name: str = "Servo Bracket Assembly"
    fullscreen: bool = False
    window_name: str = "AR Assembly Copilot"
    show_detections: bool = True

    # Logging
    log_dir: str = "logs"
    enable_logging: bool = True


class Pipeline:
    """Main pipeline that ties all modules together."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._running = False

        # Initialize modules
        self._capture = VideoCapture(
            source=self.config.camera_source,
            width=self.config.frame_width,
            height=self.config.frame_height,
            fps=self.config.target_fps,
        )
        self._detector = ObjectDetector(
            model_path=self.config.model_path,
            confidence_threshold=self.config.detection_confidence,
        )
        self._state_extractor = SceneStateExtractor()
        self._step_estimator = StepEstimator(
            persistence_frames=self.config.persistence_frames,
        )
        self._instruction_engine = InstructionEngine(
            stall_threshold_frames=self.config.stall_threshold,
        )
        self._hud = HUDRenderer(
            task_name=self.config.task_name,
            show_detections=self.config.show_detections,
        )
        self._confidence_buffer = ConfidenceBuffer(window_size=10)
        self._logger: Optional[EventLogger] = None

    def run(self):
        """Run the main pipeline loop. Press 'q' to quit."""
        if not self._capture.open():
            print("[pipeline] ERROR: Could not open camera source:", self.config.camera_source)
            return

        if self.config.enable_logging:
            self._logger = EventLogger(log_dir=self.config.log_dir)
            print(f"[pipeline] Logging to: {self._logger.filepath}")

        print(f"[pipeline] Started — press 'q' to quit")

        if self.config.fullscreen:
            cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                self.config.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )

        self._running = True
        frame_count = 0
        last_detections = []

        try:
            while self._running:
                t0 = time.time()

                # 1. Capture frame
                frame_data = self._capture.read()
                if frame_data is None:
                    break
                frame_count += 1

                # 2. Detect objects (every N frames, reuse otherwise)
                if frame_count % self.config.detect_every_n == 0:
                    last_detections = self._detector.detect(frame_data.image)

                # 3. Extract scene state
                state = self._state_extractor.extract(last_detections)

                # 4. Estimate step
                estimate = self._step_estimator.update(state)

                # 5. Generate guidance
                guidance = self._instruction_engine.generate(estimate, state)

                # 6. Render HUD
                display = self._hud.render(
                    frame_data.image,
                    last_detections,
                    estimate,
                    guidance,
                )

                inference_ms = (time.time() - t0) * 1000.0

                # 7. Log
                if self._logger is not None:
                    self._logger.log(
                        frame_number=frame_data.frame_number,
                        detections=last_detections,
                        state=state,
                        estimate=estimate,
                        guidance=guidance,
                        inference_ms=inference_ms,
                    )

                # 8. Display
                cv2.imshow(self.config.window_name, display)

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    self._step_estimator.reset()
                    print("[pipeline] FSM reset")

                # Print step changes
                if estimate.changed:
                    print(f"[step] {estimate.step_label}")

        finally:
            self._cleanup()

    def _cleanup(self):
        self._running = False
        self._capture.release()
        cv2.destroyAllWindows()
        if self._logger is not None:
            self._logger.flush()
            self._logger.close()
            print(f"[pipeline] Log saved to: {self._logger.filepath}")
        print("[pipeline] Stopped")
