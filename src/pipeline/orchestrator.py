"""Pipeline orchestrator — wires all modules into the main inference loop.

Integrates perception, task tracking, instruction generation, and all
five learning layers into a single real-time loop.
"""

import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np

from src.video.capture import VideoCapture, FrameData
from src.perception.detector import ObjectDetector
from src.state.scene_state import SceneStateExtractor
from src.task.step_estimator import StepEstimator
from src.instructions.engine import InstructionEngine
from src.hud.renderer import HUDRenderer
from src.event_log.events import EventLogger
from src.data.records import RunRecord, FrameRecord, RunStore
from src.learning.duration_model import StepDurationModel
from src.learning.transition_model import TransitionModel
from src.learning.anomaly_detector import AnomalyDetector
from src.learning.failure_classifier import FailureClassifier
from src.learning.risk_aggregator import RiskAggregator, InterruptConfig
from src.learning.online_updater import OnlineUpdater


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
    detect_every_n: int = 1

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

    # Learning
    enable_learning: bool = True
    user_id: str = "default"
    run_store_dir: str = "data/runs"
    model_dir: str = "models/learned"
    recent_frames_buffer: int = 60  # frames to keep for failure classifier

    # Interrupt
    soft_threshold: float = 0.4
    hard_threshold: float = 0.7
    interrupt_cooldown: float = 3.0


class Pipeline:
    """Main pipeline that ties all modules together, including learning layers."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._running = False

        # Core modules
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
        self._logger: Optional[EventLogger] = None

        # Learning layers
        self._learning_enabled = self.config.enable_learning
        if self._learning_enabled:
            self._run_store = RunStore(store_dir=self.config.run_store_dir)
            self._duration_model = StepDurationModel()
            self._transition_model = TransitionModel(num_steps=10)
            self._anomaly_detector = AnomalyDetector(method="knn")
            self._failure_classifier = FailureClassifier()
            self._risk_aggregator = RiskAggregator(InterruptConfig(
                soft_threshold=self.config.soft_threshold,
                hard_threshold=self.config.hard_threshold,
                cooldown_seconds=self.config.interrupt_cooldown,
            ))
            self._online_updater = OnlineUpdater(
                duration_model=self._duration_model,
                transition_model=self._transition_model,
                anomaly_detector=self._anomaly_detector,
                failure_classifier=self._failure_classifier,
                run_store=self._run_store,
                model_dir=self.config.model_dir,
            )
            # Try to load existing model state
            self._online_updater.load_models()
            # Fit from any stored runs
            self._online_updater.initial_fit()

        # Run-level tracking
        self._current_run: Optional[RunRecord] = None
        self._recent_frames: deque = deque(maxlen=self.config.recent_frames_buffer)
        self._step_regressions = 0
        self._prev_step = 0
        self._current_step_frames = 0
        self._step_frame_counter: dict[int, int] = {}

    def run(self):
        """Run the main pipeline loop. Press 'q' to quit, 'r' to reset, 's'/'f' to mark outcome."""
        if not self._capture.open():
            print("[pipeline] ERROR: Could not open camera source:", self.config.camera_source)
            return

        if self.config.enable_logging:
            self._logger = EventLogger(log_dir=self.config.log_dir)
            print(f"[pipeline] Logging to: {self._logger.filepath}")

        print("[pipeline] Started — press 'q' to quit, 'r' to reset")
        if self._learning_enabled:
            print("[pipeline] Learning enabled — press 's' for success, 'f' for failure")
            run_count = self._run_store.count()
            print(f"[pipeline] {run_count} prior runs in store")

        if self.config.fullscreen:
            cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                self.config.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )

        # Start a new run
        self._start_run()

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

                # 2. Detect objects
                if frame_count % self.config.detect_every_n == 0:
                    last_detections = self._detector.detect(frame_data.image)

                # 3. Extract scene state
                state = self._state_extractor.extract(last_detections)

                # 4. Estimate step
                estimate = self._step_estimator.update(state)

                # Track step regressions and current step duration
                step_num = estimate.step_number
                if step_num != self._prev_step:
                    if step_num < self._prev_step:
                        self._step_regressions += 1
                    # Reset current-step frame counter on any step change
                    self._current_step_frames = 0
                    self._prev_step = step_num
                self._current_step_frames += 1
                current_step_duration = float(self._current_step_frames)

                # Also track total frames per step (for run record)
                self._step_frame_counter[step_num] = \
                    self._step_frame_counter.get(step_num, 0) + 1

                # 5. Generate base guidance
                guidance = self._instruction_engine.generate(estimate, state)

                # 6. Learning layers (risk assessment)
                risk_message = ""
                anomaly_score = 0.0
                duration_z = 0.0
                failure_risk = 0.0

                if self._learning_enabled:
                    # Update belief state
                    self._transition_model.update_belief(
                        estimate.step_number, estimate.confidence
                    )

                    state_dict = state.to_dict()
                    recent_dicts = [
                        {"scene_state": f.scene_state, "step_estimate": f.step_estimate}
                        for f in self._recent_frames
                    ]

                    assessment = self._risk_aggregator.assess(
                        duration_model=self._duration_model,
                        transition_model=self._transition_model,
                        anomaly_detector=self._anomaly_detector,
                        failure_classifier=self._failure_classifier,
                        current_step=step_num,
                        current_duration=current_step_duration,
                        scene_state=state_dict,
                        recent_frames=recent_dicts,
                        step_regressions=self._step_regressions,
                        user_id=self.config.user_id,
                    )

                    anomaly_score = assessment.anomaly_score
                    duration_z = assessment.duration_z
                    failure_risk = assessment.failure_risk

                    if assessment.should_interrupt:
                        risk_message = assessment.interrupt_message
                        if assessment.interrupt_type == "hard":
                            guidance.warnings.insert(0, risk_message)
                        else:
                            guidance.warnings.append(risk_message)

                # 7. Record frame for learning
                inference_ms = (time.time() - t0) * 1000.0

                if self._learning_enabled and self._current_run is not None:
                    frame_record = FrameRecord(
                        t=frame_count,
                        timestamp=time.time(),
                        scene_state=state.to_dict(),
                        step_estimate=step_num,
                        step_confidence=estimate.confidence,
                        detections=[
                            {"class": d.class_name, "bbox": list(d.bbox),
                             "conf": round(d.confidence, 3)}
                            for d in last_detections
                        ],
                        hand_positions={},
                        anomaly_score=anomaly_score,
                        duration_z_score=duration_z,
                        failure_risk=failure_risk,
                        inference_ms=inference_ms,
                    )
                    self._current_run.add_frame(frame_record)
                    self._recent_frames.append(frame_record)

                # 8. Render HUD
                display = self._hud.render(
                    frame_data.image,
                    last_detections,
                    estimate,
                    guidance,
                )

                # Draw risk indicator if learning is active
                if self._learning_enabled and failure_risk > 0.1:
                    self._draw_risk_indicator(display, failure_risk)

                # 9. Log
                if self._logger is not None:
                    self._logger.log(
                        frame_number=frame_data.frame_number,
                        detections=last_detections,
                        state=state,
                        estimate=estimate,
                        guidance=guidance,
                        inference_ms=inference_ms,
                    )

                # 10. Display
                cv2.imshow(self.config.window_name, display)

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self._finish_run("abandoned")
                    break
                elif key == ord("r"):
                    self._finish_run("abandoned")
                    self._step_estimator.reset()
                    self._start_run()
                    frame_count = 0
                    print("[pipeline] Reset — new run started")
                elif key == ord("s"):
                    self._finish_run("success")
                    self._step_estimator.reset()
                    self._start_run()
                    frame_count = 0
                    print("[pipeline] Marked SUCCESS — new run started")
                elif key == ord("f"):
                    self._finish_run("failure", failure_step=step_num)
                    self._step_estimator.reset()
                    frame_count = 0
                    self._start_run()
                    print("[pipeline] Marked FAILURE — new run started")

                if estimate.changed:
                    print(f"[step] {estimate.step_label}")

        finally:
            self._cleanup()

    def _start_run(self):
        """Begin tracking a new run."""
        if self._learning_enabled:
            self._current_run = RunRecord(
                user_id=self.config.user_id,
                task_id="servo_bracket_assembly",
            )
            self._transition_model.reset_belief()
            self._risk_aggregator.reset()
        self._instruction_engine.reset()
        self._recent_frames.clear()
        self._step_regressions = 0
        self._prev_step = 0
        self._current_step_frames = 0
        self._step_frame_counter.clear()

    def _finish_run(self, outcome: str, failure_step: Optional[int] = None):
        """Finalize the current run and trigger online updates."""
        if not self._learning_enabled or self._current_run is None:
            return
        self._current_run.finish(outcome, failure_step=failure_step)
        print(f"[pipeline] Run {self._current_run.run_id} finished: {outcome} "
              f"({self._current_run.total_frames} frames)")
        self._online_updater.on_run_complete(self._current_run)
        self._current_run = None

    def _draw_risk_indicator(self, frame: np.ndarray, risk: float):
        """Draw a small risk gauge on the HUD."""
        h, w = frame.shape[:2]
        # Risk bar on the right edge
        bar_x = w - 25
        bar_h = 100
        bar_y = h // 2 - bar_h // 2
        fill_h = int(bar_h * min(1.0, risk))

        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 15, bar_y + bar_h), (40, 40, 40), -1)
        # Fill (green -> yellow -> red)
        if risk < 0.4:
            color = (0, 200, 0)
        elif risk < 0.7:
            color = (0, 200, 200)
        else:
            color = (0, 0, 220)
        cv2.rectangle(
            frame,
            (bar_x, bar_y + bar_h - fill_h),
            (bar_x + 15, bar_y + bar_h),
            color, -1,
        )
        # Label
        cv2.putText(
            frame, f"{risk:.0%}",
            (bar_x - 5, bar_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1,
        )

    def _cleanup(self):
        self._running = False
        self._capture.release()
        cv2.destroyAllWindows()
        if self._logger is not None:
            self._logger.flush()
            self._logger.close()
            print(f"[pipeline] Log saved to: {self._logger.filepath}")
        print("[pipeline] Stopped")
