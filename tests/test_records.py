"""Tests for data records and RunStore (src/data/records.py)."""

import json
import os
import tempfile
import pytest

from src.data.records import FrameRecord, RunRecord, RunStore


class TestFrameRecord:
    def test_defaults(self):
        fr = FrameRecord(t=0, timestamp=1.0, scene_state={}, step_estimate=1,
                         step_confidence=0.9, detections=[], hand_positions={})
        assert fr.anomaly_score == 0.0
        assert fr.inference_ms == 0.0


class TestRunRecord:
    def test_add_frame(self):
        run = RunRecord()
        fr = FrameRecord(t=0, timestamp=1.0, scene_state={}, step_estimate=1,
                         step_confidence=0.9, detections=[], hand_positions={})
        run.add_frame(fr)
        assert run.total_frames == 1
        assert len(run.frames) == 1

    def test_finish_success(self):
        run = RunRecord()
        run.add_frame(FrameRecord(t=0, timestamp=1.0, scene_state={}, step_estimate=1,
                                  step_confidence=1.0, detections=[], hand_positions={}))
        run.add_frame(FrameRecord(t=1, timestamp=2.0, scene_state={}, step_estimate=2,
                                  step_confidence=1.0, detections=[], hand_positions={}))
        run.finish("success")
        assert run.outcome == "success"
        assert run.end_time is not None
        assert run.failure_step is None

    def test_finish_failure_records_step(self):
        run = RunRecord()
        run.add_frame(FrameRecord(t=0, timestamp=1.0, scene_state={}, step_estimate=3,
                                  step_confidence=1.0, detections=[], hand_positions={}))
        run.finish("failure", failure_step=3)
        assert run.failure_step == 3
        assert run.failure_frame == 0

    def test_step_durations_computed(self):
        run = RunRecord()
        for i in range(5):
            run.add_frame(FrameRecord(
                t=i, timestamp=float(i), scene_state={},
                step_estimate=1 if i < 3 else 2,
                step_confidence=1.0, detections=[], hand_positions={},
            ))
        run.finish("success")
        assert run.step_durations == {"1": 3, "2": 2}

    def test_serialization_roundtrip(self):
        run = RunRecord(user_id="tester", task_id="test_task")
        run.add_frame(FrameRecord(
            t=0, timestamp=1.0, scene_state={"key": "val"}, step_estimate=1,
            step_confidence=0.95, detections=[{"class": "screw"}], hand_positions={},
        ))
        run.finish("success")

        d = run.to_dict()
        restored = RunRecord.from_dict(d)
        assert restored.run_id == run.run_id
        assert restored.user_id == "tester"
        assert len(restored.frames) == 1
        assert restored.frames[0].scene_state == {"key": "val"}


class TestRunStore:
    def test_save_and_load(self, tmp_path):
        store = RunStore(store_dir=str(tmp_path))
        run = RunRecord(user_id="test")
        run.add_frame(FrameRecord(
            t=0, timestamp=1.0, scene_state={}, step_estimate=1,
            step_confidence=1.0, detections=[], hand_positions={},
        ))
        run.finish("success")
        store.save(run)

        loaded = store.load(run.run_id)
        assert loaded is not None
        assert loaded.run_id == run.run_id
        assert loaded.outcome == "success"

    def test_load_nonexistent(self, tmp_path):
        store = RunStore(store_dir=str(tmp_path))
        assert store.load("does_not_exist") is None

    def test_load_index(self, tmp_path):
        store = RunStore(store_dir=str(tmp_path))
        for outcome in ["success", "failure", "success"]:
            run = RunRecord()
            run.finish(outcome)
            store.save(run)

        index = store.load_index()
        assert len(index) == 3

    def test_count(self, tmp_path):
        store = RunStore(store_dir=str(tmp_path))
        for outcome in ["success", "failure", "success"]:
            run = RunRecord()
            run.finish(outcome)
            store.save(run)

        assert store.count() == 3
        assert store.count("success") == 2
        assert store.count("failure") == 1

    def test_load_all_filtered(self, tmp_path):
        store = RunStore(store_dir=str(tmp_path))
        for outcome in ["success", "failure", "success"]:
            run = RunRecord()
            run.finish(outcome)
            store.save(run)

        successes = store.load_all(outcome="success")
        assert len(successes) == 2
        for r in successes:
            assert r.outcome == "success"
