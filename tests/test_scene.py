"""Tests for general scene understanding (src/perception/scene.py)."""

import pytest
import numpy as np

from src.perception.scene import ScenePerception, SceneUnderstanding, TextRegion
from src.perception.detector import Detection


class TestScenePerception:
    def test_empty_scene(self):
        sp = ScenePerception()
        assert sp.summary() == "empty scene"
        assert sp.people_count == 0
        assert sp.object_names == []
        assert sp.has_text is False

    def test_with_objects(self):
        sp = ScenePerception(objects=[
            Detection("person", 0, (0, 0, 100, 200), 0.9),
            Detection("laptop", 63, (200, 100, 500, 400), 0.85),
            Detection("cup", 41, (50, 300, 100, 400), 0.7),
        ])
        sp.people_count = 1
        sp.dominant_objects = ["person", "laptop", "cup"]
        assert sp.object_names == ["person", "laptop", "cup"]
        assert "person" in sp.summary()
        assert "laptop" in sp.summary()

    def test_objects_of_class(self):
        sp = ScenePerception(objects=[
            Detection("person", 0, (0, 0, 50, 100), 0.9),
            Detection("person", 0, (60, 0, 120, 100), 0.8),
            Detection("cup", 41, (200, 200, 250, 300), 0.7),
        ])
        people = sp.objects_of_class("person")
        assert len(people) == 2
        assert len(sp.objects_of_class("cup")) == 1
        assert len(sp.objects_of_class("chair")) == 0

    def test_text_regions(self):
        sp = ScenePerception(text_regions=[
            TextRegion("Hello World", (10, 10, 100, 30), 0.95),
        ])
        assert sp.has_text is True
        assert sp.all_text == "Hello World"
        assert "Hello World" in sp.summary()


class TestSceneUnderstanding:
    def test_mock_detect(self):
        """Without YOLO installed, should use mock detection."""
        su = SceneUnderstanding.__new__(SceneUnderstanding)
        su._yolo = None
        su._ocr_ready = False
        su._enable_ocr = False
        su._confidence = 0.35

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        perception = su.perceive(frame)
        assert len(perception.objects) > 0
        assert perception.inference_ms >= 0

    def test_rank_by_area(self):
        su = SceneUnderstanding.__new__(SceneUnderstanding)
        dets = [
            Detection("small", 0, (0, 0, 10, 10), 0.9),      # area 100
            Detection("big", 1, (0, 0, 200, 200), 0.9),       # area 40000
            Detection("medium", 2, (0, 0, 50, 50), 0.9),      # area 2500
        ]
        ranked = su._rank_by_area(dets, top_n=3)
        assert ranked[0] == "big"
        assert ranked[1] == "medium"
        assert ranked[2] == "small"

    def test_scene_tags_workspace(self):
        su = SceneUnderstanding.__new__(SceneUnderstanding)
        perception = ScenePerception(objects=[
            Detection("laptop", 63, (0, 0, 100, 100), 0.9),
            Detection("keyboard", 66, (0, 100, 100, 150), 0.8),
        ])
        tags = su._infer_scene_tags(perception)
        assert "workspace" in tags

    def test_scene_tags_kitchen(self):
        su = SceneUnderstanding.__new__(SceneUnderstanding)
        perception = ScenePerception(objects=[
            Detection("cup", 41, (0, 0, 50, 50), 0.9),
            Detection("bowl", 45, (60, 0, 110, 50), 0.8),
        ])
        tags = su._infer_scene_tags(perception)
        assert "kitchen/dining" in tags

    def test_scene_tags_social(self):
        su = SceneUnderstanding.__new__(SceneUnderstanding)
        perception = ScenePerception(people_count=3)
        tags = su._infer_scene_tags(perception)
        assert "social" in tags

    def test_scene_tags_outdoors(self):
        su = SceneUnderstanding.__new__(SceneUnderstanding)
        perception = ScenePerception(objects=[
            Detection("car", 2, (0, 0, 200, 100), 0.9),
        ])
        tags = su._infer_scene_tags(perception)
        assert "outdoors" in tags
