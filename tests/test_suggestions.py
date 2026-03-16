"""Tests for proactive suggestion engine (src/assistant/suggestions.py)."""

import time
import pytest

from src.assistant.suggestions import SuggestionEngine, Suggestion


class TestSuggestionEngine:
    def test_no_suggestions_empty_scene(self):
        engine = SuggestionEngine()
        result = engine.suggest(
            object_names=set(),
            scene_tags=[],
        )
        assert result == []

    def test_phone_on_desk_suggestion(self):
        engine = SuggestionEngine(cooldown=0)
        result = engine.suggest(
            object_names={"cell phone", "laptop"},
            scene_tags=["workspace"],
        )
        focus = [s for s in result if s.key == "focus_mode"]
        assert len(focus) == 1
        assert "focus" in focus[0].text.lower()

    def test_multi_screen_suggestion(self):
        engine = SuggestionEngine(cooldown=0)
        result = engine.suggest(
            object_names={"tv", "laptop"},
            scene_tags=["workspace"],
        )
        multi = [s for s in result if s.key == "multi_screen"]
        assert len(multi) == 1

    def test_person_approaching(self):
        engine = SuggestionEngine(cooldown=0)
        result = engine.suggest(
            object_names={"person"},
            scene_tags=[],
            people_count=2,
            prev_people_count=1,
        )
        approach = [s for s in result if s.key == "person_approach"]
        assert len(approach) == 1

    def test_dark_scene_warning(self):
        engine = SuggestionEngine(cooldown=0)
        result = engine.suggest(
            object_names=set(),
            scene_tags=[],
            avg_brightness=15.0,
        )
        dark = [s for s in result if s.key == "low_light"]
        assert len(dark) == 1

    def test_cooldown_prevents_repeat(self):
        engine = SuggestionEngine(cooldown=999)
        result1 = engine.suggest(
            object_names={"cell phone"},
            scene_tags=["workspace"],
        )
        result2 = engine.suggest(
            object_names={"cell phone"},
            scene_tags=["workspace"],
        )
        # First time should suggest, second time should be suppressed
        focus1 = [s for s in result1 if s.key == "focus_mode"]
        focus2 = [s for s in result2 if s.key == "focus_mode"]
        assert len(focus1) == 1
        assert len(focus2) == 0

    def test_sorted_by_priority(self):
        engine = SuggestionEngine(cooldown=0)
        result = engine.suggest(
            object_names={"cell phone", "person"},
            scene_tags=["workspace"],
            people_count=2,
            prev_people_count=1,
            avg_brightness=15.0,
        )
        if len(result) >= 2:
            assert result[0].priority >= result[1].priority

    def test_reset(self):
        engine = SuggestionEngine(cooldown=999)
        engine.suggest(
            object_names={"cell phone"},
            scene_tags=["workspace"],
        )
        engine.reset()
        result = engine.suggest(
            object_names={"cell phone"},
            scene_tags=["workspace"],
        )
        focus = [s for s in result if s.key == "focus_mode"]
        assert len(focus) == 1  # should work again after reset

    def test_leaving_workspace(self):
        engine = SuggestionEngine(cooldown=0)
        result = engine.suggest(
            object_names=set(),
            scene_tags=["kitchen/dining"],
            prev_tags=["workspace"],
        )
        leaving = [s for s in result if s.key == "leaving_workspace"]
        assert len(leaving) == 1

    def test_coffee_reminder_needs_time(self):
        engine = SuggestionEngine(cooldown=0)
        # Set scene start way in the past
        engine._scene_start = time.time() - 400
        result = engine.suggest(
            object_names={"cup"},
            scene_tags=["workspace"],
        )
        coffee = [s for s in result if s.key == "coffee_reminder"]
        assert len(coffee) == 1

    def test_coffee_no_reminder_too_early(self):
        engine = SuggestionEngine(cooldown=0)
        result = engine.suggest(
            object_names={"cup"},
            scene_tags=["workspace"],
        )
        coffee = [s for s in result if s.key == "coffee_reminder"]
        assert len(coffee) == 0  # too soon
