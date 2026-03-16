"""Proactive suggestion engine — the assistant volunteers useful info.

Based on what the assistant sees, it generates contextual suggestions
like "Pro tip: check battery level" or "You have a coffee nearby".

This is what makes the assistant feel intelligent — it anticipates
what you might need rather than just reacting.

Usage:
    engine = SuggestionEngine()
    suggestions = engine.suggest(perception, context)
    for s in suggestions:
        print(f"[{s.category}] {s.text}")
"""

import time
from dataclasses import dataclass, field


@dataclass
class Suggestion:
    """A proactive suggestion from the assistant."""
    text: str
    category: str       # "safety", "productivity", "social", "info"
    priority: float     # 0.0 - 1.0 (higher = more important)
    speak: bool = True  # whether to speak this aloud
    key: str = ""       # dedup key


# Suggestion rules: each takes perception/context and returns suggestions
# Format: (condition_fn, suggestion_fn)

def _check_phone_on_desk(names: set[str], tags: list[str]) -> list[Suggestion]:
    """Phone on desk during work → offer focus mode."""
    if "cell phone" in names and "workspace" in tags:
        return [Suggestion(
            text="Phone detected — want to enable focus mode?",
            category="productivity",
            priority=0.3,
            key="focus_mode",
        )]
    return []


def _check_keys_visible(names: set[str], tags: list[str]) -> list[Suggestion]:
    """Keys visible → remind user not to forget them."""
    # Closest COCO match — using common objects that might be keys-adjacent
    return []


def _check_multiple_screens(names: set[str], tags: list[str]) -> list[Suggestion]:
    """Multiple monitors → productivity tip."""
    if "tv" in names and "laptop" in names:
        return [Suggestion(
            text="Multi-screen setup detected",
            category="info",
            priority=0.1,
            speak=False,
            key="multi_screen",
        )]
    return []


def _check_coffee_cooling(names: set[str], scene_duration: float) -> list[Suggestion]:
    """Cup visible for a while → remind about coffee."""
    if "cup" in names and scene_duration > 300:  # 5 minutes
        return [Suggestion(
            text="Your drink has been sitting for a while",
            category="info",
            priority=0.2,
            key="coffee_reminder",
        )]
    return []


def _check_person_approaching(people_count: int, prev_count: int) -> list[Suggestion]:
    """New person entered → social awareness."""
    if people_count > prev_count and people_count > 0:
        return [Suggestion(
            text="Someone is approaching",
            category="social",
            priority=0.5,
            key="person_approach",
        )]
    return []


def _check_leaving_workspace(names: set[str], prev_tags: list[str],
                              tags: list[str]) -> list[Suggestion]:
    """Transitioning away from workspace → reminder."""
    if "workspace" in prev_tags and "workspace" not in tags:
        items = []
        if "cell phone" not in names:
            items.append("phone")
        if items:
            return [Suggestion(
                text=f"Leaving workspace — don't forget your {', '.join(items)}",
                category="safety",
                priority=0.6,
                key="leaving_workspace",
            )]
    return []


def _check_dark_scene(avg_brightness: float) -> list[Suggestion]:
    """Very dark environment → suggest turning on lights."""
    if avg_brightness < 30:
        return [Suggestion(
            text="Low light detected — consider turning on a light",
            category="safety",
            priority=0.4,
            key="low_light",
        )]
    return []


class SuggestionEngine:
    """Generates proactive suggestions based on scene context.

    Rate-limited per suggestion key to avoid nagging.
    """

    def __init__(self, cooldown: float = 120.0):
        self._cooldown = cooldown
        self._last_suggested: dict[str, float] = {}
        self._scene_start: float = time.time()

    def suggest(
        self,
        object_names: set[str],
        scene_tags: list[str],
        people_count: int = 0,
        prev_people_count: int = 0,
        prev_tags: list[str] | None = None,
        avg_brightness: float = 128.0,
    ) -> list[Suggestion]:
        """Generate suggestions based on current scene state."""
        now = time.time()
        scene_duration = now - self._scene_start
        all_suggestions = []

        all_suggestions.extend(_check_phone_on_desk(object_names, scene_tags))
        all_suggestions.extend(_check_multiple_screens(object_names, scene_tags))
        all_suggestions.extend(_check_coffee_cooling(object_names, scene_duration))
        all_suggestions.extend(_check_person_approaching(people_count, prev_people_count))
        all_suggestions.extend(
            _check_leaving_workspace(object_names, prev_tags or [], scene_tags)
        )
        all_suggestions.extend(_check_dark_scene(avg_brightness))

        # Filter by cooldown
        filtered = []
        for s in all_suggestions:
            key = s.key or s.text[:30]
            last = self._last_suggested.get(key, 0)
            if (now - last) >= self._cooldown:
                filtered.append(s)
                self._last_suggested[key] = now

        # Sort by priority (highest first)
        filtered.sort(key=lambda s: s.priority, reverse=True)
        return filtered

    def reset_scene_timer(self):
        """Reset the scene duration timer (e.g., on major scene change)."""
        self._scene_start = time.time()

    def reset(self):
        """Full reset."""
        self._last_suggested.clear()
        self._scene_start = time.time()
