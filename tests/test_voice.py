"""Tests for voice engine (src/assistant/voice.py)."""

import pytest

from src.assistant.voice import VoiceEngine, MockVoiceEngine


class TestMockVoiceEngine:
    def test_say(self):
        voice = MockVoiceEngine()
        voice.say("hello world")
        assert voice.spoken == ["hello world"]

    def test_multiple_messages(self):
        voice = MockVoiceEngine()
        voice.say("one")
        voice.say("two")
        voice.say("three")
        assert len(voice.spoken) == 3

    def test_shutdown_noop(self):
        voice = MockVoiceEngine()
        voice.shutdown()  # should not raise

    def test_not_enabled(self):
        voice = MockVoiceEngine()
        assert voice.enabled is False

    def test_is_ready(self):
        voice = MockVoiceEngine()
        assert voice.is_ready is True


class TestVoiceEngine:
    def test_disabled_voice_doesnt_start(self):
        voice = VoiceEngine(enabled=False)
        assert voice._running is False
        voice.shutdown()

    def test_say_when_disabled(self):
        voice = VoiceEngine(enabled=False)
        voice.say("hello")  # should not raise
        voice.shutdown()

    def test_shutdown_twice(self):
        voice = VoiceEngine(enabled=False)
        voice.shutdown()
        voice.shutdown()  # should not raise
