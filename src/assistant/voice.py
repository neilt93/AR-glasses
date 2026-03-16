"""Voice output for the AR assistant — makes it talk.

Uses pyttsx3 for offline text-to-speech. Runs in a background thread
so speech never blocks the main perception loop.

Usage:
    voice = VoiceEngine()
    voice.say("Hello, I see a laptop")
    voice.say("Warning: person approaching", priority=True)
    voice.shutdown()
"""

import queue
import threading
from dataclasses import dataclass, field


@dataclass
class _Utterance:
    """A queued speech utterance."""
    text: str
    priority: bool = False


class VoiceEngine:
    """Async text-to-speech engine for the AR assistant.

    Speaks in a background thread. Priority messages jump the queue.
    Duplicate messages within a cooldown window are suppressed.
    """

    def __init__(
        self,
        enabled: bool = True,
        rate: int = 180,        # words per minute
        volume: float = 0.9,    # 0.0 - 1.0
        cooldown: float = 5.0,  # suppress duplicates within N seconds
    ):
        self.enabled = enabled
        self._rate = rate
        self._volume = volume
        self._cooldown = cooldown
        self._queue: queue.Queue[_Utterance | None] = queue.Queue(maxsize=20)
        self._recent: dict[str, float] = {}
        self._engine = None
        self._thread: threading.Thread | None = None
        self._running = False

        if enabled:
            self._start()

    def _start(self):
        """Start the background TTS thread."""
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        """Background thread: pull utterances from queue and speak them."""
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self._rate)
            self._engine.setProperty("volume", self._volume)
        except Exception as e:
            print(f"[voice] TTS init failed: {e}")
            print("[voice] Install pyttsx3: pip install pyttsx3")
            self._running = False
            return

        while self._running:
            try:
                utterance = self._queue.get(timeout=0.5)
                if utterance is None:  # shutdown signal
                    break
                self._speak(utterance.text)
            except queue.Empty:
                continue

    def _speak(self, text: str):
        """Actually speak the text (runs in worker thread)."""
        if self._engine is None:
            return
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception:
            pass  # TTS errors shouldn't crash the assistant

    def say(self, text: str, priority: bool = False):
        """Queue a message to be spoken.

        Args:
            text: What to say.
            priority: If True, clears the queue and speaks this immediately.
        """
        if not self.enabled or not self._running:
            return

        # Suppress duplicates within cooldown
        import time
        now = time.time()
        key = text[:40]  # dedup key
        last = self._recent.get(key, 0)
        if (now - last) < self._cooldown:
            return
        self._recent[key] = now

        # Clean old entries
        cutoff = now - self._cooldown * 2
        self._recent = {k: v for k, v in self._recent.items() if v > cutoff}

        if priority:
            # Clear queue for urgent messages
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

        try:
            self._queue.put_nowait(_Utterance(text=text, priority=priority))
        except queue.Full:
            pass  # drop if queue is full

    def shutdown(self):
        """Stop the TTS thread cleanly."""
        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    @property
    def is_ready(self) -> bool:
        return self._running and self._engine is not None


class MockVoiceEngine:
    """Silent voice engine for testing or when TTS is disabled."""

    def __init__(self):
        self.enabled = False
        self.spoken: list[str] = []

    def say(self, text: str, priority: bool = False):
        self.spoken.append(text)

    def shutdown(self):
        pass

    @property
    def is_ready(self) -> bool:
        return True
