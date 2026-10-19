"""The name when Vosk misses it: Whisper (already on the GPU as the speech recognizer) measures how sure
it is that the phrase starts with "Джарвис" - wakeword/name_score.py - and over SCORE_THRESHOLD, Jarvis
wakes; the phrase then goes to speech recognition like any other, so "Джарвис, включи свет" is done at
once and the bare name just opens listening.

History: the first cascade asked Whisper to *write* the phrase with "Джарвис." as a hint and woke if the
name came back. On the residents' recordings that was 137 of 139 names and no ordinary phrase; live, over
TV and talk, Whisper wrote the hint back - "Джарвис." - about 14 times an hour, and it was switched off.
Measured instead of written, with no hint to repeat: on the residents' recordings Vosk plus this hears
109 of 139 names (Vosk alone 91); on 237 phrases of nonstop synthetic chatter salted with near-names
("Джарвин", "Дарвис") it woke once, on 51 phrases from the room none.

Every decision goes to wake_stats.log - the time, the score, woke or not; no words - so false wakes per
hour can be counted from real days without keeping what was said.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

SCORE_THRESHOLD = -2.5
MIN_SPEECH_SECONDS = 0.3
MAX_SPEECH_SECONDS = 8.0  # longer is talk, not someone calling Jarvis
STATS = Path(__file__).resolve().parents[1] / "wake_stats.log"


class Cascade:
    def __init__(self, threshold: float = SCORE_THRESHOLD, whisper: str = "turbo", collect: bool = False):
        self.threshold = threshold
        self.whisper = whisper
        self.collect = collect  # kept for the settings; the scored cascade keeps no audio
        self._verifier = None
        self._scorer = None
        self._ready = threading.Event()
        self.error: str | None = None

    def load(self) -> None:
        """Whisper (shared with speech recognition) and the scorer - a few seconds; on a thread at startup."""
        try:
            from wakeword.name_score import NameScorer
            from wakeword.verify import NameVerifier

            self._verifier = NameVerifier(self.whisper)
            self._scorer = NameScorer(self._verifier.model)
        except Exception as exc:  # noqa: BLE001 - no GPU, no whisper: Vosk alone, as before
            self.error = repr(exc)
        finally:
            self._ready.set()

    @property
    def ready(self) -> bool:
        return self._ready.is_set() and self.error is None

    def check(self, audio: np.ndarray, speech_seconds: float) -> tuple[bool, float | None, str]:
        """(the name is there, its score, a note) - for a phrase Vosk heard no name in."""
        if not self.ready or not MIN_SPEECH_SECONDS <= speech_seconds <= MAX_SPEECH_SECONDS:
            return False, None, ""
        started = time.perf_counter()
        score = self._scorer.score(audio)
        heard = score >= self.threshold
        seconds = time.perf_counter() - started
        self._stat(score, heard, speech_seconds)
        return heard, score, f"name score {score:.2f} ({seconds:.2f}s) -> {'NAME' if heard else 'not the name'}"

    def transcribe(self, audio: np.ndarray, prompt: str = "") -> str:
        """The loaded Whisper as the local speech recognizer."""
        if not self.ready:
            raise RuntimeError("the local Whisper isn't loaded")
        return self._verifier.transcribe_phrase(audio, prompt)

    def _stat(self, score: float, heard: bool, speech_seconds: float) -> None:
        try:
            with STATS.open("a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{score:.2f}\t{'woke' if heard else '-'}\t{speech_seconds:.1f}\n")
        except OSError:
            pass
