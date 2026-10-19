"""The baseline: a small general Russian recognizer (Vosk) listening for the name in what it transcribes.
VOSK_MODEL_PATH: vosk-model-small-ru-0.22 (https://alphacephei.com/vosk/models)."""
import json
import os

import numpy as np

WAKE_WORDS = ("джарвис", "джервис")
_recognizer = None


def heard(audio: np.ndarray, rate: int = 16000) -> bool:
    global _recognizer
    from vosk import KaldiRecognizer, Model, SetLogLevel

    if _recognizer is None:
        SetLogLevel(-1)
        _recognizer = KaldiRecognizer(Model(os.environ.get("VOSK_MODEL_PATH", "models/vosk-model-small-ru-0.22")), rate)
    _recognizer.Reset()
    _recognizer.AcceptWaveform(audio.astype(np.int16).tobytes())
    words = json.loads(_recognizer.FinalResult()).get("text", "").split()
    return any(w in WAKE_WORDS for w in words)
