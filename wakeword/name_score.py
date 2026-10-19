"""How sure Whisper is that a phrase starts with the name - measured, not asked.

The cascade first asked Whisper to write what it heard, with "Джарвис." as the hint; over TV and talk it
wrote the hint back - "Джарвис." - ~14 times an hour. Here nothing is written and there is no hint: the
audio goes through the encoder once, and the decoder is made to read " Джарвис" (or " Эй, Джарвис") as the
first words; the score is the mean log-probability it gives those tokens. Over a phrase that isn't the
name, the model expects other words, and the score is low - there's no hint to repeat.
"""

from __future__ import annotations

import numpy as np

TARGETS = (" Джарвис", " Эй, Джарвис", " Джарвис,")


class NameScorer:
    def __init__(self, model):
        """model: a loaded openai-whisper model (wakeword/verify.py's NameVerifier.model)."""
        import whisper
        from whisper.tokenizer import get_tokenizer

        self.model = model
        self.whisper = whisper
        tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages, language="ru",
                                  task="transcribe")
        self.prefix = list(tokenizer.sot_sequence_including_notimestamps)
        self.targets = [tokenizer.encode(t) for t in TARGETS]

    def score(self, audio: np.ndarray) -> float:
        """Mean log-probability of the best of TARGETS as the phrase's first words (0 is certain)."""
        import torch

        x = self.whisper.pad_or_trim(audio.astype(np.float32) / 32768)
        mel = self.whisper.log_mel_spectrogram(x, n_mels=self.model.dims.n_mels).to(self.model.device)
        dtype = next(self.model.parameters()).dtype
        with torch.no_grad():
            features = self.model.embed_audio(mel[None].to(dtype))
            best = -1e9
            for ids in self.targets:
                tokens = torch.tensor([self.prefix + ids], device=self.model.device)
                logprobs = torch.log_softmax(self.model.logits(tokens, features).float(), dim=-1)[0]
                start = len(self.prefix) - 1  # the position that predicts the first target token
                lp = [float(logprobs[start + i, t]) for i, t in enumerate(ids)]
                best = max(best, sum(lp) / len(lp))
        return best
