import numpy as np

from wakeword import cascade
from wakeword.verify import names_in


class Scorer:
    def __init__(self, value):
        self.value, self.asked = value, 0

    def score(self, audio):
        self.asked += 1
        return self.value


def _ready(score, tmp_path, monkeypatch):
    monkeypatch.setattr(cascade, "STATS", tmp_path / "wake_stats.log")
    c = cascade.Cascade()
    c._scorer = Scorer(score)
    c._ready.set()
    return c


def test_the_name_is_measured_and_every_decision_counted_without_words(tmp_path, monkeypatch):
    audio = np.zeros(16000, dtype=np.int16)
    named = _ready(-1.2, tmp_path, monkeypatch)
    heard, score, note = named.check(audio, 0.8)
    assert heard and score == -1.2 and "NAME" in note
    other = _ready(-5.4, tmp_path, monkeypatch)  # what talk and near-names scored
    assert other.check(audio, 0.8)[0] is False
    lines = (tmp_path / "wake_stats.log").read_text(encoding="utf-8").splitlines()
    assert [line.split("\t")[1:] for line in lines] == [["-1.20", "woke", "0.8"], ["-5.40", "-", "0.8"]]


def test_too_short_or_too_long_isnt_scored_and_nothing_loaded_is_vosk_alone(tmp_path, monkeypatch):
    c = _ready(-0.5, tmp_path, monkeypatch)
    assert c.check(np.zeros(1600, dtype=np.int16), 0.1)[0] is False  # a click
    assert c.check(np.zeros(16000, dtype=np.int16), 12.0)[0] is False  # someone talking at length
    assert c._scorer.asked == 0
    failed = cascade.Cascade()
    failed.error = "no GPU"
    failed._ready.set()
    assert failed.check(np.zeros(16000, dtype=np.int16), 1.0) == (False, None, "")


def test_whisper_text_counts_as_the_name_only_when_it_is_the_name():
    assert names_in("Джарвис!") and names_in("Эй, джарвиз, слушай")  # one letter off is still it
    assert not names_in("Джордж") and not names_in("Червис") and not names_in("")
