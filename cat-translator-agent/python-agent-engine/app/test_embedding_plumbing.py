"""Plumbing test with synthetic 'meows' (pitch glides). Run:  python -m app.test_embedding_plumbing"""
import tempfile
from pathlib import Path
import numpy as np
from app.embedding_classifier import EmbeddingIntentClassifier, fit_and_save
from app.features import mfcc_stats

SR, LABELS, rng = 22050, ["brushing", "isolation", "waiting_for_food"], np.random.default_rng(1)

def fake_meow(c):                      # class c = different base pitch glide
    t = np.linspace(0, 0.7, int(0.7 * SR)); f0 = 350 + 250 * c + 80 * t / 0.7 * (c + 1)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    y = sum(np.sin(h * ph) / h for h in (1, 2, 3)) * np.hanning(len(t))
    return (y * 0.3 + rng.normal(0, 0.01, len(t))).astype(np.float32)

waves = [fake_meow(c) for c in range(3) for _ in range(30)]
y = np.repeat(np.arange(3), 30); groups = np.array([f"cat{i % 10}" for i in range(90)])
X = np.stack([mfcc_stats(w) for w in waves])

assert EmbeddingIntentClassifier(None).predict_waveforms(waves[:2]).status == "untrained"
assert EmbeddingIntentClassifier(None).predict_waveforms([]).status == "no_meow"
print("[ok] untrained refuses / empty -> no_meow")

with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "m.joblib"
    T = fit_and_save(X, y, groups, LABELS, "mfcc", "full", p)
    clf = EmbeddingIntentClassifier(p)
for c in range(3):
    r = clf.predict_waveforms([fake_meow(c) for _ in range(3)])
    assert r.status == "ok" and r.label == LABELS[c], r.to_dict()
print(f"[ok] fit -> save -> load -> predict works (T={T:.2f}); example:", r.label, r.confidence)
print("ALL PLUMBING CHECKS PASSED")