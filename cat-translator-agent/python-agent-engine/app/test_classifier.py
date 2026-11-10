"""
Smoke test.  Run from python-agent-engine/ :   python -m app.test_classifier
Needs only torch (+ torchvision for the resnet/mobilenet checks).  No dataset.
"""

import tempfile
from pathlib import Path

import numpy as np
import torch

from app.cnn_classifier import CatIntentClassifier, build_model

LABELS = ["brushing", "isolation", "waiting_for_food"]

# 1. every backbone accepts BOTH frame lengths (65 = new pipeline, 130 = old 3 s format)
for bb in ["small", "resnet18", "mobilenet_v3_small"]:
    try:
        m = build_model(3, bb).eval()
        for frames in (65, 130):
            out = m(torch.randn(2, 1, 128, frames))
            assert out.shape == (2, 3), out.shape
        print(
            f"[ok] backbone {bb:20s} params={sum(p.numel() for p in m.parameters()) / 1e6:.2f}M"
        )
    except ImportError as e:
        print(f"[skip] {bb}: {e}")

# 2. untrained model must REFUSE to give an intent
clf = CatIntentClassifier(labels=LABELS)
r = clf.predict(np.random.randn(3, 1, 128, 65).astype(np.float32))
assert r.status == "untrained" and r.label is None, r
print("[ok] untrained model refuses:", r.message)

# 3. empty input -> no_meow
assert clf.predict(np.empty((0, 1, 128, 65), np.float32)).status == "no_meow"
print("[ok] no-meow input handled")


# 4. tiny synthetic training run proves learning + save/load + decision path
def fake(c, n):
    x = np.random.randn(n, 1, 128, 65).astype(np.float32) * 0.3
    x[:, :, 30 * (c + 1) : 30 * (c + 1) + 10, :] += (
        2.0  # class = which mel band is bright
    )
    return x


X = np.concatenate([fake(c, 24) for c in range(3)])
Y = np.repeat(np.arange(3), 24)
torch.manual_seed(0)
opt = torch.optim.Adam(clf.model.parameters(), 2e-3)
lf = torch.nn.CrossEntropyLoss()
clf.model.train()
for step in range(60):
    ids = np.random.choice(len(X), 16, replace=False)
    opt.zero_grad()
    lf(clf.model(torch.from_numpy(X[ids])), torch.from_numpy(Y[ids])).backward()
    opt.step()
clf.trained = True
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "t.pt"
    clf.save(p)
    clf2 = CatIntentClassifier(labels=LABELS, checkpoint=p)
for c in range(3):
    r = clf2.predict(fake(c, 3))  # 3 "meows" in one clip
    assert r.status == "ok" and r.label == LABELS[c], (c, r.to_dict())
print("[ok] train -> save -> load -> predict works, e.g.", r.label, r.confidence)
print("ALL CHECKS PASSED")
