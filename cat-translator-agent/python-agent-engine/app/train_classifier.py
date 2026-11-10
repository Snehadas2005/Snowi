"""
Train the intent classifier on the public CatMeows dataset (Zenodo, ~440 clips,
3 situations: brushing / isolation / waiting for food).

    python -m app.train_classifier --data path/to/CatMeows --out models/cat_intent.pt

File names look like  B_ANI01_MC_FN_SIM01_101.wav :
    first letter = situation (B brushing, I isolation, F waiting for food)
    second field = cat ID
THE SPLIT IS DONE BY CAT, NOT BY CLIP.  Clips from the same cat are very
similar; mixing them across train/test makes accuracy look far better than it
really is.  Reported numbers are on cats the model never saw.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn

from app.audio_dsp import AudioProcessor
from app.cnn_classifier import (
    INTENT_LABELS_CATMEOWS,
    CatIntentClassifier,
    fit_temperature,
    softmax,
)

CONTEXT = {"B": "brushing", "I": "isolation", "F": "waiting_for_food"}


def parse_name(p: Path):
    parts = p.stem.split("_")
    return CONTEXT[parts[0].upper()], parts[1]


def load_spec(proc: AudioProcessor, path: Path) -> np.ndarray:
    """Clips are already single meows: clean WITHOUT event gating, trim, fit length."""
    y = proc.load_audio(path)
    clean, _ = proc.clean_waveform(y, gate=False)
    trimmed, _ = librosa.effects.trim(clean, top_db=30)
    seg = trimmed if len(trimmed) > int(0.1 * proc.sr) else clean
    return proc.log_mel(proc._fit_length(seg))[None]  # (1, mels, frames)


def augment(spec: np.ndarray, rng: random.Random) -> np.ndarray:
    s = spec.copy()
    s = np.roll(s, rng.randint(-6, 6), axis=2)  # small time shift
    f = rng.randint(0, 12)
    f0 = rng.randint(0, s.shape[1] - f)
    s[:, f0 : f0 + f, :] = 0  # frequency mask
    t = rng.randint(0, 8)
    t0 = rng.randint(0, max(1, s.shape[2] - t))
    s[:, :, t0 : t0 + t] = 0  # time mask
    return (s + np.random.normal(0, 0.05, s.shape)).astype(np.float32)


def macro_f1(y, p, k):
    f1s = []
    for c in range(k):
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/cat_intent.pt")
    ap.add_argument(
        "--backbone",
        default="small",
        choices=["small", "resnet18", "mobilenet_v3_small"],
    )
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    labels = INTENT_LABELS_CATMEOWS
    files = sorted(Path(a.data).rglob("*.wav"))
    proc = AudioProcessor()
    X, Y, G = [], [], []
    for f in files:
        try:
            lab, cat = parse_name(f)
        except (KeyError, IndexError):
            continue
        X.append(load_spec(proc, f))
        Y.append(labels.index(lab))
        G.append(cat)
    X, Y, G = np.stack(X), np.array(Y), np.array(G)
    print(f"{len(X)} clips, {len(set(G))} cats, class counts {np.bincount(Y)}")

    cats = sorted(set(G))
    rng.shuffle(cats)
    n = len(cats)
    tr_c, va_c, te_c = (
        cats[: int(0.6 * n)],
        cats[int(0.6 * n) : int(0.8 * n)],
        cats[int(0.8 * n) :],
    )
    idx = lambda cs: np.where(np.isin(G, cs))[0]
    tr, va, te = idx(tr_c), idx(va_c), idx(te_c)
    print(f"train {len(tr)} | val {len(va)} | test {len(te)} clips (split by cat)")

    clf = CatIntentClassifier(labels=labels, backbone=a.backbone)
    model, dev = clf.model, clf.device
    w = torch.tensor(
        len(tr)
        / (len(labels) * np.maximum(np.bincount(Y[tr], minlength=len(labels)), 1)),
        dtype=torch.float32,
        device=dev,
    )
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, a.lr * 3, total_steps=a.epochs * max(1, len(tr) // a.bs)
    )

    def logits_of(ids):
        model.eval()
        with torch.no_grad():
            return model(torch.from_numpy(X[ids]).to(dev)).cpu().numpy()

    best, best_state = -1.0, None
    for ep in range(a.epochs):
        model.train()
        order = rng.sample(list(tr), len(tr))
        for i in range(0, len(order) - a.bs + 1, a.bs):  # drop_last (BatchNorm)
            b = order[i : i + a.bs]
            xb = torch.from_numpy(np.stack([augment(X[j], rng) for j in b])).to(dev)
            yb = torch.from_numpy(Y[b]).to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
        lv = logits_of(va)
        f1 = macro_f1(Y[va], lv.argmax(1), len(labels))
        if f1 > best:
            best, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"epoch {ep:3d}  loss {loss.item():.3f}  val macroF1 {f1:.3f}")

    model.load_state_dict(best_state)
    clf.temperature = fit_temperature(logits_of(va), Y[va])
    lt = logits_of(te)
    pt = softmax(lt, clf.temperature).argmax(1)
    acc = float((pt == Y[te]).mean())
    chance = float(np.bincount(Y[te]).max() / len(te))
    print(
        f"\nHELD-OUT CATS  accuracy {acc:.3f}  macroF1 {macro_f1(Y[te], pt, len(labels)):.3f}"
        f"  (majority-class baseline {chance:.3f}, random {1 / len(labels):.3f})  T={clf.temperature:.2f}"
    )
    cm = np.zeros((len(labels),) * 2, int)
    for t, p in zip(Y[te], pt):
        cm[t, p] += 1
    print("confusion (rows=true, cols=pred):", labels, "\n", cm)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    clf.trained = True
    clf.save(
        a.out, extra={"test_accuracy": acc, "backbone": a.backbone, "seed": a.seed}
    )
    print("saved", a.out)


if __name__ == "__main__":
    main()
