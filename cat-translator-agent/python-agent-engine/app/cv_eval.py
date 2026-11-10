"""
Leave-cats-out cross-validation: the honest way to measure this model.

    python -m app.cv_eval --data path/to/CatMeows --features mfcc ast \
                          --cleaning raw bandpass full --out reports/

What it answers
  1. How good is it really?  mean +/- spread over 5 folds x 3 repeats, plus a
     95% confidence interval that treats each CAT (not each clip) as one sample.
  2. Is it better than trivial guessing?  (majority-class baseline)
  3. Does MY cleaning help or hurt?        (--cleaning raw vs bandpass vs full)
  4. Do pretrained embeddings beat hand-made features?   (--features mfcc vs ast)
  5. If the app only answers when confident, how accurate is it and how often
     does it answer?                       (selective-prediction table)
  6. (--show-leak) How much would a careless clip-level split lie to us?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

K = 3  # number of classes (overridden from data in run())


# ----------------------------------------------------------------- splitting
def make_folds(groups: np.ndarray, n_splits: int, seed: int):
    """Split by CAT: shuffle the cats, deal them into folds like cards.
    A cat's clips all land in ONE fold, so test cats are never seen in training."""
    cats = np.unique(groups)
    np.random.default_rng(seed).shuffle(cats)
    fold_of_cat = {c: i % n_splits for i, c in enumerate(cats)}
    fold = np.array([fold_of_cat[g] for g in groups])
    return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(n_splits)]


def make_model(C: float = 0.5):
    # StandardScaler: puts every feature on the same scale. LogisticRegression: a
    # simple linear classifier; class_weight="balanced" stops it favouring the big class.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=3000, class_weight="balanced"),
    )


def out_of_fold_probs(
    X, y, groups, n_splits=5, repeats=3, C=0.5, seed=0, leaky=False, n_classes=None
):
    """Returns probs[r, i, k]: for every clip i, the probability predicted by a model
    that NEVER saw clip i's cat (unless leaky=True, the deliberately-wrong version)."""
    k = n_classes or K
    probs = np.zeros((repeats, len(y), k))
    for r in range(repeats):
        if (
            leaky
        ):  # WRONG on purpose: random clip split, cats shared between train and test
            folds = [
                (tr, te)
                for tr, te in StratifiedKFold(
                    n_splits, shuffle=True, random_state=seed + r
                ).split(X, y)
            ]
        else:
            folds = make_folds(groups, n_splits, seed + r)
        for tr, te in folds:
            m = make_model(C).fit(X[tr], y[tr])
            p = np.zeros((len(te), k))
            p[:, m.classes_] = m.predict_proba(X[te])
            probs[r, te] = p
    return probs


# ------------------------------------------------------------------ metrics
def macro_f1(y, pred, k=None):
    k = k or K
    f = []
    for c in range(k):
        tp = np.sum((y == c) & (pred == c))
        fp = np.sum((y != c) & (pred == c))
        fn = np.sum((y == c) & (pred != c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def cat_bootstrap_ci(correct: np.ndarray, groups: np.ndarray, n_boot=2000, seed=0):
    """95% CI for accuracy, resampling CATS with replacement (clips of one cat move
    together). correct: (repeats, n_clips) booleans."""
    rng = np.random.default_rng(seed)
    cats = np.unique(groups)
    hit = np.stack([correct[:, groups == c].sum(1) for c in cats])  # (cats, repeats)
    cnt = np.array([(groups == c).sum() for c in cats])  # (cats,)
    accs = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, len(cats), len(cats))
        accs[b] = (hit[s].sum(0) / cnt[s].sum()).mean()
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def balanced_accuracy(y, pred, k=None):
    """Average of per-class recall. Immune to class imbalance: always guessing the
    biggest class scores exactly 1/K, however big that class is."""
    k = k or K
    return float(
        np.mean([(pred[y == c] == c).mean() for c in range(k) if np.any(y == c)])
    )


def cat_bootstrap_ci_balanced(pred, y, groups, n_boot=2000, seed=0):
    """Same idea as cat_bootstrap_ci, but for balanced accuracy. pred: (repeats, n)."""
    rng = np.random.default_rng(seed)
    cats, k = np.unique(groups), K
    hit = np.array(
        [
            [
                [(pred[r][(groups == c) & (y == cl)] == cl).sum() for cl in range(k)]
                for r in range(pred.shape[0])
            ]
            for c in cats
        ]
    )  # (cats, repeats, k)
    cnt = np.array(
        [[((groups == c) & (y == cl)).sum() for cl in range(k)] for c in cats]
    )  # (cats, k)
    out = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, len(cats), len(cats))
        h, n = hit[s].sum(0), cnt[s].sum(0)
        ok = n > 0
        out[b] = (h[:, ok] / n[ok]).mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def selective_table(
    probs: np.ndarray, y: np.ndarray, thresholds=(0.0, 0.4, 0.5, 0.6, 0.7, 0.8)
):
    """If we only answer when top-probability >= t: what share of clips do we answer
    (coverage) and how often are those answers right?  Averaged over repeats."""
    rows = []
    for t in thresholds:
        cov, acc = [], []
        for p in probs:
            conf, pred = p.max(1), p.argmax(1)
            keep = conf >= t
            cov.append(keep.mean())
            acc.append((pred[keep] == y[keep]).mean() if keep.any() else np.nan)
        rows.append(
            {
                "threshold": t,
                "coverage": float(np.mean(cov)),
                "accuracy": float(np.nanmean(acc))
                if not np.all(np.isnan(acc))
                else None,
            }
        )
    return rows


def summarize(probs, y, groups, n_splits):
    """Every headline number for one (features, cleaning) setting."""
    pred = probs.argmax(2)  # (repeats, n)
    accs = (pred == y).mean(1)
    f1s = np.array([macro_f1(y, p) for p in pred])
    base = np.bincount(y).max() / len(y)  # always guess the biggest class
    lo, hi = cat_bootstrap_ci(pred == y, groups)
    bal = np.array([balanced_accuracy(y, p) for p in pred])
    blo, bhi = cat_bootstrap_ci_balanced(pred, y, groups)
    per_cat = {
        str(c): float((pred[:, groups == c] == y[groups == c]).mean())
        for c in np.unique(groups)
    }
    cm = np.zeros((K, K), int)
    for p in pred:
        for t, q in zip(y, p):
            cm[t, q] += 1
    return {
        "accuracy_mean": float(accs.mean()),
        "accuracy_std_over_repeats": float(accs.std()),
        "accuracy_ci95_by_cat": [lo, hi],
        "macro_f1": float(f1s.mean()),
        "balanced_accuracy_mean": float(bal.mean()),
        "balanced_accuracy_ci95_by_cat": [blo, bhi],
        "per_cat_accuracy": per_cat,
        "majority_baseline": float(base),
        "chance": 1 / K,
        "confusion_rows_true_cols_pred_avg_per_repeat": (cm / len(pred))
        .round(1)
        .tolist(),
        "selective": selective_table(probs, y),
    }


# ----------------------------------------------------------- feature caching
def build_features(files, feat_name, cleaning, cache_dir: Path, use_cache=True):
    from app.audio_dsp import AudioProcessor
    from app.catmeows_data import load_clip
    from app.features import get_feature_fn

    cache = cache_dir / f"{feat_name}_{cleaning}.npy"
    if use_cache and cache.is_file():
        X = np.load(cache)
        if len(X) == len(files):
            return X
    proc, fn = AudioProcessor(), get_feature_fn(feat_name)
    X = []
    for i, f in enumerate(files):
        X.append(fn(load_clip(proc, f, cleaning)))
        if (i + 1) % 50 == 0:
            print(f"   {feat_name}/{cleaning}: {i + 1}/{len(files)} clips")
    X = np.stack(X)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache, X)
    return X


def run(a):
    global K
    from app.catmeows_data import LABELS, list_clips

    files, y, groups = list_clips(a.data)
    K = len(LABELS)
    print(
        f"{len(files)} clips | {len(set(groups))} cats | class counts {np.bincount(y).tolist()}"
    )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for feat in a.features:
        for clean in a.cleaning:
            print(f"\n== features={feat}  cleaning={clean} ==")
            X = build_features(
                files, feat, clean, out / "feature_cache", not a.no_cache
            )
            probs = out_of_fold_probs(X, y, groups, a.splits, a.repeats, a.C, a.seed)
            res = summarize(probs, y, groups, a.splits)
            if a.show_leak:
                lk = out_of_fold_probs(
                    X, y, groups, a.splits, 1, a.C, a.seed, leaky=True
                )
                res["leaky_clip_split_accuracy_DO_NOT_TRUST"] = float(
                    (lk.argmax(2) == y).mean()
                )
            results[f"{feat}/{clean}"] = res
            np.save(out / f"oof_{feat}_{clean}.npy", probs)
    print_report(results)
    (out / "cv_report.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out / 'cv_report.json'}")


def print_report(results):
    print("\n" + "=" * 100)
    print(
        "HEADLINE = balanced accuracy (chance = 0.333). Plain accuracy is shown only for reference."
    )
    print(
        f"{'setting':16s} {'balanced acc':>12s} {'95% CI (by cat)':>17s} {'macroF1':>8s} {'accuracy':>9s} {'majority':>9s} {'leaky*':>8s}"
    )
    for k, r in results.items():
        lo, hi = r["balanced_accuracy_ci95_by_cat"]
        leak = r.get("leaky_clip_split_accuracy_DO_NOT_TRUST")
        print(
            f"{k:16s} {r['balanced_accuracy_mean']:12.3f} {f'[{lo:.2f}, {hi:.2f}]':>17s} {r['macro_f1']:8.3f} "
            f"{r['accuracy_mean']:9.3f} {r['majority_baseline']:9.3f} {('%.3f' % leak) if leak else '-':>8s}"
        )
    print(
        "* leaky = random clip split (plain accuracy); shown only to illustrate how misleading it is"
    )
    best = max(results, key=lambda k: results[k]["balanced_accuracy_mean"])
    r = results[best]
    print(f"\nBest by balanced accuracy: {best}")
    print(
        "Confusion (rows=true, cols=pred, avg per repeat):",
        r["confusion_rows_true_cols_pred_avg_per_repeat"],
    )
    pc = np.array(list(r["per_cat_accuracy"].values()))
    print(
        f"Per-cat accuracy: min {pc.min():.2f} | median {np.median(pc):.2f} | max {pc.max():.2f} | "
        f"{(pc > 1 / K).sum()} of {len(pc)} cats above 1/{K}"
    )
    print("Answer-only-when-confident (plain accuracy of the answers given):")
    for row in r["selective"]:
        acc = "n/a" if row["accuracy"] is None else f"{row['accuracy']:.3f}"
        print(
            f"   confidence >= {row['threshold']:.1f}: answers {row['coverage']:.0%} of clips, right {acc}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--features", nargs="+", default=["mfcc"], choices=["mfcc", "ast"])
    ap.add_argument(
        "--cleaning", nargs="+", default=["full"], choices=["raw", "bandpass", "full"]
    )
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--C", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--show-leak", action="store_true")
    run(ap.parse_args())
