"""Checks the evaluation machinery itself on synthetic data (no dataset, no torch).
Run from python-agent-engine/ :  python -m app.test_cv_logic"""

import numpy as np
import app.cv_eval as cv

rng = np.random.default_rng(0)

# --- synthetic world: 24 cats x 18 clips, 3 situations.
# Each cat has (a) a personal "voice" and (b) personal QUIRKS in how it sounds in each
# situation, on top of (c) a weak pattern shared by all cats.  Only (c) generalises to
# new cats.  A clip-level split lets the model learn (b) from the same cat's other
# clips and look brilliant; a cat-level split cannot.  That is exactly the leak.
n_cats, per_cat, K, d = 24, 18, 3, 60
cv.K = K
cat_idx = np.repeat(np.arange(n_cats), per_cat)
groups = np.array([f"cat{i}" for i in cat_idx])
y = np.tile(np.arange(K), n_cats * per_cat // K)
voice = rng.normal(0, 2.0, (n_cats, d))[cat_idx]
shared = rng.normal(0, 1, (K, d))
shared *= 1.5 / np.linalg.norm(shared, axis=1, keepdims=True)
quirk = rng.normal(0, 1, (n_cats, K, d))
quirk *= 4.0 / np.linalg.norm(quirk, axis=2, keepdims=True)
X = voice + shared[y] + quirk[cat_idx, y] + rng.normal(0, 1, (len(y), d))

# 1. folds: cats never straddle train/test, every clip tested exactly once
folds = cv.make_folds(groups, 5, seed=1)
seen = np.concatenate([te for _, te in folds])
assert sorted(seen) == list(range(len(y)))
for tr, te in folds:
    assert not set(groups[tr]) & set(groups[te]), "cat leaked across split!"
print("[ok] folds: disjoint cats, full coverage")

# 2. leaky split must look much better than the honest one (that's the lie we avoid)
honest = cv.out_of_fold_probs(X, y, groups, 5, 3)
leaky = cv.out_of_fold_probs(X, y, groups, 5, 1, leaky=True)
a_h, a_l = (honest.argmax(2) == y).mean(), (leaky.argmax(2) == y).mean()
print(f"     honest (by cat) accuracy {a_h:.3f}   leaky (by clip) accuracy {a_l:.3f}")
assert a_l > a_h + 0.08, "leaky split should be clearly inflated"
print("[ok] leak demo: clip-level split inflates accuracy")

# 3. summary + CI sanity
s = cv.summarize(honest, y, groups, 5)
lo, hi = s["accuracy_ci95_by_cat"]
assert lo < s["accuracy_mean"] < hi and hi - lo < 0.5
assert s["accuracy_mean"] > s["chance"] + 0.05, "weak signal should still beat chance"
print(
    f"[ok] CI sane: {s['accuracy_mean']:.3f} in [{lo:.3f}, {hi:.3f}]  (chance {s['chance']:.3f})"
)

# 4. selective table: answering less often should not reduce accuracy overall
sel = s["selective"]
assert sel[0]["coverage"] == 1.0
assert all(
    sel[i]["coverage"] >= sel[i + 1]["coverage"] - 1e-9 for i in range(len(sel) - 1)
)
print("[ok] selective-prediction table: coverage shrinks as threshold rises")

# 5. pure-noise features must score ~chance (guards against accidental leakage in the code)
noise = cv.out_of_fold_probs(rng.normal(size=(len(y), d)), y, groups, 5, 3)
a_n = (noise.argmax(2) == y).mean()
assert abs(a_n - 1 / K) < 0.08, a_n
print(f"[ok] pure noise features -> {a_n:.3f} (~chance): no hidden leakage in pipeline")
# 6. balanced accuracy: a "always guess one class" model must score exactly 1/K
assert abs(cv.balanced_accuracy(y, np.zeros_like(y)) - 1 / K) < 1e-9
imb = np.r_[np.zeros(90, int), np.ones(5, int), np.full(5, 2)]
assert (
    abs(cv.balanced_accuracy(imb, np.zeros_like(imb)) - 1 / 3) < 1e-9
)  # 90% accuracy, still 1/3
blo, bhi = s["balanced_accuracy_ci95_by_cat"]
assert blo < s["balanced_accuracy_mean"] < bhi and set(s["per_cat_accuracy"]) == set(
    groups
)
print("[ok] balanced accuracy: majority-guesser scores 1/3 even at 90% plain accuracy")
print("ALL CV CHECKS PASSED")
