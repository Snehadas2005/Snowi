"""
Deployable classifier built on feature vectors (hand-made MFCC set or pretrained AST
embeddings) + logistic regression.  Same output type as the CNN wrapper
(IntentPrediction), so the rest of the app doesn't care which one is behind it.

Train & save (after cv_eval tells you which setting is best):
    python -m app.embedding_classifier --data path/to/CatMeows --features ast \
           --cleaning full --out models/cat_intent_embed.joblib
Use:
    clf = EmbeddingIntentClassifier("models/cat_intent_embed.joblib")
    waves, _ = AudioProcessor().extract_meow_waveforms("clip.mp4")
    print(clf.predict_waveforms(waves).to_dict())
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import joblib
import numpy as np

from app.cnn_classifier import IntentPrediction, decide, fit_temperature, softmax
from app.cv_eval import make_model, out_of_fold_probs


def fit_and_save(X, y, groups, labels: Sequence[str], features: str, cleaning: str,
                 out_path: str | Path, C: float = 0.5, seed: int = 0) -> float:
    """1) fit on ALL cats  2) learn a calibration temperature from cat-wise
    out-of-fold predictions (so confidences are honest)  3) save one file."""
    pipe = make_model(C).fit(X, y)
    oof = out_of_fold_probs(X, y, groups, 5, 1, C, seed, n_classes=len(labels))[0]
    T = fit_temperature(np.log(oof + 1e-9), y)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "labels": list(labels), "temperature": T,
                 "features": features, "cleaning": cleaning,
                 "oof_accuracy": float((oof.argmax(1) == y).mean())}, out_path)
    return T


class EmbeddingIntentClassifier:
    def __init__(self, checkpoint: Optional[str | Path] = None,
                 min_confidence: float = 0.55, min_margin: float = 0.15):
        self.min_confidence, self.min_margin = min_confidence, min_margin
        self.trained = False
        if checkpoint is not None and Path(checkpoint).is_file():
            ck = joblib.load(checkpoint)
            self.pipe, self.labels, self.T = ck["pipeline"], ck["labels"], ck["temperature"]
            self.feat_name, self.cleaning, self.meta = ck["features"], ck["cleaning"], ck
            self.trained = True

    def predict_waveforms(self, waves: Sequence[np.ndarray]) -> IntentPrediction:
        if len(waves) == 0:
            return IntentPrediction(status="no_meow", message="No meow detected in this clip.")
        if not self.trained:
            return IntentPrediction(status="untrained", n_meows=len(waves),
                                    message="No trained model loaded; refusing to output an intent.")
        from app.features import get_feature_fn
        fn = get_feature_fn(self.feat_name)
        X = np.stack([fn(w) for w in waves])
        p = np.zeros((len(waves), len(self.labels)))
        p[:, self.pipe.classes_] = self.pipe.predict_proba(X)
        p = softmax(np.log(p + 1e-9), self.T)          # temperature calibration
        return decide(p, self.labels, self.min_confidence, self.min_margin)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--features", default="mfcc", choices=["mfcc", "ast"])
    ap.add_argument("--cleaning", default="full", choices=["raw", "bandpass", "full"])
    ap.add_argument("--out", default="models/cat_intent_embed.joblib")
    ap.add_argument("--C", type=float, default=0.5)
    a = ap.parse_args()
    from app.catmeows_data import LABELS, list_clips
    from app.cv_eval import build_features
    files, y, groups = list_clips(a.data)
    X = build_features(files, a.features, a.cleaning, Path("reports/feature_cache"))
    T = fit_and_save(X, y, groups, LABELS, a.features, a.cleaning, a.out, a.C)
    print(f"saved {a.out}  (calibration temperature {T:.2f})")