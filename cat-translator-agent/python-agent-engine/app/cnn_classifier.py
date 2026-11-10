"""
Acoustic intent classifier for cat vocalisations (Part 1, step 1).

Input  : log-mel spectrograms from audio_dsp.AudioProcessor
         shape (n_meows, 1, n_mels, frames)  -- or a single (1, n_mels, frames)
         `frames` may be ANY length (adaptive pooling), so 65 or 130 both work.
Output : an IntentPrediction with a label, a confidence and the full probability
         table -- or an explicit "uncertain" result.  It never fakes confidence.

Honest-by-design rules
----------------------
1. An untrained network outputs arbitrary numbers.  If no trained checkpoint is
   loaded, `predict()` returns status="untrained" and NO label.
2. If the best class is not clearly ahead (low confidence or small margin), the
   status is "uncertain" instead of a guessed label.
3. Several meows in one clip are combined (mean of probabilities) into one
   clip-level answer, while per-meow results are kept.
4. Probabilities are temperature-scaled (fit on validation data) so that
   "0.8 confidence" is closer to "right ~80% of the time".
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Label sets.  Keep the label set tied to what the DATA can support.

INTENT_LABELS_TARGET = ["hungry_food", "pain_discomfort", "greeting_attention", "defensive_angry"]
# The only public labelled cat-meow dataset (CatMeows, Zenodo, 3 situations):
INTENT_LABELS_CATMEOWS = ["brushing", "isolation", "waiting_for_food"]

# Human-readable text shown to the user for each label.
INTENT_TEXT = {
    "hungry_food": "I'm hungry / I want food.",
    "pain_discomfort": "Something hurts or I feel uncomfortable.",
    "greeting_attention": "Hello! Please pay attention to me.",
    "defensive_angry": "Back off, I feel threatened.",
    "brushing": "That feels nice (relaxed, being groomed).",
    "isolation": "I'm alone and I want company.",
    "waiting_for_food": "I'm waiting for my food.",
}


@dataclass
class IntentPrediction:
    status: str                         # "ok" | "uncertain" | "untrained" | "no_meow"
    label: Optional[str] = None         # only set when status == "ok"
    text: Optional[str] = None
    confidence: float = 0.0             # calibrated probability of `label`
    probabilities: dict = field(default_factory=dict)   # clip-level
    per_meow: list = field(default_factory=list)        # [{"label","confidence"}, ...]
    n_meows: int = 0
    message: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# Pure-numpy decision logic (no torch needed -> unit-testable anywhere)

def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / max(temperature, 1e-6)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def decide(
    probs_per_meow: np.ndarray,
    labels: Sequence[str],
    min_confidence: float = 0.55,
    min_margin: float = 0.15,
) -> IntentPrediction:
    """probs_per_meow: (n_meows, n_classes) -> clip-level IntentPrediction."""
    probs_per_meow = np.atleast_2d(probs_per_meow)
    n = probs_per_meow.shape[0]
    if n == 0:
        return IntentPrediction(status="no_meow", message="No meow detected in this clip.")

    clip_probs = probs_per_meow.mean(axis=0)
    order = np.argsort(clip_probs)[::-1]
    top, second = clip_probs[order[0]], clip_probs[order[1]] if len(order) > 1 else 0.0
    per_meow = [
        {"label": labels[int(p.argmax())], "confidence": round(float(p.max()), 4)}
        for p in probs_per_meow
    ]
    table = {lab: round(float(p), 4) for lab, p in zip(labels, clip_probs)}

    if top < min_confidence or (top - second) < min_margin:
        return IntentPrediction(
            status="uncertain", confidence=round(float(top), 4), probabilities=table,
            per_meow=per_meow, n_meows=n,
            message=f"Not sure: best guess '{labels[int(order[0])]}' at {top:.0%} "
                    f"is too close to the alternatives.")
    lab = labels[int(order[0])]
    return IntentPrediction(
        status="ok", label=lab, text=INTENT_TEXT.get(lab, lab),
        confidence=round(float(top), 4), probabilities=table,
        per_meow=per_meow, n_meows=n)


def fit_temperature(logits: np.ndarray, y: np.ndarray, grid=np.linspace(0.3, 10.0, 195)) -> float:
    """Pick the temperature that minimises validation NLL (simple grid search)."""
    best_t, best_nll = 1.0, np.inf
    for t in grid:
        p = softmax(logits, t)
        nll = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return best_t


# Torch model + wrapper (imported lazily so the numpy logic above stays usable)

def build_model(num_classes: int, backbone: str = "small", pretrained: bool = False):
    """
    backbone:
      "small"              custom 4-block CNN (~0.3 M params).  Best default for
                           a few hundred training clips; fast on CPU.
      "resnet18"           torchvision ResNet18, 1-channel stem.
      "mobilenet_v3_small" torchvision MobileNetV3-Small, 1-channel stem.
    pretrained=True loads ImageNet weights (needs internet once) and folds the
    3-channel stem into 1 channel by summing its filters.
    """
    import torch
    import torch.nn as nn

    if backbone == "small":
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))
        return nn.Sequential(
            block(1, 16), block(16, 32), block(32, 64), block(64, 128),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes))

    import torchvision.models as tvm
    if backbone == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        w = m.conv1.weight.data.sum(dim=1, keepdim=True)
        m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        if pretrained:
            m.conv1.weight.data = w
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
        return m
    if backbone == "mobilenet_v3_small":
        m = tvm.mobilenet_v3_small(
            weights=tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        old = m.features[0][0]
        new = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        if pretrained:
            new.weight.data = old.weight.data.sum(dim=1, keepdim=True)
        m.features[0][0] = new
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    raise ValueError(f"Unknown backbone: {backbone}")


class CatIntentClassifier:
    """Wrapper: spectrogram tensor in -> IntentPrediction out."""

    def __init__(
        self,
        labels: Sequence[str] = tuple(INTENT_LABELS_CATMEOWS),
        backbone: str = "small",
        checkpoint: Optional[str | Path] = None,
        device: Optional[str] = None,
        min_confidence: float = 0.55,
        min_margin: float = 0.15,
    ) -> None:
        import torch
        self.torch = torch
        self.labels = list(labels)
        self.backbone = backbone
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.temperature = 1.0
        self.trained = False
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(len(self.labels), backbone).to(self.device).eval()
        if checkpoint is not None and Path(checkpoint).is_file():
            self.load(checkpoint)
        else:
            logger.warning("No trained checkpoint loaded: predictions will be refused "
                           "(status='untrained') until you train/load one.")

    # ---------------------------------------------------------- persistence
    def save(self, path: str | Path, extra: Optional[dict] = None) -> None:
        self.torch.save({
            "state_dict": self.model.state_dict(),
            "labels": self.labels,
            "backbone": self.backbone,
            "temperature": self.temperature,
            "extra": extra or {},
        }, str(path))

    def load(self, path: str | Path) -> None:
        ck = self.torch.load(str(path), map_location=self.device)
        if ck["labels"] != self.labels or ck["backbone"] != self.backbone:
            # rebuild to match the checkpoint rather than silently mis-mapping labels
            self.labels, self.backbone = ck["labels"], ck["backbone"]
            self.model = build_model(len(self.labels), self.backbone).to(self.device)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.temperature = float(ck.get("temperature", 1.0))
        self.trained = True
        logger.info("Loaded %s (%s) with labels %s, T=%.2f",
                    path, self.backbone, self.labels, self.temperature)

    # ------------------------------------------------------------ inference
    def _prep(self, spec: np.ndarray):
        x = np.asarray(spec, dtype=np.float32)
        if x.ndim == 3:                  # (1, mels, frames) -> one meow
            x = x[None]
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected (n,1,mels,frames) or (1,mels,frames), got {x.shape}")
        return self.torch.from_numpy(x).to(self.device)

    def predict_logits(self, spec: np.ndarray) -> np.ndarray:
        x = self._prep(spec)
        with self.torch.no_grad():
            return self.model(x).cpu().numpy()

    def predict(self, spec: np.ndarray) -> IntentPrediction:
        """spec: output['spectrograms'] from AudioProcessor.process_file_to_tensor_input."""
        spec = np.asarray(spec)
        if spec.size == 0 or spec.shape[0] == 0:
            return IntentPrediction(status="no_meow", message="No meow detected in this clip.")
        if not self.trained:
            return IntentPrediction(
                status="untrained", n_meows=int(spec.shape[0]),
                message="Model has no trained weights yet; refusing to output an intent.")
        probs = softmax(self.predict_logits(spec), self.temperature)
        return decide(probs, self.labels, self.min_confidence, self.min_margin)