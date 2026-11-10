"""
Turning one meow waveform into a fixed-length feature vector.

Two options:
  "mfcc" : ~100 hand-made numbers (timbre + pitch contour). Fast, no downloads.
  "ast"  : 768 numbers from a transformer PRE-TRAINED on AudioSet (2 M sound
           clips, incl. cats). It already "knows what sounds are like", which
           helps a lot when you only have a few hundred labelled clips.
"""
from __future__ import annotations

import numpy as np
import librosa

SR = 22050


def mfcc_stats(y: np.ndarray, sr: int = SR) -> np.ndarray:
    y = np.pad(y, (0, max(0, 2048 - len(y)))).astype(np.float32)   # guard very short clips

    def ms(a):  # summarise a time-series by mean and std
        return np.r_[a.mean(axis=-1), a.std(axis=-1)].ravel()

    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=1024, hop_length=256)
    d = librosa.feature.delta(mf, width=5, mode="nearest") if mf.shape[1] >= 5 else np.zeros_like(mf)
    spec = [librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024, hop_length=256),
            librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=1024, hop_length=256),
            librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=1024, hop_length=256),
            librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=256),
            librosa.feature.rms(y=y, frame_length=1024, hop_length=256)]
    # pitch (fundamental frequency) contour: meow "melody" is a key cue
    f0 = librosa.yin(y, fmin=150, fmax=1500, sr=sr, frame_length=1024, hop_length=256)
    t = np.arange(len(f0))
    slope = np.polyfit(t, f0, 1)[0] if len(f0) > 2 else 0.0
    pitch = np.array([f0.mean(), f0.std(), f0.min(), f0.max(), f0.max() - f0.min(), slope])
    feat = np.concatenate([ms(mf), ms(d), *[ms(s) for s in spec], pitch,
                           [len(y) / sr]])                      # duration is informative too
    return np.nan_to_num(feat.astype(np.float32))


class ASTEmbedder:
    """Pre-trained Audio Spectrogram Transformer -> 768-d embedding. Needs internet
    ONCE to download the weights (~350 MB), then works offline from the HF cache."""
    MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

    def __init__(self, device: str | None = None):
        import torch
        from transformers import ASTFeatureExtractor, ASTModel
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fe = ASTFeatureExtractor.from_pretrained(self.MODEL)
        self.model = ASTModel.from_pretrained(self.MODEL).to(self.device).eval()

    def __call__(self, y: np.ndarray, sr: int = SR) -> np.ndarray:
        y16 = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000)   # AST wants 16 kHz
        inp = self.fe(y16, sampling_rate=16000, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inp)
        return out.pooler_output.squeeze(0).cpu().numpy().astype(np.float32)


_FEATURE_FNS = {}


def get_feature_fn(name: str):
    """Returns a function waveform -> vector (loads the AST model only if asked for)."""
    if name not in _FEATURE_FNS:
        if name == "mfcc":
            _FEATURE_FNS[name] = mfcc_stats
        elif name == "ast":
            _FEATURE_FNS[name] = ASTEmbedder()
        else:
            raise ValueError(f"unknown features '{name}' (use 'mfcc' or 'ast')")
    return _FEATURE_FNS[name]