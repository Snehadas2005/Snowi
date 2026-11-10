"""
Audio analysis = OUR filtering (audio_dsp.py)  +  THEIR analyser (meow-decoder package).

    pip install "meow-decoder @ git+https://github.com/ATHARV-IIT-BOMBAY/meow-decoder"   (Python >= 3.11)

Why combine them instead of just calling meow-decoder on the video?
  * meow-decoder was trained on ONE short meow per clip.  A 10-second video with 5 meows,
    music and silence gives it a blurred average.  We cut out each cleaned meow first and
    analyse them one by one, then combine the answers.
  * Its "is there a meow / is it a cat" check (AudioSet AST) throws away false events from
    our simple energy gate (clicks, pops).
  * Purring is LOW-pitched (~25-150 Hz).  Our bandpass (250 Hz+) deliberately removes it, so
    purr / hiss / yowl are read from the RAW audio, not the cleaned one.

LICENCE: encoder="beats" (better) is CC BY-NC-SA 4.0 = NON-COMMERCIAL ONLY.
         encoder="ast"  (default here) is BSD-3 = fine for commercial use.
         Switch with MeowDecoderAnalyzer(encoder="beats") or env SNOWI_ENCODER=beats.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from app.audio_dsp import AudioProcessor
from app.cnn_classifier import IntentPrediction, decide

logger = logging.getLogger(__name__)
SR16 = 16000
AST_WINDOW_S = 10.0  # AST looks at ~10 s at a time


class MeowDecoderAnalyzer:
    def __init__(
        self,
        encoder: Optional[str] = None,
        decoder=None,
        min_confidence: float = 0.55,
        min_margin: float = 0.15,
    ):
        self.encoder = encoder or os.environ.get("SNOWI_ENCODER", "ast")
        self._decoder = decoder  # injectable for tests
        self.min_confidence, self.min_margin = min_confidence, min_margin

    # ---- lazy load so importing this file never needs torch / downloads ----
    @property
    def decoder(self):
        if self._decoder is None:
            try:
                from meow_decoder import Decoder
            except ImportError as e:
                raise ImportError(
                    "meow-decoder is not installed. Run:\n  pip install "
                    '"meow-decoder @ git+https://github.com/ATHARV-IIT-BOMBAY/meow-decoder"\n'
                    "(needs Python >= 3.11)"
                ) from e
            self._decoder = Decoder(
                self.encoder
            )  # downloads weights from Hugging Face once
        return self._decoder

    @staticmethod
    def _to16k(y: np.ndarray, sr: int) -> np.ndarray:
        return librosa.resample(np.asarray(y, np.float32), orig_sr=sr, target_sr=SR16)

    def _signals_from_raw(self, raw16: np.ndarray) -> list[dict]:
        """Purr / hiss / yowl over the whole RAW clip, 10 s window at a time (max over windows)."""
        win = int(AST_WINDOW_S * SR16)
        best: dict[str, dict] = {}
        for i in range(0, max(len(raw16), 1), win):
            seg = raw16[i : i + win]
            if len(seg) < SR16 // 4:
                continue
            emb, sounds = self.decoder.embed(seg)
            for s in self.decoder.rank(emb, sounds)["signals"]:
                if s["key"] not in best or s["p"] > best[s["key"]]["p"]:
                    best[s["key"]] = s
        return list(best.values())

    def analyze_file(
        self, path: str | Path, proc: Optional[AudioProcessor] = None
    ) -> dict:
        proc = proc or AudioProcessor()
        raw = proc.load_audio(path)  # 22.05 kHz mono, untouched
        waves, events = proc.extract_meow_waveforms(path)  # OUR cleaning + meow cutting
        notes: list[str] = []

        signals = self._signals_from_raw(self._to16k(raw, proc.sr))

        probs, kept, labels, lines = [], [], None, {}
        for w, e in zip(waves, events):
            w16 = self._to16k(w, proc.sr)
            if len(w16) < SR16 // 2:  # very short: pad to 0.5 s
                w16 = np.pad(w16, (0, SR16 // 2 - len(w16)))
            emb, sounds = self.decoder.embed(w16)
            r = self.decoder.rank(emb, sounds)
            if not (r["is_cat"] and r["has_meow"]):
                notes.append(
                    f"Dropped event at {e.start:.1f}-{e.end:.1f}s: no cat meow confirmed."
                )
                continue
            by_key = {x["key"]: x for x in r["results"]}
            keys = sorted(by_key)
            labels = labels or [by_key[k]["title"] for k in keys]
            lines = {by_key[k]["title"]: by_key[k]["line"] for k in keys}
            probs.append([by_key[k]["p"] for k in keys])
            kept.append((round(e.start, 2), round(e.end, 2)))

        if probs:
            pred = decide(np.array(probs), labels, self.min_confidence, self.min_margin)
            if pred.label:
                pred.text = lines.get(pred.label, pred.label)
        else:
            pred = IntentPrediction(
                status="no_meow", message="No confirmed meow in this clip."
            )
        if any(s["key"] == "Purr" for s in signals):
            notes.append(
                "Purring heard: usually relaxed, but cats also purr when stressed or hurt."
            )
        return {
            "intent": pred.to_dict(),
            "signals": signals,
            "meow_segments_used": kept,
            "events_found_by_filter": len(events),
            "encoder": self.encoder,
            "notes": notes,
        }
