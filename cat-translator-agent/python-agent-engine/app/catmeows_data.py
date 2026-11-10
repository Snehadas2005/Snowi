"""
Reading the CatMeows dataset + the three "cleaning" variants we want to compare.

File name  B_ANI01_MC_FN_SIM01_101.wav
           ^ ^
           | +-- cat ID  (we split train/test by THIS)
           +---- situation: B brushing, I isolation, F waiting for food
(Please eyeball a few real file names to confirm this layout.)
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from app.audio_dsp import AudioProcessor

LABELS = ["brushing", "isolation", "waiting_for_food"]
CONTEXT = {"B": "brushing", "I": "isolation", "F": "waiting_for_food"}
CLEANING_MODES = ("raw", "bandpass", "full")


def parse_name(path: Path) -> tuple[str, str]:
    parts = path.stem.split("_")
    return CONTEXT[parts[0].upper()], parts[1]


def list_clips(data_dir: str | Path):
    """Returns (files, y, groups): y = class index, groups = cat ID per clip."""
    files, ys, groups = [], [], []
    for f in sorted(Path(data_dir).rglob("*.wav")):
        try:
            lab, cat = parse_name(f)
        except (KeyError, IndexError):
            continue  # not a CatMeows-style name
        files.append(f)
        ys.append(LABELS.index(lab))
        groups.append(cat)
    if not files:
        raise FileNotFoundError(f"No CatMeows-style .wav files under {data_dir}")
    return files, np.array(ys), np.array(groups)


def load_clip(proc: AudioProcessor, path: Path, mode: str = "full") -> np.ndarray:
    """One meow waveform (22.05 kHz) under a chosen cleaning mode.

    raw      : decode only                       (control: no cleaning at all)
    bandpass : 250-8000 Hz filter only           (gentle)
    full     : bandpass + tone suppression + denoise, WITHOUT event gating
               (what the app pipeline does; clips are already single meows)
    """
    if mode not in CLEANING_MODES:
        raise ValueError(f"mode must be one of {CLEANING_MODES}")
    y = proc.load_audio(path)
    if mode == "bandpass":
        y = proc._bandpass(y)
    elif mode == "full":
        y, _ = proc.clean_waveform(y, gate=False)
    trimmed, _ = librosa.effects.trim(y, top_db=30)  # drop leading/trailing silence
    return trimmed if len(trimmed) > int(0.1 * proc.sr) else y
