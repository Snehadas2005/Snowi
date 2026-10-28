"""
Cat-vocalisation DSP front-end (Part 1: video/audio -> clean meow audio -> model input).

Design notes
------------
* Background music / hum is NOT removable with a bandpass or stationary
  denoiser: it lives in the same 250-8000 Hz range as a meow.  What *does*
  differ is behaviour over time:
      - music / electronic tones / hum  -> flat, sustained (> ~0.6 s) lines
      - meows                           -> short (0.2-1 s), gliding harmonics
  So we (1) suppress spectral components that persist longer than a meow
  (temporal morphological opening), (2) gate everything outside detected
  meow events, (3) optionally let a neural sound-event detector (YAMNet / PANNs,
  AudioSet class "Meow") decide where the meows are -- see `event_detector`.
* `clean_waveform()` is LENGTH-PRESERVING so the cleaned audio can be muxed
  back onto the original video without drifting out of sync.  Trimming /
  cropping happens only when building model tensors.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
from scipy.ndimage import (
    maximum_filter,
    maximum_filter1d,
    minimum_filter1d,
    uniform_filter1d,
)
from scipy.signal import butter, sosfiltfilt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


@dataclass
class MeowEvent:
    start: float  # seconds
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


# (waveform, sample_rate) -> list[(start_s, end_s)]  e.g. wrapper around YAMNet/PANNs
EventDetector = Callable[[np.ndarray, int], list]


class AudioProcessor:
    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 256,
        f_min: float = 250.0,
        f_max: float = 8000.0,
        target_duration_sec: float = 1.5,
        # --- persistent-tone (music/hum) suppression ---
        tone_window_sec: float = 0.6,   # anything steadier than this is "not a meow"
        tone_alpha: float = 1.2,        # higher = gentler, lower = more aggressive
        tone_floor: float = 0.05,       # min mask gain (avoids musical-noise artefacts)
        # --- meow event gating ---
        gate_margin_db: float = 9.0,    # event must exceed noise floor by this much
        min_event_sec: float = 0.15,
        merge_gap_sec: float = 0.12,
        event_pad_sec: float = 0.08,
        event_detector: Optional[EventDetector] = None,
    ) -> None:
        self.sr = sample_rate
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.target_duration_sec = target_duration_sec
        self.tone_window_sec = tone_window_sec
        self.tone_alpha = tone_alpha
        self.tone_floor = tone_floor
        self.gate_margin_db = gate_margin_db
        self.min_event_sec = min_event_sec
        self.merge_gap_sec = merge_gap_sec
        self.event_pad_sec = event_pad_sec
        self.event_detector = event_detector

    # ------------------------------------------------------------------ I/O
    @staticmethod
    def is_video(path: Path) -> bool:
        return path.suffix.lower() in SUPPORTED_VIDEO_EXTS

    def load_audio(self, file_path: str | Path) -> np.ndarray:
        """Decode any audio/video file to mono float32 at self.sr (ffmpeg pipe)."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Media file not found: {path}")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH")
        cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
               "-ar", str(self.sr), "-f", "f32le", "pipe:1"]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.decode()}")
        y = np.frombuffer(proc.stdout, dtype=np.float32).copy()
        if y.size == 0:
            raise RuntimeError(f"No audio stream found in {path}")
        return y

    # ------------------------------------------------------ cleaning stages
    def _bandpass(self, y: np.ndarray) -> np.ndarray:
        # zero-phase (sosfiltfilt) -> no time shift, unlike sosfilt
        nyq = 0.5 * self.sr
        sos = butter(4, [self.f_min / nyq, min(self.f_max, 0.99 * nyq) / nyq],
                     btype="bandpass", output="sos")
        return sosfiltfilt(sos, y).astype(np.float32)

    def suppress_persistent_tones(self, y: np.ndarray) -> np.ndarray:
        """Remove sustained flat spectral lines (music notes, hum, beeps).

        Temporal opening (min-filter then max-filter along time) keeps only
        structure that lasts longer than `tone_window_sec`.  That "persistent"
        layer is subtracted with a soft mask.  Linear time, ~1 s for 10 s audio.
        """
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop)
        S = np.abs(D)
        w = int(self.tone_window_sec * self.sr / self.hop) | 1
        Ss = uniform_filter1d(maximum_filter(S, size=(3, 1)), 3, axis=1)
        persistent = maximum_filter1d(minimum_filter1d(Ss, w, axis=1), w, axis=1)
        mask = np.clip((S - self.tone_alpha * persistent) / (S + 1e-8),
                       self.tone_floor, 1.0)
        mask = uniform_filter1d(mask, 5, axis=1)
        return librosa.istft(D * mask, hop_length=self.hop, length=len(y)).astype(np.float32)

    def _frame_db(self, y: np.ndarray) -> np.ndarray:
        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop)) ** 2
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        # meows carry harmonics well above 2 kHz; leftover music lines / rumble are
        # narrow or low, so gating on this band separates them from meows
        band = (freqs >= 1800) & (freqs <= 7000)
        return 10 * np.log10(S[band].sum(axis=0) + 1e-10)

    def detect_events(self, y_tone_free: np.ndarray, y_original: np.ndarray) -> list[MeowEvent]:
        """Find meow segments.  Uses the neural detector if supplied, else an energy gate."""
        dur = len(y_original) / self.sr
        if self.event_detector is not None:
            spans = self.event_detector(y_original, self.sr)
            return [MeowEvent(max(0, a), min(dur, b)) for a, b in spans]

        db = self._frame_db(y_tone_free)
        db = uniform_filter1d(db, 5)
        floor = np.percentile(db, 20)
        hi, lo = floor + self.gate_margin_db, floor + self.gate_margin_db - 3.0
        active = np.zeros_like(db, dtype=bool)
        on = False
        for i, v in enumerate(db):               # hysteresis threshold
            on = (v > lo) if on else (v > hi)
            active[i] = on
        fps = self.sr / self.hop
        events: list[MeowEvent] = []
        i = 0
        while i < len(active):
            if active[i]:
                j = i
                while j < len(active) and active[j]:
                    j += 1
                events.append(MeowEvent(i / fps, j / fps))
                i = j
            else:
                i += 1
        merged: list[MeowEvent] = []
        for e in events:
            if merged and e.start - merged[-1].end < self.merge_gap_sec:
                merged[-1].end = e.end
            else:
                merged.append(e)
        merged = [e for e in merged if e.duration >= self.min_event_sec]
        return [MeowEvent(max(0, e.start - self.event_pad_sec),
                          min(dur, e.end + self.event_pad_sec)) for e in merged]

    def _apply_gate(self, y: np.ndarray, events: list[MeowEvent]) -> np.ndarray:
        gain = np.zeros(len(y), dtype=np.float32)
        for e in events:
            gain[int(e.start * self.sr): int(e.end * self.sr)] = 1.0
        fade = int(0.02 * self.sr)                # 20 ms fades -> no clicks
        gain = uniform_filter1d(gain, max(fade, 1))
        return y * gain

    def clean_waveform(self, y: np.ndarray, gate: bool = True) -> tuple[np.ndarray, list[MeowEvent]]:
        """Length-preserving clean-up. Returns (clean_audio, meow_events)."""
        y1 = self._bandpass(y)
        y2 = self.suppress_persistent_tones(y1)
        y3 = nr.reduce_noise(y=y2, sr=self.sr, stationary=False, prop_decrease=0.7,
                             n_fft=1024, n_jobs=1).astype(np.float32)
        events = self.detect_events(y2, y)
        out = self._apply_gate(y3, events) if gate and events else y3
        peak = np.max(np.abs(out)) + 1e-9
        if peak > 0.99:                           # only protect from clipping; keep intensity
            out = out * (0.99 / peak)
        return out.astype(np.float32), events

    # ------------------------------------------------------ model features
    def _fit_length(self, y: np.ndarray) -> np.ndarray:
        n = int(self.target_duration_sec * self.sr)
        if len(y) >= n:
            return y[:n]
        return np.pad(y, (0, n - len(y)))

    def log_mel(self, y: np.ndarray) -> np.ndarray:
        m = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop * 2,
            n_mels=self.n_mels, fmin=self.f_min, fmax=self.f_max)
        db = librosa.power_to_db(m, ref=np.max)
        return ((db - db.mean()) / (db.std() + 1e-6)).astype(np.float32)

    def process_file_to_tensor_input(self, file_path: str | Path) -> dict:
        """One spectrogram PER MEOW (not one center-crop for the whole clip)."""
        y = self.load_audio(file_path)
        clean, events = self.clean_waveform(y)
        specs = []
        for e in events:
            seg = clean[int(e.start * self.sr): int(e.end * self.sr)]
            specs.append(self.log_mel(self._fit_length(seg)))
        if not specs:                              # no meow found -> flag it, don't guess
            logger.warning("No meow events detected in %s", file_path)
        return {
            "spectrograms": np.stack(specs)[:, None] if specs else np.empty((0, 1, self.n_mels, 0), np.float32),
            "events": [(round(e.start, 3), round(e.end, 3)) for e in events],
            "sample_rate": self.sr,
        }

    # ------------------------------------------------------------- output
    def save_cleaned_media(self, src: str | Path, out_dir: str | Path,
                           stem: Optional[str] = None) -> dict:
        """Video in -> .mp4 out (original picture + cleaned audio).
        Audio in -> .wav out.  Also returns detected events."""
        src = Path(src)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = stem or f"cleaned_{src.stem}"

        y = self.load_audio(src)
        clean, events = self.clean_waveform(y)

        wav_path = out_dir / f"{stem}.wav"
        sf.write(str(wav_path), clean, self.sr)
        if not self.is_video(src):
            return {"output": str(wav_path), "events": events}

        mp4_path = out_dir / f"{stem}.mp4"
        cmd = ["ffmpeg", "-y", "-v", "error",
               "-i", str(src), "-i", str(wav_path),
               "-map", "0:v:0", "-map", "1:a:0",
               "-c:v", "copy",                    # no re-encode: instant, lossless picture
               "-c:a", "aac", "-b:a", "192k",
               "-shortest", "-movflags", "+faststart", str(mp4_path)]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg mux failed: {proc.stderr.decode()}")
        return {"output": str(mp4_path), "audio_wav": str(wav_path), "events": events}