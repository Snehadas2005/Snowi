import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import numpy as np

from app.audio_dsp import AudioProcessor

FILES = ["blind_cat_video.mp4", "kitten_meowing_audio.mp3"]


def find_samples_dir() -> Path:
    """Walk upward from this file until a folder named test_samples holds our files."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "test_samples"
        if all((cand / f).is_file() for f in FILES):
            return cand
    raise FileNotFoundError(
        f"Could not find a 'test_samples' folder containing {FILES} "
        f"anywhere above {here}"
    )


SAMPLES = find_samples_dir()
OUT = SAMPLES / "out"
print("Using samples from:", SAMPLES)

proc = AudioProcessor()

for name in FILES:
    src = SAMPLES / name
    t0 = time.time()
    res = proc.save_cleaned_media(src, OUT)
    dt = time.time() - t0
    print(f"\n{name}\n  -> {res['output']}   ({dt:.2f}s)")
    print("  meow events (s):", [(round(e.start, 2), round(e.end, 2)) for e in res["events"]])

    tens = proc.process_file_to_tensor_input(src)
    print("  model input:", tens["spectrograms"].shape, "(n_meows, 1, mels, frames)")

    y = proc.load_audio(src)
    clean, _ = proc.clean_waveform(y)
    fig, ax = plt.subplots(2, 1, figsize=(11, 6))
    ref = np.abs(librosa.stft(y, n_fft=2048, hop_length=256)).max()
    for a, x, title in zip(ax, [y, clean], ["before", "after"]):
        S = librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=2048, hop_length=256)), ref=ref)
        librosa.display.specshow(S, sr=proc.sr, hop_length=256, x_axis="time",
                                 y_axis="log", ax=a, fmax=10000, vmin=-80, vmax=0)
        a.set_title(title)
    plt.tight_layout()
    plt.savefig(OUT / f"compare_{Path(name).stem}.png", dpi=90)
    plt.close()