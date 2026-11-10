"""Download only cat .wav files from mmoreaux/audio-cats-and-dogs into data/Cat-train."""
import os, shutil
from pathlib import Path
import kagglehub

# Point kagglehub at the credential file
os.environ.setdefault("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))

print("Downloading dataset mmoreaux/audio-cats-and-dogs …")
path = kagglehub.dataset_download("mmoreaux/audio-cats-and-dogs")
print(f"Dataset cached at: {path}")

dest = Path(__file__).parent / "data" / "Cat-train"
dest.mkdir(parents=True, exist_ok=True)

count = 0
for f in sorted(Path(path).rglob("*.wav")):
    name_lower = f.name.lower()
    parent_lower = str(f.parent).lower()
    if "cat" in name_lower or "cat" in parent_lower:
        target = dest / f.name
        shutil.copy2(f, target)
        count += 1
        print(f"  copied: {f.name}")

print(f"\nDone — copied {count} cat .wav files → {dest}")

