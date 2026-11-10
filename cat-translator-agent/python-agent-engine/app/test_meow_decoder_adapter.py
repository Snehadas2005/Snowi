"""Tests the glue code with a FAKE decoder (no model download).  Run: python -m app.test_meow_decoder_adapter
Uses test_samples/blind_cat_video.mp4 found automatically."""

from pathlib import Path

import numpy as np

from app.meow_decoder_adapter import MeowDecoderAnalyzer


def find_video() -> Path:
    for parent in Path(__file__).resolve().parents:
        p = parent / "test_samples" / "blind_cat_video.mp4"
        if p.is_file():
            return p
    raise FileNotFoundError("test_samples/blind_cat_video.mp4 not found")


class FakeDecoder:
    """Mimics meow_decoder.Decoder.embed/rank. Behaviour is scripted per test."""

    def __init__(self, p_food=0.8, purr=0.0, reject_nth=None):
        self.p_food, self.purr, self.reject_nth, self.calls = (
            p_food,
            purr,
            reject_nth,
            0,
        )

    def embed(self, audio):
        self.calls += 1
        return np.array([len(audio)], np.float32), {
            "Cat": 0.9,
            "Meow": 0.9,
            "Purr": self.purr,
            "Caterwaul": 0.0,
            "Hiss": 0.0,
        }

    def rank(self, emb, sounds):
        rest = (1 - self.p_food) / 2
        res = [
            {
                "key": "F",
                "title": "Food, please",
                "line": "I'm hungry.",
                "p": self.p_food,
            },
            {"key": "B", "title": "Enjoying the attention", "line": "Nice.", "p": rest},
            {
                "key": "I",
                "title": "Stressed somewhere new",
                "line": "Where am I?",
                "p": rest,
            },
        ]
        sig = (
            [
                {
                    "key": "Purr",
                    "title": "Purring",
                    "meaning": "relaxed",
                    "p": sounds["Purr"],
                }
            ]
            if sounds["Purr"] >= 0.3
            else []
        )
        return {
            "is_cat": True,
            "has_meow": True,
            "sounds": sounds,
            "signals": sig,
            "results": res,
        }


v = find_video()

r = MeowDecoderAnalyzer(decoder=FakeDecoder(0.8)).analyze_file(v)
assert r["events_found_by_filter"] == 5 and len(r["meow_segments_used"]) == 5, r
assert r["intent"]["status"] == "ok" and r["intent"]["label"] == "Food, please", r[
    "intent"
]
print(
    "[ok] 5 meows cut by our filter -> analysed one by one -> combined:",
    r["intent"]["label"],
    r["intent"]["confidence"],
)


class RejectOne(FakeDecoder):  # their AST says "no meow" for one cut
    def rank(self, emb, sounds):
        out = super().rank(emb, sounds)
        if emb[0] < 16000 * 0.78 and not getattr(self, "done", False):
            self.done = True
            out["has_meow"] = False
        return out


r = MeowDecoderAnalyzer(decoder=RejectOne()).analyze_file(v)
assert len(r["meow_segments_used"]) == 4 and any("Dropped" in n for n in r["notes"]), r[
    "notes"
]
print("[ok] false events are dropped when their model says 'no meow'")

r = MeowDecoderAnalyzer(decoder=FakeDecoder(p_food=0.36)).analyze_file(v)
assert r["intent"]["status"] == "uncertain" and r["intent"]["label"] is None, r[
    "intent"
]
print("[ok] near-flat probabilities -> 'uncertain', no label")

r = MeowDecoderAnalyzer(decoder=FakeDecoder(0.8, purr=0.7)).analyze_file(v)
assert any(s["key"] == "Purr" for s in r["signals"]) and any(
    "Purring" in n for n in r["notes"]
)
print("[ok] purr read from RAW audio and reported alongside the intent")

try:
    import meow_decoder  # noqa: F401
except ImportError:
    try:
        MeowDecoderAnalyzer().decoder
    except ImportError as e:
        assert "pip install" in str(e)
        print("[ok] helpful message when meow-decoder isn't installed")
print("ALL ADAPTER CHECKS PASSED")
