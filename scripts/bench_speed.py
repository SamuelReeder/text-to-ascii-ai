"""Milliseconds per image and peak VRAM: AsciiNet vs the pipeline, on 40 held-out images.

usage: python scripts/bench_speed.py
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asciiart.core import load_image  # noqa: E402
from asciiart.neural import NeuralConverter  # noqa: E402
from asciiart.pipeline import Converter, Options  # noqa: E402


def main():
    recs = [json.loads(l) for l in open("data/images/eval.jsonl")][::250][:40]
    imgs = [load_image(r["path"]) for r in recs]
    for name, make in [("AsciiNet", NeuralConverter), ("pipeline", Converter)]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        conv = make()
        load = time.time() - t0
        for cols in (32, 80, 120):
            conv.ids(imgs[0], Options(cols=cols))  # warm-up
            torch.cuda.synchronize()
            t0 = time.time()
            for im in imgs:
                conv.ids(im, Options(cols=cols))
            torch.cuda.synchronize()
            print(f"{name} {cols} cols: {(time.time() - t0) / len(imgs) * 1000:.0f} ms/image", flush=True)
        print(f"{name}: load {load:.1f} s, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        del conv


if __name__ == "__main__":
    main()
