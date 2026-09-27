"""Run the BiRefNet subject matte once per dataset image and cache it as an 8-bit PNG.

Training labels come from the pipeline (asciiart/pipeline.py), whose only expensive step is this
matte; with it cached, the pipeline labels any crop, width or option combination on the fly.

usage: python scripts/cache_mattes.py [--train-per-source 3000] [--eval-per-source 30]
"""
import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asciiart.core import load_image  # noqa: E402
from asciiart.subject import MEAN, STD, SubjectMatte  # noqa: E402

MATTES = Path("data/mattes")


def matte_path(image_path: str) -> Path:
    return MATTES / Path(image_path).relative_to("data/images").with_suffix(".png")


class Images(torch.utils.data.Dataset):
    def __init__(self, paths, size):
        self.paths, self.size = paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_image(self.paths[i])
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255).permute(2, 0, 1)[None]
        xi = F.interpolate(x, size=(self.size, self.size), mode="bilinear", antialias=True)[0]
        return ((xi - MEAN[0]) / STD[0]), torch.tensor([img.height, img.width]), i


def subset(split, per_source, overrides=None):
    """A fixed random subset of each source (the matte costs ~0.1 s per image at 1024 px, so
    training uses up to `per_source` images of each domain rather than all 180k)."""
    by = {}
    for line in open(f"data/images/{split}.jsonl"):
        r = json.loads(line)
        by.setdefault(r["source"], []).append(r["path"])
    out = []
    for src, paths in by.items():
        n = (overrides or {}).get(src, per_source)
        out += sorted(random.Random(src).sample(sorted(paths), min(n, len(paths))))
    return out


def save_png(arr, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.png")  # written atomically: an interrupted run leaves no bad file
    Image.fromarray(arr).save(tmp)
    tmp.rename(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "eval"])
    ap.add_argument("--train-per-source", type=int, default=3000)
    ap.add_argument("--eval-per-source", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()
    paths = []
    if "train" in args.splits:
        paths += subset("train", args.train_per_source, {"imagenet": 8000, "generated": 10000})
    if "eval" in args.splits:
        paths += subset("eval", args.eval_per_source)
    todo = [p for p in paths if not matte_path(p).exists()]
    random.Random(0).shuffle(todo)  # interleave sources, so a partial cache still covers every domain
    print(f"{len(todo)} of {len(paths)} images need a matte", flush=True)
    if not todo:
        return
    torch.backends.cudnn.benchmark = True  # fixed 1024x1024 input
    sm = SubjectMatte("cuda")
    dl = torch.utils.data.DataLoader(Images(todo, sm.size), batch_size=args.batch, num_workers=4)
    t0 = time.time()
    with ThreadPoolExecutor(2) as writer:  # PNG encoding off the GPU loop
        for n, (x, hw, idx) in enumerate(dl):
            with torch.no_grad():
                m = sm.model(x.to("cuda", sm.dtype))[-1].float().sigmoid()
            for k in range(len(idx)):
                h, w = hw[k].tolist()
                mk = F.interpolate(m[k:k + 1], size=(h, w), mode="bilinear")[0, 0]
                arr = (mk.clamp(0, 1) * 255).round().byte().cpu().numpy()
                writer.submit(save_png, arr, matte_path(todo[int(idx[k])]))
            if n % 500 == 0:
                print(n * args.batch, "/", len(todo), f"{(time.time() - t0) / max(1, n * args.batch):.3f} s/img",
                      flush=True)


if __name__ == "__main__":
    main()
