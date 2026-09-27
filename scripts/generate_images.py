"""Generate training pictures with the text->image model, in the style prompt mode uses.

Prompts combine object, animal, food, scene and doodle vocabularies (ImageNet, ImageNet-R,
SUN397, Food-101, Pets, QuickDraw, Caltech-101) with a few simple modifiers, so the network sees
the kind of pictures `ascii.py -p` converts. Output: data/generated/NNNNN.jpg + .json (prompt).
Run scripts/prepare_data.py afterwards to add them to the manifests.

usage: python scripts/generate_images.py --n 6000 --model sana-sprint
"""
import argparse
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
OUT = Path("data/generated")
RAW = Path("data/raw")


def vocab():
    from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES
    words = set(IMAGENET_CLASSNAMES)
    for ds in ("imagenet-r", "food101", "vtab-pets", "vtab-caltech101"):
        words |= set(Path(f"{RAW}/clip-benchmark__wds_{ds}/classnames.txt").read_text().strip().split("\n"))
    qd = pq.read_schema(next(RAW.glob("Xenova__quickdraw-small/data/*.parquet"))).metadata
    words |= set(json.loads(qd[b"huggingface"])["info"]["features"]["label"]["names"])
    words -= {"background", "off-center face", "centered face"}
    scenes = Path(f"{RAW}/clip-benchmark__wds_sun397/classnames.txt").read_text().strip().split("\n")
    return sorted(words), sorted(scenes)


MODIFIERS = ["", "", "", "a cute", "a big", "a small", "an old", "a red", "a blue", "a green", "a happy",
             "a sleeping", "a running", "a flying", "a smiling", "a wooden", "a golden", "a black and white"]
PEOPLE = ["a woman", "a man", "a child", "an astronaut", "a chef", "a knight", "a wizard", "a robot",
          "a surfer", "a skier", "a firefighter", "a dancer", "a guitarist", "a pirate", "a ballerina", "a cowboy"]
ACTIONS = ["riding a bicycle", "holding an umbrella", "playing soccer", "reading a book", "waving",
           "jumping", "sitting on a chair", "walking a dog", "playing the violin", "drinking coffee"]


def make_prompts(n, seed=0):
    rng = random.Random(seed)
    words, scenes = vocab()
    out = []
    for i in range(n):
        u = rng.random()
        if u < 0.6:
            mod = rng.choice(MODIFIERS)
            w = rng.choice(words)
            out.append(f"{mod} {w}".strip() if mod else f"a {w}")
        elif u < 0.8:
            out.append(f"a {rng.choice(scenes)}")
        elif u < 0.95:
            out.append(f"{rng.choice(PEOPLE)} {rng.choice(ACTIONS)}")
        else:
            out.append(f"a {rng.choice(words)} and a {rng.choice(words)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--model", default="sana-sprint")
    ap.add_argument("--size", type=int, default=768, help="saved size (px)")
    args = ap.parse_args()
    from asciiart.text2img import TextToImage
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = make_prompts(args.n)
    todo = [(i, p) for i, p in enumerate(prompts) if not (OUT / f"{i:05d}.jpg").exists()]
    print(f"{len(todo)} to generate", flush=True)
    t2i = TextToImage(args.model)
    t2i.encode([p for _, p in todo])
    for k, (i, p) in enumerate(todo):
        im = t2i(p, n=1, seed=10_000 + i)[0]
        im.resize((args.size, args.size), Image.LANCZOS).save(OUT / f"{i:05d}.jpg", quality=92)
        (OUT / f"{i:05d}.json").write_text(json.dumps({"prompt": p, "model": args.model}))
        if k % 250 == 0:
            print(k, p, flush=True)


if __name__ == "__main__":
    main()
