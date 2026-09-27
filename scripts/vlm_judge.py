"""Human-like legibility check: a vision-language model reads the ASCII art at full resolution
and answers a 5-way multiple-choice question about what it depicts.

Needs a llama-server with a VLM (e.g. Qwen3-VL-8B-Instruct GGUF + mmproj) on --url.

usage: python scripts/vlm_judge.py --methods aic pipe net --cols 32 80 --n 60 --sources all --stage render
       python scripts/vlm_judge.py --methods aic pipe net --cols 32 80 --n 60 --sources all --stage ask
"""
import argparse
import base64
import io
import json
import random
import re
import sys
import time
from pathlib import Path

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asciiart.core import grid_rows, load_image  # noqa: E402
from asciiart.glyphs import glyph_atlas  # noqa: E402
from asciiart.viz import ids_to_pil  # noqa: E402
from scripts.eval_methods import make_method  # noqa: E402

PROMPT = ("This picture is ASCII art: text characters arranged to depict something. "
          "Which option best describes what the ASCII art depicts?\n{opts}\nAnswer with only the letter.")
PROMPT_ORIG = "Which option best describes this image?\n{opts}\nAnswer with only the letter."


CORE = ("caltech101", "imagenet", "flickr8k")
BREADTH = ("sun397", "imagenet_r", "sketch", "food101", "pets")  # scenes, renditions, sketches, food, animals


def classnames(src):
    from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES
    if src in ("imagenet", "sketch"):
        return list(IMAGENET_CLASSNAMES)
    ds = {"caltech101": "vtab-caltech101", "sun397": "sun397", "imagenet_r": "imagenet-r", "food101": "food101",
          "pets": "vtab-pets"}[src]
    return Path(f"data/raw/clip-benchmark__wds_{ds}/classnames.txt").read_text().strip().split("\n")


def questions(n, seed=0, sources=CORE):
    """n 5-way questions per source. The core sources come first with their own RNG, so their
    questions stay identical whichever other sources are added."""
    out = core_questions(n, seed) if set(CORE) & set(sources) else []
    out = [q for q in out if q["rec"]["source"] in sources]
    recs = [json.loads(l) for l in open("data/images/eval.jsonl")]
    rng = random.Random(seed + 1)
    for src in [s for s in BREADTH if s in sources]:
        names = classnames(src)
        pool = [r for r in recs if r["source"] == src and "label" in r]
        for r in rng.sample(pool, min(n, len(pool))):
            opts = [names[r["label"]]] + rng.sample([c for i, c in enumerate(names) if i != r["label"]], 4)
            order = list(range(5))
            rng.shuffle(order)
            out.append({"rec": r, "options": [opts[i] for i in order], "answer": "ABCDE"[order.index(0)]})
    return out


def core_questions(n, seed=0):
    from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES
    recs = [json.loads(l) for l in open("data/images/eval.jsonl")]
    cal_names = Path("data/raw/clip-benchmark__wds_vtab-caltech101/classnames.txt").read_text().strip().split("\n")
    rng = random.Random(seed)
    qs = []
    for src, names in [("caltech101", cal_names), ("imagenet", list(IMAGENET_CLASSNAMES))]:
        pool = [r for r in recs if r["source"] == src and names[r["label"]] not in ("background", "faces")]
        for r in rng.sample(pool, n):
            opts = [names[r["label"]]] + rng.sample([c for i, c in enumerate(names) if i != r["label"]], 4)
            qs.append((r, opts))
    fl = [r for r in recs if r["source"] == "flickr8k"]
    for r in rng.sample(fl, n):
        others = rng.sample([o for o in fl if o is not r], 4)
        qs.append((r, [r["captions"][0]] + [o["captions"][0] for o in others]))
    out = []
    for r, opts in qs:
        order = list(range(5))
        rng.shuffle(order)
        out.append({"rec": r, "options": [opts[i] for i in order], "answer": "ABCDE"[order.index(0)]})
    return out


def ask(url, img: Image.Image, prompt: str) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": prompt}]}], "temperature": 0, "max_tokens": 4}
    for attempt in range(3):
        try:
            r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=300)
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            print("retry", e)
            time.sleep(3)
    return ""


def cache_path(cache, method, cols, rec):
    tag = method.replace("/", "_").replace(":", "_")
    return Path(cache) / f"{tag}_{cols}" / Path(rec["path"]).relative_to("data/images").with_suffix(".png")


def render_all(qs, methods, cols_list, cache):
    """Render every question's ASCII art once (GPU needed; the VLM server can be stopped)."""
    atlas = glyph_atlas()
    for m in methods:
        f = None
        for cols in cols_list:
            for q in qs:
                out = cache_path(cache, m, cols, q["rec"])
                if out.exists():
                    continue
                f = f or make_method(m, "cuda")
                img = load_image(q["rec"]["path"])
                rows = grid_rows(img.width, img.height, cols)
                out.parent.mkdir(parents=True, exist_ok=True)
                ids_to_pil(f(q["rec"], img, cols, rows), atlas, scale=2).save(out)
        del f
        import torch
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["aic", "match_auto", "pipe"])
    ap.add_argument("--cols", type=int, nargs="+", default=[80])
    ap.add_argument("--n", type=int, default=50, help="questions per source")
    ap.add_argument("--sources", nargs="+", default=list(CORE),
                    help=f"question sources: {' '.join(CORE + BREADTH)} (or 'all')")
    ap.add_argument("--url", default="http://127.0.0.1:8089")
    ap.add_argument("--orig", action="store_true", help="also score the original images")
    ap.add_argument("--stage", choices=["render", "ask", "both"], default="both",
                    help="render: draw the ASCII art only (no VLM server needed); ask: query the VLM")
    ap.add_argument("--cache", default="runs/vlm_cache")
    ap.add_argument("--out", default="runs/vlm_results.jsonl")
    args = ap.parse_args()
    sources = CORE + BREADTH if args.sources == ["all"] else tuple(args.sources)
    qs = questions(args.n, sources=sources)
    if args.stage in ("render", "both"):
        render_all(qs, args.methods, args.cols, args.cache)
    if args.stage == "render":
        return
    methods = (["original"] if args.orig else []) + args.methods
    for m in methods:
        for cols in ([0] if m == "original" else args.cols):
            per = {}
            for q in qs:
                opts = "\n".join(f"{'ABCDE'[i]}. {o}" for i, o in enumerate(q["options"]))
                if m == "original":
                    pic, prompt = load_image(q["rec"]["path"]), PROMPT_ORIG.format(opts=opts)
                else:
                    pic, prompt = Image.open(cache_path(args.cache, m, cols, q["rec"])), PROMPT.format(opts=opts)
                a = ask(args.url, pic, prompt)
                mm = re.search(r"[A-E]", a.upper())
                ok = bool(mm) and mm.group(0) == q["answer"]
                per.setdefault(q["rec"]["source"], []).append(ok)
            row = {"method": m, "cols": cols, "n": args.n, **{k: sum(v) / len(v) for k, v in per.items()}}
            row["mean"] = sum(row[k] for k in per) / len(per)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            with open(args.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
