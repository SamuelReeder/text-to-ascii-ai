"""Make a contact sheet comparing methods on a fixed, diverse set of eval images."""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asciiart.core import grid_rows, load_image  # noqa: E402
from asciiart.viz import contact_sheet, ids_to_pil  # noqa: E402
from scripts.eval_methods import make_method  # noqa: E402


def pick(n_per_source, seed=1, sources=None):
    recs = [json.loads(l) for l in open("data/images/eval.jsonl")]
    rng = random.Random(seed)
    out = []
    for s in sources or sorted({r["source"] for r in recs}):
        pool = [r for r in recs if r["source"] == s]
        out += rng.sample(pool, n_per_source)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["aic", "match"])
    ap.add_argument("--cols", type=int, default=80)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--images", nargs="*", default=None, help="explicit image paths instead of eval picks")
    ap.add_argument("--out", default="runs/viz/sheet.png")
    ap.add_argument("--chunk", type=int, default=4, help="images per sheet")
    args = ap.parse_args()
    recs = [{"path": p} for p in args.images] if args.images else pick(args.n, args.seed, args.sources)
    fns = [make_method(m, "cuda") for m in args.methods]
    rows = []
    for r in recs:
        img = load_image(r["path"])
        nr = grid_rows(img.width, img.height, args.cols)
        rows.append([img] + [ids_to_pil(f(r, img, args.cols, nr)) for f in fns])
    for k in range(0, len(rows), args.chunk):
        out = args.out.replace(".png", f"_{k // args.chunk}.png")
        contact_sheet(rows[k:k + args.chunk], ["original"] + args.methods).save(out)
        print(out)


if __name__ == "__main__":
    main()
