"""Score converters on the held-out legibility benchmark.

usage: python scripts/eval_methods.py --methods aic ramp match match_auto@clean pipe pipe:off --cols 80

methods: aic (ascii-image-converter CLI), ramp (density ramp), match (per-cell glyph matching),
         match_auto, pipe[:focus[:opt=val,...]] (the full pipeline; see asciiart.pipeline.Options),
         net[:checkpoint] (the trained AsciiNet);
         append @clean to restrict any glyph-matching method to letter-free characters.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asciiart.baselines import aic_ids, match_ids, ramp_ids  # noqa: E402
from asciiart.core import grid_rows, image_to_tensor, load_image, tone_map  # noqa: E402
from asciiart.evaluate import LegibilityEval, ascii_render_tensor, load_eval  # noqa: E402
from asciiart.glyphs import PRINTABLE, glyph_atlas  # noqa: E402


CHARSETS = {
    "clean": " .'`,:;-_~\"^!*/\\|()<>+=?[]{}#%@",
    "cleanx": " .'`,:;-_~\"^!*/\\|()<>+=?[]{}#%@ivxcoOVTYL",
}


def make_method(name, device):
    if "@" in name:  # method@charset: run with a restricted charset, map ids back to PRINTABLE
        base, cs = name.split("@")
        chars = CHARSETS[cs]
        lut = torch.tensor([PRINTABLE.index(c) for c in chars])
        inner = make_method_chars(base, device, chars)
        return lambda rec, img, cols, rows: lut[inner(rec, img, cols, rows)]
    return make_method_chars(name, device, PRINTABLE)


def make_method_chars(name, device, chars):
    atlas = glyph_atlas(chars).to(device)
    if name == "aic":
        def f(rec, img, cols, rows):
            return aic_ids(rec["path"], cols, rows, PRINTABLE)
    elif name == "ramp":
        def f(rec, img, cols, rows):
            x = image_to_tensor(img, cols, rows)[None].to(device)
            return ramp_ids(x, PRINTABLE, tone=tone_map(x))[0].cpu()
    elif name == "match":
        def f(rec, img, cols, rows):
            x = image_to_tensor(img, cols, rows)[None].to(device)
            return match_ids(tone_map(x), atlas)[0].cpu()
    elif name == "match_auto":  # per-cell glyph matching + the pipeline's background-aware polarity
        from asciiart.pipeline import bright_background
        def f(rec, img, cols, rows):
            x = image_to_tensor(img, cols, rows)[None].to(device)
            return match_ids(tone_map(x, bright_background(x)), atlas)[0].cpu()
    elif name.startswith("pipe"):
        # pipe[:focus[:key=val,...]]  -> the full pipeline (ids mapped back to PRINTABLE for rendering)
        from asciiart.pipeline import Converter, Options
        parts = name.split(":")
        focus = parts[1] if len(parts) > 1 else "auto"
        kw = {}
        if len(parts) > 2:
            for kv in parts[2].split(","):
                k, v = kv.split("=")
                kw[k] = type(getattr(Options(), k))(v) if not isinstance(getattr(Options(), k), bool) else v == "1"
        conv = Converter(device)
        lut = torch.tensor([PRINTABLE.index(c) for c in conv.chars])
        def f(rec, img, cols, rows):
            return lut[conv.ids(img, Options(cols=cols, focus=focus, **kw), rows=rows)]
    elif name.startswith("net"):
        # net[:checkpoint] -> the trained AsciiNet
        from asciiart.neural import DEFAULT_CKPT, NeuralConverter
        from asciiart.pipeline import Options
        parts = name.split(":", 1)
        conv = NeuralConverter(parts[1] if len(parts) > 1 else DEFAULT_CKPT, device)
        lut = torch.tensor([PRINTABLE.index(c) for c in conv.chars])
        def f(rec, img, cols, rows):
            return lut[conv.ids(img, Options(cols=cols), rows=rows)]
    else:
        raise ValueError(name)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["aic", "ramp", "match"])
    ap.add_argument("--cols", nargs="+", type=int, default=[80])
    ap.add_argument("--per-source", type=int, default=None)
    ap.add_argument("--out", default="runs/eval_results.jsonl")
    args = ap.parse_args()
    device = "cuda"
    recs = load_eval(per_source=args.per_source)
    ev = LegibilityEval(recs, device)
    atlas = glyph_atlas()
    print("original images:", json.dumps({k: round(v, 3) for k, v in ev.orig_scores.items()}))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imgs = [load_image(r["path"]) for r in recs]
    for m in args.methods:
        f = make_method(m, device)
        for cols in args.cols:
            t0 = time.time()
            renders = []
            for r, img in zip(recs, imgs):
                rows = grid_rows(img.width, img.height, cols)
                renders.append(ascii_render_tensor(f(r, img, cols, rows), atlas))
            dt = time.time() - t0
            s = ev.score_renders(renders)
            row = {"method": m, "cols": cols, "sec_per_img": dt / len(recs), **s}
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            with open(args.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
