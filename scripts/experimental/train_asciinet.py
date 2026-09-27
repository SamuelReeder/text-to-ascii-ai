"""Train AsciiNet by rendering its characters and comparing the rendering to the image.

Loss = multi-scale render loss (tone + structure; what the ASCII looks like at several
viewing distances) + optional CLIP perceptual loss (does the rendering depict the same
thing?) + a short warm-up imitating per-cell glyph matching.
Characters are chosen with straight-through Gumbel-softmax so training renders are
exactly what inference produces.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from asciiart.baselines import match_ids  # noqa: E402
from asciiart.core import (CELL_H, CELL_W, auto_polarity, gaussian_blur, grid_rows,  # noqa: E402
                           image_to_tensor, load_image, render, render_ids, tone_map)
from asciiart.glyphs import PRINTABLE, glyph_atlas  # noqa: E402
from asciiart.experimental.model import AsciiNet  # noqa: E402
from asciiart.experimental.synthetic import make_synthetic  # noqa: E402

SOURCE_WEIGHTS = {"imagenet": 0.30, "coco": 0.12, "celeba": 0.07, "ffhq": 0.05, "wikiart": 0.08,
                  "pokemon": 0.03, "cartoon": 0.07, "quickdraw": 0.03, "synthetic": 0.25}


class Batches(IterableDataset):
    """Yields whole batches that share one random grid size (cols x rows)."""

    def __init__(self, manifest, batch, min_cols, max_cols, max_cells, seed):
        recs = [json.loads(l) for l in open(manifest)]
        self.by = defaultdict(list)
        for r in recs:
            self.by[r["source"]].append(r["path"])
        self.sources = [s for s in SOURCE_WEIGHTS if s == "synthetic" or self.by.get(s)]
        self.w = np.array([SOURCE_WEIGHTS[s] for s in self.sources])
        self.w /= self.w.sum()
        self.batch, self.min_cols, self.max_cols, self.max_cells, self.seed = batch, min_cols, max_cols, max_cells, seed

    def sample_image(self, rng):
        s = self.sources[np.searchsorted(np.cumsum(self.w), rng.random())]
        if s == "synthetic":
            photo = None
            if rng.random() < 0.3:
                photo = Image.open(rng.choice(self.by["imagenet"]))
            return make_synthetic(rng, photo)
        return load_image(rng.choice(self.by[s]))

    def crop(self, img, R, C, rng):
        aspect = (R * CELL_H) / (C * CELL_W)  # target h/w
        w, h = img.size
        area = rng.uniform(0.55, 1.0) * w * h
        cw = min(w, math.sqrt(area / aspect))
        ch = min(h, cw * aspect)
        cw = ch / aspect
        x0, y0 = rng.uniform(0, w - cw), rng.uniform(0, h - ch)
        img = img.crop((x0, y0, x0 + cw, y0 + ch))
        if rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        x = image_to_tensor(img, C, R)
        r = rng.random()  # photometric robustness: under/over-exposed and washed-out inputs
        if r < 0.15:
            x = x ** rng.uniform(1.5, 3.0)
        elif r < 0.25:
            x = x ** rng.uniform(0.35, 0.7)
        elif r < 0.35:
            m = x.mean()
            x = m + (x - m) * rng.uniform(0.25, 0.6)
        return x.clamp(0, 1)

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (wi.id if wi else 0) * 7919 + int(time.time()))
        while True:
            C = rng.randint(self.min_cols, self.max_cols)
            R = int(round(C * rng.uniform(0.5, 1.5) * CELL_W / CELL_H))
            R = max(6, min(R, self.max_cells // C))
            xs = []
            while len(xs) < self.batch:
                try:
                    xs.append(self.crop(self.sample_image(rng), R, C, rng))
                except Exception as e:  # noqa: BLE001
                    print("bad sample", e, file=sys.stderr)
            yield torch.stack(xs)


def st_gumbel(logits, tau, noise):
    """Straight-through Gumbel-softmax over dim 1. Forward: one-hot. Backward: softmax grads."""
    if noise > 0:
        g = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-9, 1 - 1e-9)))
        logits = logits + noise * g
    soft = F.softmax(logits / tau, dim=1)
    hard = F.one_hot(soft.argmax(1), soft.shape[1]).permute(0, 3, 1, 2).to(soft.dtype)
    return hard - soft.detach() + soft, soft


def render_loss(img, target, sigmas, weights, norm):
    loss = 0.0
    for s, w in zip(sigmas, weights):
        loss = loss + w * F.mse_loss(gaussian_blur(img, s), gaussian_blur(target, s))
    return loss / norm


class ClipLoss:
    def __init__(self, spec, device):
        import open_clip
        model, _, pre = open_clip.create_model_and_transforms(spec[0], pretrained=spec[1], device=device)
        self.model = model.eval().requires_grad_(False)
        cfg = model.visual.preprocess_cfg
        self.size = cfg["size"] if isinstance(cfg["size"], int) else cfg["size"][0]
        self.mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)

    def prep(self, x):
        x = x.expand(-1, 3, -1, -1)
        H, W = x.shape[-2:]
        S = max(H, W)
        x = F.pad(x, ((S - W) // 2, S - W - (S - W) // 2, (S - H) // 2, S - H - (S - H) // 2))
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", antialias=True, align_corners=False)
        return (x - self.mean) / self.std

    def embed(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return F.normalize(self.model.encode_image(self.prep(x)).float(), dim=-1)


def prepare_batch(rgb, gen=None, flip_p=0.2):
    """rgb (B,3,H,W) -> ink target, polarity-adjusted rgb. Mostly auto polarity, some random flips."""
    inv = auto_polarity(rgb, "dark")
    if flip_p > 0:
        flip = torch.rand(inv.shape, device=rgb.device, generator=gen) < flip_p
        inv = inv ^ flip
    ink = tone_map(rgb, inv)
    rgb_adj = torch.where(inv.view(-1, 1, 1, 1), 1 - rgb, rgb)
    return ink, rgb_adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-cols", type=int, default=32)
    ap.add_argument("--max-cols", type=int, default=110)
    ap.add_argument("--max-cells", type=int, default=4400)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--sigma-weights", type=float, nargs="+", default=[1, 1, 1, 1])
    ap.add_argument("--teacher-steps", type=int, default=2000)
    ap.add_argument("--clip-weight", type=float, default=0.0)
    ap.add_argument("--clip-model", nargs=2, default=["ViT-B-16-SigLIP", "webli"])
    ap.add_argument("--clip-every", type=int, default=1)
    ap.add_argument("--tau", type=float, nargs=2, default=[1.0, 0.3])
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--init", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--eval-per-source", type=int, default=100)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    device = "cuda"
    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    chars = PRINTABLE
    atlas = glyph_atlas(chars).to(device)
    cmax = atlas.mean((1, 2)).max().item()
    model = AsciiNet(len(chars)).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device)["model"])
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02, betas=(0.9, 0.99))
    warm = 500

    def lr_at(s):
        if s < warm:
            return args.lr * s / warm
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / (args.steps - warm)))) + 1e-6

    clip = ClipLoss(tuple(args.clip_model), device) if args.clip_weight > 0 else None
    ds = Batches("data/images/train.jsonl", args.batch, args.min_cols, args.max_cols, args.max_cells, seed=1234)
    dl = DataLoader(ds, batch_size=None, num_workers=args.workers, pin_memory=True, prefetch_factor=4,
                    persistent_workers=True)
    it = iter(dl)

    evaluator = None
    log = open(out / "log.jsonl", "a")
    t0 = time.time()
    agg = defaultdict(float)
    n_agg = 0
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        rgb = next(it).to(device, non_blocking=True)
        with torch.no_grad():
            ink, rgb_adj = prepare_batch(rgb)
            target = ink * cmax
        frac = step / args.steps
        tau = args.tau[0] * (args.tau[1] / args.tau[0]) ** frac
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ink, rgb_adj)
        logits = logits.float()
        y, soft = st_gumbel(logits, tau, args.noise * (1 - frac))
        img = render(y, atlas)
        l_render = render_loss(img, target, args.sigmas, args.sigma_weights, cmax ** 2)
        loss = l_render
        stats = {"render": l_render.item()}
        tw = max(0.0, 1 - step / args.teacher_steps) if args.teacher_steps else 0.0
        if tw > 0:
            with torch.no_grad():
                teach = match_ids(ink, atlas)
            l_t = F.cross_entropy(logits, teach)
            loss = loss + tw * l_t
            stats["teacher"] = l_t.item()
        if clip is not None and step % args.clip_every == 0:
            with torch.no_grad():
                e_t = clip.embed(ink)
            e_r = clip.embed((img / atlas.max()).clamp(0, 1))
            l_clip = (1 - (e_r * e_t).sum(-1)).mean()
            loss = loss + args.clip_weight * l_clip
            stats["clip"] = l_clip.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            stats["ent"] = -(F.softmax(logits, 1) * F.log_softmax(logits, 1)).sum(1).mean().item()
        for k, v in stats.items():
            agg[k] += v
        n_agg += 1
        if step % 100 == 0:
            row = {"step": step, "lr": lr_at(step), "tau": tau, "gn": gn.item(), "sec": time.time() - t0,
                   "grid": list(ink.shape[-2:]), **{k: v / n_agg for k, v in agg.items()}}
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            agg.clear()
            n_agg = 0
        if step % args.eval_every == 0 or step == args.steps:
            ck = {"model": model.state_dict(), "chars": chars, "arch": "AsciiNet", "step": step, "args": vars(args)}
            torch.save(ck, out / "ckpt.pt")
            if args.eval_per_source:
                evaluator = run_eval(model, atlas, chars, evaluator, args, log, step)
            model.train()


@torch.no_grad()
def run_eval(model, atlas, chars, evaluator, args, log, step):
    from asciiart.evaluate import LegibilityEval, ascii_render_tensor, load_eval
    model.eval()
    if evaluator is None:
        evaluator = LegibilityEval(load_eval(per_source=args.eval_per_source))
    renders = []
    for r in evaluator.recs:
        img = load_image(r["path"])
        x = image_to_tensor(img, 80)[None].cuda()
        ink, rgb_adj = prepare_batch(x, flip_p=0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ids = model(ink, rgb_adj).argmax(1)[0].cpu()
        renders.append(ascii_render_tensor(ids, atlas.cpu()))
    s = evaluator.score_renders(renders)
    row = {"step": step, "eval": s}
    print("EVAL", json.dumps({k: round(v, 3) for k, v in s.items()}), flush=True)
    log.write(json.dumps(row) + "\n")
    log.flush()
    return evaluator


if __name__ == "__main__":
    main()
