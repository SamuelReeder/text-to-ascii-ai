"""Train AsciiNet (asciiart/model.py) from scratch to reproduce the pipeline's ASCII art.

Every step draws a batch at a random grid size (16-128 columns, weighted toward small grids),
crops and augments the images, labels them on the GPU with the batched pipeline (cached subject
mattes stand in for BiRefNet), and trains the network to predict each cell's glyph.

usage: python scripts/train.py --steps 30000   (-> checkpoints/asciinet.pt, loaded by asciiart/neural.py)
"""
import argparse
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

# grid sizes change every step: growable segments keep the CUDA cache from fragmenting
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asciiart.core import CELL_H, CELL_W, gaussian_blur  # noqa: E402
from asciiart.glyphs import glyph_atlas  # noqa: E402
from asciiart.model import BACKGROUNDS, FOCUS, INVERTS, AsciiNet, ctx_size  # noqa: E402
from asciiart.pipeline import CHARSET, Converter, Options  # noqa: E402
from scripts.cache_mattes import matte_path  # noqa: E402

MIN_COLS, MAX_COLS = 16, 128


def load_records(split):
    recs = [json.loads(l) for l in open(f"data/images/{split}.jsonl")]
    return [r for r in recs if matte_path(r["path"]).exists()]


def sample_options(rng):
    """Mostly the default (dark terminal, auto invert, auto focus), sometimes the others."""
    u = rng.random()
    bg = "light" if u < 0.15 else "dark"
    inv = rng.choices(INVERTS, weights=(0.9, 0.05, 0.05))[0]
    focus = "off" if rng.random() < 0.1 else "auto"
    return bg, inv, focus


class Batches(torch.utils.data.Sampler):
    """Yields lists of (record index, rows, cols, seed, options); all items share a grid size."""

    def __init__(self, recs, cells_per_batch, seed=0, upsample_limit=2.0):
        self.recs, self.cells, self.rng = recs, cells_per_batch, random.Random(seed)
        src = [r["source"] for r in recs]
        names = sorted(set(src))
        counts = {s: src.count(s) for s in names}
        # sources are sampled ~ sqrt(size): small domains (logos, sketches, ...) are seen often
        w = np.array([counts[s] ** 0.5 / counts[s] for s in src])
        self.p = w / w.sum()
        self.side = np.array([r["w"] for r in recs])  # image width in pixels
        self.upsample_limit = upsample_limit
        self.np_rng = np.random.default_rng(seed)

    def __iter__(self):
        while True:
            u = self.rng.random()
            cols = int(round(math.exp(self.rng.uniform(math.log(MIN_COLS), math.log(MAX_COLS)))))
            aspect = math.exp(self.rng.uniform(math.log(0.5), math.log(1.8)))  # image h / w
            rows = max(4, min(96, int(round(cols * aspect * CELL_W / CELL_H))))
            B = max(4, min(48, self.cells // (rows * cols)))
            p = self.p
            if u > 0.1:  # mostly avoid heavily upsampled images (10% of batches keep them)
                ok = self.side * self.upsample_limit >= cols * CELL_W
                if ok.any():
                    p = np.where(ok, p, 0)
                    p = p / p.sum()
            idx = self.np_rng.choice(len(self.recs), size=B, p=p)
            opts = sample_options(self.rng)
            yield [(int(i), rows, cols, self.rng.getrandbits(31), opts) for i in idx]


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        from PIL import ImageEnhance
        for E in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
            img = E(img).enhance(math.exp(rng.uniform(-0.4, 0.4)))
    if rng.random() < 0.2:
        g = math.exp(rng.uniform(-0.5, 0.5))
        img = img.point(lambda v: int(255 * (v / 255) ** g))
    if rng.random() < 0.08:
        img = ImageOps.grayscale(img).convert("RGB")
    if rng.random() < 0.03:
        img = ImageOps.invert(img)
    if rng.random() < 0.1:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.0)))
    if rng.random() < 0.1:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=rng.randint(15, 60))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img


class Data(torch.utils.data.Dataset):
    def __init__(self, recs, train=True):
        self.recs, self.train = recs, train

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, item):
        i, rows, cols, seed, opts = item
        rng = random.Random(seed)
        r = self.recs[i]
        img = Image.open(r["path"]).convert("RGB")
        m = Image.open(matte_path(r["path"])).convert("L")
        if m.size != img.size:
            m = m.resize(img.size, Image.BILINEAR)
        W, H = img.size
        target = (rows * CELL_H) / (cols * CELL_W)  # h / w of the crop
        if self.train:
            # largest crop with the grid's aspect ratio, then zoom in a little at random
            cw, ch = (W, W * target) if W * target <= H else (H / target, H)
            s = rng.uniform(0.75, 1.0) if rng.random() < 0.6 else 1.0
            cw, ch = cw * s, ch * s
            x0, y0 = rng.uniform(0, W - cw), rng.uniform(0, H - ch)
        else:
            cw, ch = (W, W * target) if W * target <= H else (H / target, H)
            x0, y0 = (W - cw) / 2, (H - ch) / 2
        box = (int(x0), int(y0), int(round(x0 + cw)), int(round(y0 + ch)))
        img, m = img.crop(box), m.crop(box)
        if self.train:
            if rng.random() < 0.5:
                img, m = ImageOps.mirror(img), ImageOps.mirror(m)
            img = augment(img, rng)
        hc, wc = ctx_size(rows, cols)
        x = torch.from_numpy(np.array(img.resize((cols * CELL_W, rows * CELL_H), Image.LANCZOS))).permute(2, 0, 1)
        ctx = torch.from_numpy(np.array(img.resize((wc, hc), Image.LANCZOS))).permute(2, 0, 1)
        mt = torch.from_numpy(np.asarray(m, dtype=np.float32) / 255)[None, None]
        mt = F.interpolate(mt, size=(rows * CELL_H, cols * CELL_W), mode="bilinear", antialias=True)[0]
        return x, ctx, (mt * 255).round().byte(), torch.tensor(AsciiNet.cond_index(*opts))


def collate(items):
    return [torch.stack(t) for t in zip(*items)]


class Teacher:
    """The pipeline, batched on the GPU, with cached mattes."""

    def __init__(self, device):
        self.conv = Converter(device, matte=False, selector=False)
        assert self.conv.chars == CHARSET

    @torch.no_grad()
    def __call__(self, x, m, cond):
        """x (B,3,H,W) float, m (B,1,H,W) float; cond: every item shares the same options."""
        c = int(cond[0])
        focus = FOCUS[c % len(FOCUS)]
        inv = INVERTS[(c // len(FOCUS)) % len(INVERTS)]
        bg = BACKGROUNDS[c // (len(FOCUS) * len(INVERTS))]
        opt = Options(cols=x.shape[-1] // CELL_W, background=bg, invert=inv, focus=focus)
        p = self.conv.prepare_tensor(x, m if focus != "off" else None, opt)
        ids = self.conv.convert_batch(p, opt)
        ink = F.avg_pool2d(p["ink"], (CELL_H, CELL_W))
        return ids, ink


def soft_targets(eps=0.1):
    """Label smoothing toward look-alike glyphs: confusing '/' with '(' costs less than with '@'."""
    a = glyph_atlas(CHARSET)
    g = gaussian_blur(a[:, None], 1.0)[:, 0].flatten(1)
    d = torch.cdist(g, g) ** 2 / g.shape[1]
    d.fill_diagonal_(float("inf"))
    tau = d[d.isfinite()].median() / 4
    q = torch.softmax(-d / tau, 1)
    return (1 - eps) * torch.eye(len(CHARSET)) + eps * q


def evaluate(model, teacher, batches, device):
    model.eval()
    stats = {}
    with torch.no_grad():
        for key, (x, ctx, m, cond) in batches:
            x, ctx, m, cond = x.to(device), ctx.to(device), m.to(device), cond.to(device)
            ids, _ = teacher(x.float() / 255, m.float() / 255, cond)  # full precision, as in training
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _, _ = model(x.float() / 255, ctx.float() / 255, cond)
            pred = logits.argmax(1)
            ink = ids != 0
            s = stats.setdefault(key, [0, 0, 0, 0])
            s[0] += (pred == ids).sum().item()
            s[1] += ids.numel()
            s[2] += ((pred == ids) & ink).sum().item()
            s[3] += ink.sum().item()
    model.train()
    return {k: (round(a / n, 4), round(b / max(1, c), 4)) for k, (a, n, b, c) in stats.items()}


def eval_batches(recs, n_per=48):
    """Fixed held-out batches at small, medium and large widths (dark terminal, default options)."""
    rng = random.Random(0)
    data = Data(recs, train=False)
    out = []
    pool = rng.sample(range(len(recs)), min(len(recs), 4 * n_per))
    for cols in (24, 40, 80):
        rows = max(4, round(cols * CELL_W / CELL_H))
        for k in range(0, len(pool), 16):
            items = [(i, rows, cols, 0, ("dark", "auto", "auto")) for i in pool[k:k + 16]]
            out.append((cols, collate([data[it] for it in items])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--cells", type=int, default=40000, help="character cells per batch")
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="checkpoints/asciinet_train.pt", help="full training state (for --resume)")
    ap.add_argument("--export", default="checkpoints/asciinet.pt", help="inference weights (EMA only)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", help="start from this training state's weights, with a fresh optimizer and schedule")
    ap.add_argument("--matte-w", type=float, default=0.3, help="weight of the subject-map BCE")
    ap.add_argument("--dice-w", type=float, default=0.0, help="weight of a soft-Dice subject-map loss (whole-subject coverage)")
    ap.add_argument("--ckpt", action="store_true", help="gradient checkpointing (less memory)")
    ap.add_argument("--vram-gb", type=float, default=10.5, help="hard cap on this process's CUDA memory")
    args = ap.parse_args()
    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(args.vram_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # grid sizes change every step, and cuDNN compiles new bf16 kernels for many new shapes (seconds each,
    # ~100 MB of host RAM each, never freed); PyTorch's native kernels are as fast here without that cost
    torch.backends.cudnn.enabled = False

    train_recs, eval_recs = load_records("train"), load_records("eval")
    print(f"train {len(train_recs)} images, eval {len(eval_recs)}", flush=True)
    dl = torch.utils.data.DataLoader(Data(train_recs), batch_sampler=Batches(train_recs, args.cells),
                                     num_workers=args.workers, collate_fn=collate,
                                     prefetch_factor=2, persistent_workers=True)
    evb = eval_batches(eval_recs)
    teacher = Teacher(device)
    model = AsciiNet(len(CHARSET)).to(device)
    ema = AsciiNet(len(CHARSET)).to(device)
    ema.load_state_dict(model.state_dict())
    ema.requires_grad_(False)
    decay = [p for n, p in model.named_parameters() if p.ndim > 1 and "cond" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.ndim > 1 and "cond" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0}],
                            lr=args.lr, betas=(0.9, 0.98), fused=True)
    Q = soft_targets().to(device)
    step, t0 = 0, time.time()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = open(out.with_suffix(".log"), "a")
    if args.resume and out.exists():
        ck = torch.load(out, map_location=device)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print("resumed at", step, flush=True)
    elif args.init:
        ck = torch.load(args.init, map_location=device)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        print("initialized from", args.init, "step", ck["step"], flush=True)
    agg = {}
    for x, ctx, m, cond in dl:
        if step >= args.steps:
            break
        lr = args.lr * min(1, (step + 1) / args.warmup) * (0.5 * (1 + math.cos(math.pi * min(1, step / args.steps))))
        for g in opt.param_groups:
            g["lr"] = lr
        x = x.to(device, non_blocking=True).float() / 255
        ctx = ctx.to(device, non_blocking=True).float() / 255
        m = m.to(device, non_blocking=True).float() / 255
        cond = cond.to(device, non_blocking=True)
        ids, ink = teacher(x, m, cond)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, matte, ink_pred = model(x, ctx, cond, ckpt=args.ckpt)
        logits = logits.float()
        target = Q[ids].permute(0, 3, 1, 2)  # B,V,R,C
        ce = -(target * F.log_softmax(logits, 1)).sum(1).mean()
        mt = F.adaptive_avg_pool2d(m, matte.shape[-2:])
        l_matte = F.binary_cross_entropy_with_logits(matte.float(), mt)
        if args.dice_w:
            ps = matte.float().sigmoid()
            inter, tot = (ps * mt).sum((1, 2, 3)), (ps + mt).sum((1, 2, 3))
            has = mt.mean((1, 2, 3)) > 0.015  # images where the teacher found a subject
            dice = (1 - (2 * inter + 1) / (tot + 1))[has].mean() if has.any() else matte.sum() * 0
            l_matte = l_matte + args.dice_w / args.matte_w * dice
        l_ink = F.l1_loss(ink_pred.float(), ink)
        loss = ce + args.matte_w * l_matte + 0.3 * l_ink
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            d = min(0.999, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        step += 1
        with torch.no_grad():
            pred = logits.argmax(1)
            a = agg.setdefault("n", [0.0] * 6)
            a[0] += ce.item()
            a[1] += l_matte.item()
            a[2] += l_ink.item()
            a[3] += (pred == ids).float().mean().item()
            nz = ids != 0
            a[4] += ((pred == ids) & nz).sum().item() / max(1, nz.sum().item())
            a[5] += 1
        if step % 100 == 0:
            a = agg.pop("n")
            n = a[5]
            msg = (f"step {step} lr {lr:.2e} ce {a[0]/n:.4f} matte {a[1]/n:.4f} ink {a[2]/n:.4f} "
                   f"acc {a[3]/n:.4f} ink_acc {a[4]/n:.4f} gn {gn:.2f} {(time.time()-t0)/100:.3f}s/step "
                   f"mem {torch.cuda.max_memory_allocated()/1e9:.1f}/{torch.cuda.max_memory_reserved()/1e9:.1f}GB")
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
            t0 = time.time()
        if step % 2000 == 0 or step == args.steps:
            ev = evaluate(ema, teacher, evb, device)
            msg = f"eval step {step} (acc, ink-cell acc) by cols: {ev}"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "step": step, "chars": CHARSET, "args": vars(args)}, out)
            # the EMA weights alone are what inference loads (asciiart/neural.py)
            torch.save({"ema": ema.state_dict(), "step": step, "chars": CHARSET, "args": vars(args)}, args.export)
            t0 = time.time()


if __name__ == "__main__":
    main()
