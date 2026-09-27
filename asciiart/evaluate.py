"""Legibility metrics: can a strong CLIP model recognize what an ASCII rendering depicts?

* zero-shot top-1 on the rendered ASCII (ImageNet 1000-way, Caltech-101 102-way)
* retrieval: find the source image of an ASCII rendering among all eval images of that domain
* caption retrieval (Flickr8k): find the right caption for an ASCII rendering among 1000
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .core import load_image, render_ids

EVAL_MODEL = ("ViT-L-14", "datacomp_xl_s13b_b90k")
ASCII_TEMPLATES = ["ascii art of a {c}.", "an ascii art drawing of a {c}.", "a {c} drawn with text characters.",
                   "text art of a {c}.", "a monochrome ascii picture of a {c}."]
_clip_cache = {}


def get_clip(device="cuda", spec=EVAL_MODEL):
    if spec not in _clip_cache:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(spec[0], pretrained=spec[1], device=device)
        model.eval().half()
        _clip_cache[spec] = (model, open_clip.get_tokenizer(spec[0]))
    return _clip_cache[spec]


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def to_clip_input(x: torch.Tensor, size=224, pad_value=0.0) -> torch.Tensor:
    """(B,C,H,W) in [0,1] (C=1 or 3) -> pad to square, resize, normalize."""
    if x.shape[1] == 1:
        x = x.expand(-1, 3, -1, -1)
    H, W = x.shape[-2:]
    S = max(H, W)
    x = F.pad(x, ((S - W) // 2, S - W - (S - W) // 2, (S - H) // 2, S - H - (S - H) // 2), value=pad_value)
    x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    return (x - CLIP_MEAN.to(x)) / CLIP_STD.to(x)


@torch.no_grad()
def embed_images(tensors, device="cuda", bs=64):
    model, _ = get_clip(device)
    out = []
    for i in range(0, len(tensors), bs):
        batch = torch.cat([to_clip_input(t[None].to(device).float()) for t in tensors[i:i + bs]])
        out.append(F.normalize(model.encode_image(batch.half()).float(), dim=-1))
    return torch.cat(out)


@torch.no_grad()
def embed_texts(texts, device="cuda", bs=256):
    model, tok = get_clip(device)
    out = []
    for i in range(0, len(texts), bs):
        out.append(F.normalize(model.encode_text(tok(texts[i:i + bs]).to(device)).float(), dim=-1))
    return torch.cat(out)


@torch.no_grad()
def classifier_weights(classnames, templates, device="cuda"):
    ws = []
    for c in classnames:
        e = embed_texts([t.format(c=c) for t in templates], device)
        ws.append(F.normalize(e.mean(0), dim=-1))
    return torch.stack(ws)


def load_eval(manifest="data/images/eval.jsonl", per_source=None):
    recs = [json.loads(l) for l in open(manifest)]
    if per_source:
        by = defaultdict(list)
        for r in recs:
            if len(by[r["source"]]) < per_source:
                by[r["source"]].append(r)
        recs = [r for v in by.values() for r in v]
    return recs


def ascii_render_tensor(ids: torch.Tensor, atlas: torch.Tensor, polarity="dark") -> torch.Tensor:
    """ids (rows, cols) -> (1, H, W) image as shown on a terminal of the given background."""
    img = render_ids(ids[None].cpu(), atlas.cpu())[0].clamp(0, 1)
    img = img / atlas.max()  # glyph strokes at full intensity, as a terminal draws them
    return img.clamp(0, 1) if polarity == "dark" else 1 - img.clamp(0, 1)


class LegibilityEval:
    """Precomputes CLIP embeddings of the originals + text classifiers once; scores any converter."""

    def __init__(self, recs, device="cuda"):
        self.recs = recs
        self.device = device
        imgs = [torch.from_numpy(np.asarray(load_image(r["path"]), dtype=np.float32) / 255).permute(2, 0, 1)
                for r in recs]
        self.orig_emb = embed_images(imgs, device)
        self.sources = sorted({r["source"] for r in recs})
        # Renders are classified with ASCII-specific prompts: with photo prompts, CLIP files most
        # white-on-black text under "monitor"/"digital clock" regardless of content.
        from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES
        cal = Path("data/raw/clip-benchmark__wds_vtab-caltech101")
        self.classnames = {"imagenet": list(IMAGENET_CLASSNAMES),
                           "caltech101": (cal / "classnames.txt").read_text().strip().split("\n")}
        self.cls = {k: classifier_weights(v, ASCII_TEMPLATES, device) for k, v in self.classnames.items()}
        fl = [r for r in recs if r["source"] == "flickr8k"]
        if fl:
            self.flickr_caps = embed_texts([c for r in fl for c in r["captions"]], device).view(len(fl), 5, -1)
        self.orig_scores = self._score(self.orig_emb)

    def _score(self, emb):
        res = {}
        idx = defaultdict(list)
        for i, r in enumerate(self.recs):
            idx[r["source"]].append(i)
        for s, ii in idx.items():
            ii_t = torch.tensor(ii, device=self.device)
            e = emb[ii_t]
            if s in self.cls:
                labels = torch.tensor([self.recs[i]["label"] for i in ii], device=self.device)
                top5 = (e @ self.cls[s].T).topk(5, 1).indices
                res[f"{s}/zs_top1"] = (top5[:, 0] == labels).float().mean().item()
                res[f"{s}/zs_top5"] = (top5 == labels[:, None]).any(1).float().mean().item()
            if s == "flickr8k":
                sims = torch.einsum("nd,mkd->nmk", e, self.flickr_caps).max(-1).values
                res[f"{s}/cap_r1"] = (sims.argmax(1) == torch.arange(len(ii), device=self.device)).float().mean().item()
            sims = e @ self.orig_emb[ii_t].T
            res[f"{s}/img_r1"] = (sims.argmax(1) == torch.arange(len(ii), device=self.device)).float().mean().item()
        img_r1 = [v for k, v in res.items() if k.endswith("img_r1")]
        res["mean_img_r1"] = float(np.mean(img_r1))
        zs = [v for k, v in res.items() if k.endswith("zs_top1") or k.endswith("cap_r1")]
        res["mean_semantic"] = float(np.mean(zs))
        return res

    def score_renders(self, renders):
        """renders: list of (1,H,W) tensors aligned with recs."""
        return self._score(embed_images(renders, self.device))
