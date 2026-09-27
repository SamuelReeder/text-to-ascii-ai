"""Perceptually guided glyph search: make the ASCII rendering *recognizable*, not just similar in tone.

Each iteration renders the current characters, backpropagates a CLIP-style similarity loss
(between the rendering and the source image) to the rendered pixels, and converts that
pixel gradient into a first-order score for swapping every cell to every glyph. Those
scores are combined with the exact render-loss deltas from LocalSearch, and the most
beneficial swaps are applied. This is used offline to create training targets for AsciiNet.
"""
import torch
import torch.nn.functional as F

from ..clip import ClipEmbedder  # noqa: F401
from ..core import cells_to_pixels, render_ids
from .refine import LocalSearch


def clip_loss(embedder, render01, targets):
    es = embedder.embed(render01)
    return sum((1 - (e * t).sum(-1)) for e, t in zip(es, targets)) / len(es)


class GuidedSearch:
    def __init__(self, atlas, embedder, sigmas=(1, 2, 4, 8), weights=(1, 1, 1, 1)):
        self.atlas = atlas
        self.A = atlas.flatten(1)
        self.ls = LocalSearch(atlas, sigmas, weights)
        self.emb = embedder
        self.gmax = atlas.max()

    def target_embeddings(self, rgb, ink, mix=0.5):
        """Blend of the original color image and the tone-mapped ink image (what ASCII can show)."""
        with torch.no_grad():
            a = self.emb.embed(rgb)
            b = self.emb.embed(ink)
        return [F.normalize(mix * x + (1 - mix) * y, dim=-1) for x, y in zip(a, b)]

    def search(self, ids, ink_target, emb_targets, iters=24, lam=1.0, frac=0.08, min_gap=1):
        """ids (B,r,c) init; ink_target (B,1,H,W) scaled to glyph range.
        lam weighs the CLIP term against the render loss (both per-image mean-scaled)."""
        ids = ids.clone()
        B, R, C = ids.shape
        P = ink_target[0].numel()
        best_ids, best_obj = ids.clone(), None
        for it in range(iters):
            img = render_ids(ids, self.atlas).requires_grad_(True)
            lc = clip_loss(self.emb, (img / self.gmax).clamp(0, 1), emb_targets)
            g, = torch.autograd.grad(lc.sum(), img)
            with torch.no_grad():
                lr = self.ls.loss(ids, ink_target)
                obj = lr + lam * lc
                if best_obj is None:
                    best_obj = obj.clone()
                better = obj < best_obj
                best_obj = torch.where(better, obj, best_obj)
                best_ids = torch.where(better.view(-1, 1, 1), ids, best_ids)
                gl = torch.einsum("bprc,vp->bvrc", cells_to_pixels(g), self.A)
                d_clip = gl - gl.gather(1, ids[:, None])
                d_render = self.ls.deltas(ids, ink_target) / (P * self.ls.norm)
                d = d_render + lam * d_clip
                best = d.argmin(1)
                gain = d.gather(1, best[:, None])[:, 0]  # B,r,c (negative = improvement)
                # apply only the top `frac` most beneficial swaps per image (first-order model is local)
                k = max(1, int(frac * R * C))
                thr = (-gain).flatten(1).topk(k, dim=1).values[:, -1].view(B, 1, 1)
                upd = (-gain >= thr) & (gain < 0)
                if min_gap:
                    # avoid updating horizontally adjacent cells together (they interact the most)
                    keep = torch.ones_like(upd)
                    keep[:, :, 1:] = ~(upd[:, :, 1:] & upd[:, :, :-1] & ((-gain[:, :, 1:]) < (-gain[:, :, :-1])))
                    upd = upd & keep
                ids = torch.where(upd, best, ids)
        with torch.no_grad():
            img = render_ids(ids, self.atlas)
            obj = self.ls.loss(ids, ink_target) + lam * clip_loss(self.emb, (img / self.gmax).clamp(0, 1), emb_targets)
            better = obj < best_obj
            best_ids = torch.where(better.view(-1, 1, 1), ids, best_ids)
        return best_ids
