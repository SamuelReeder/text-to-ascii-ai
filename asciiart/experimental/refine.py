"""Exact local search over glyph choices for the multi-scale render loss.

For L(R) = sum_s w_s ||G_s R - G_s T||^2 / norm (G_s = Gaussian blur), swapping the glyph of a
single cell from u to v changes the loss by exactly

    dL = <dL/dR, g_v - g_u>_cell + Q[v, u],   Q[v,u] = sum_s w_s ||G_s (g_v - g_u)||^2 / norm

(ignoring image-border effects). Q is a VxV table computed once, so evaluating every
alternative glyph for every cell costs one gradient image + one matmul.
"""
import torch

from ..core import CELL_H, CELL_W, cells_to_pixels, gaussian_blur, render_ids


class LocalSearch:
    def __init__(self, atlas: torch.Tensor, sigmas=(1, 2, 4, 8), weights=(1, 1, 1, 1)):
        self.atlas = atlas
        self.sigmas, self.weights = list(sigmas), list(weights)
        cmax = atlas.mean((1, 2)).max().item()
        self.norm = cmax ** 2
        V = atlas.shape[0]
        pad = int(3 * max(sigmas)) + 2
        canvas = torch.zeros(V, 1, CELL_H + 2 * pad, CELL_W + 2 * pad, device=atlas.device)
        canvas[:, 0, pad:pad + CELL_H, pad:pad + CELL_W] = atlas
        Q = torch.zeros(V, V, device=atlas.device)
        for s, w in zip(self.sigmas, self.weights):
            b = gaussian_blur(canvas, s).flatten(1)
            sq = (b * b).sum(1)
            Q += w * (sq[:, None] + sq[None, :] - 2 * b @ b.T)
        # MSE is a mean over pixels; express Q in the same units as the gradient term below.
        self.Q = Q.clamp_min(0)
        self.A = atlas.flatten(1)  # V, P

    def grad_image(self, ids, target):
        """dL/dR * (#pixels * norm) — the unnormalized gradient of sum of squared errors."""
        R = render_ids(ids, self.atlas)
        g = torch.zeros_like(R)
        for s, w in zip(self.sigmas, self.weights):
            g += 2 * w * gaussian_blur(gaussian_blur(R, s) - gaussian_blur(target, s), s)
        return g

    def deltas(self, ids, target):
        """(B, V, rows, cols) exact change of sum-of-squared-errors for swapping each cell to glyph v."""
        g = cells_to_pixels(self.grad_image(ids, target))  # B, P, r, c
        lin = torch.einsum("bprc,vp->bvrc", g, self.A)
        cur = lin.gather(1, ids[:, None])
        quad = self.Q[:, ids].permute(1, 0, 2, 3)  # B, V, r, c : Q[v, ids]
        return lin - cur + quad

    def loss(self, ids, target):
        R = render_ids(ids, self.atlas)
        tot = 0.0
        for s, w in zip(self.sigmas, self.weights):
            tot = tot + w * ((gaussian_blur(R, s) - gaussian_blur(target, s)) ** 2).mean((1, 2, 3))
        return tot / self.norm

    @torch.no_grad()
    def refine(self, ids, target, sweeps=4, stride=(2, 3), extra=None):
        """Coordinate descent: cells on a strided lattice are updated in parallel per phase.
        extra: optional (B,V,r,c) additive score (e.g. a prior from the network) in SSE units."""
        ids = ids.clone()
        B, R, C = ids.shape
        sr, sc = stride
        rr = torch.arange(R, device=ids.device)[:, None]
        cc = torch.arange(C, device=ids.device)[None, :]
        for _ in range(sweeps):
            changed = 0
            for pr in range(sr):
                for pc in range(sc):
                    mask = ((rr % sr) == pr) & ((cc % sc) == pc)
                    d = self.deltas(ids, target)
                    if extra is not None:
                        d = d + extra - extra.gather(1, ids[:, None])
                    best = d.argmin(1)
                    gain = d.gather(1, best[:, None])[:, 0]
                    upd = mask[None] & (gain < -1e-6)
                    ids = torch.where(upd, best, ids)
                    changed += int(upd.sum())
            if changed == 0:
                break
        return ids
