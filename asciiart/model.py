"""AsciiNet: our image -> ASCII network, trained from scratch (random init) in PyTorch.

Two branches meet on the character grid:

    context image (~384 px) ─► conv stages ─► transformer (global) ─► FPN ─► subject map (aux)
                                                                        └─► context features ─┐
    grid image (8x16 px per cell) ─► conv stages at 1/4 and 1/8 ─► per-cell features ─────────┤
    exact cell pixels ──────────────────────────────────────────────► per-cell features ──────┤
    options (dark/light background, invert, focus) ──────────────────► embedding ─────────────┤
                                                                                              ▼
                                         cell-level ConvNeXt blocks (stroke continuity) ─► glyph logits

The context branch sees the whole picture at a fixed resolution, so it can find the subject
(what to keep and what to fade) at any output size, including very small grids. The grid branch
sees every pixel of every cell, so glyph shapes follow fine structure. It is fully convolutional
over the grid: one model serves every width.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .core import CELL_H, CELL_W, cells_to_pixels

CTX_AREA = 384 * 384  # context image pixels (aspect ratio follows the grid)
# conditioning: background dark/light x invert auto/yes/no x focus auto/off
BACKGROUNDS, INVERTS, FOCUS = ("dark", "light"), ("auto", "yes", "no"), ("auto", "off")


def ctx_size(rows: int, cols: int) -> tuple[int, int]:
    """Context image (H, W): ~CTX_AREA pixels, same aspect as the grid, multiples of 32."""
    aspect = (rows * CELL_H) / (cols * CELL_W)
    h = math.sqrt(CTX_AREA * aspect)
    w = CTX_AREA / h
    return max(64, int(round(h / 32)) * 32), max(64, int(round(w / 32)) * 32)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Block(nn.Module):
    """ConvNeXt block."""

    def __init__(self, dim, k=7):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, 4 * dim, 1)
        self.pw2 = nn.Conv2d(4 * dim, dim, 1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), 0.1))

    def forward(self, x):
        return x + self.gamma * self.pw2(F.gelu(self.pw1(self.norm(self.dw(x)))))


class Attention(nn.Module):
    """Pre-norm transformer block over a feature map; a depthwise conv supplies position."""

    def __init__(self, dim, heads=8):
        super().__init__()
        self.pos = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.heads = heads

    def forward(self, x):
        B, C, H, W = x.shape
        x = x + self.pos(x)
        t = x.flatten(2).transpose(1, 2)
        q, k, v = self.qkv(self.n1(t)).view(B, -1, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, -1, C)
        t = t + self.proj(a)
        t = t + self.mlp(self.n2(t))
        return t.transpose(1, 2).reshape(B, C, H, W)


def stage(dim, n, k=7):
    return nn.Sequential(*[Block(dim, k) for _ in range(n)])


def down(cin, cout, stride=2):
    return nn.Sequential(LayerNorm2d(cin), nn.Conv2d(cin, cout, stride, stride))


class AsciiNet(nn.Module):
    def __init__(self, vocab: int, ctx_dims=(64, 128, 256, 512), ctx_depths=(2, 2, 6), attn_blocks=6,
                 grid_dims=(48, 96), cell_dim=256, cell_blocks=6):
        super().__init__()
        c0, c1, c2, c3 = ctx_dims
        # --- context branch (whole picture, fixed resolution) ---
        self.c_stem = nn.Sequential(nn.Conv2d(3, c0, 4, 4), LayerNorm2d(c0))
        self.c_s0 = stage(c0, ctx_depths[0])
        self.c_d1, self.c_s1 = down(c0, c1), stage(c1, ctx_depths[1])
        self.c_d2, self.c_s2 = down(c1, c2), stage(c2, ctx_depths[2])
        self.c_d3 = down(c2, c3)
        self.c_attn = nn.Sequential(*[Attention(c3) for _ in range(attn_blocks)])
        self.cond = nn.Embedding(len(BACKGROUNDS) * len(INVERTS) * len(FOCUS), c3)
        self.lat2, self.lat1 = nn.Conv2d(c2, c2, 1), nn.Conv2d(c1, c1, 1)
        self.top2, self.top1 = nn.Conv2d(c3, c2, 1), nn.Conv2d(c2, c1, 1)
        self.f2, self.f1 = stage(c2, 1), stage(c1, 1)
        self.matte_head = nn.Conv2d(c1, 1, 1)
        # --- grid branch (every pixel of every cell) ---
        g0, g1 = grid_dims
        self.g_stem = nn.Sequential(nn.Conv2d(4, g0, 4, 4), LayerNorm2d(g0))  # cell 16x8 -> 4x2
        self.g_s0 = stage(g0, 2)
        self.g_d1, self.g_s1 = down(g0, g1), stage(g1, 2)  # -> 2x1
        self.g_d2 = nn.Sequential(LayerNorm2d(g1), nn.Conv2d(g1, cell_dim, (2, 1), (2, 1)))  # -> cell
        self.cell_pix = nn.Conv2d(CELL_H * CELL_W * 3, cell_dim, 1)
        # --- fusion on the character grid ---
        self.ctx_to_cell = nn.Sequential(LayerNorm2d(c1 + c2 + 1), nn.Conv2d(c1 + c2 + 1, cell_dim, 1))
        self.cond_cell = nn.Embedding(len(BACKGROUNDS) * len(INVERTS) * len(FOCUS), cell_dim)
        self.cell = stage(cell_dim, cell_blocks)
        self.head = nn.Sequential(LayerNorm2d(cell_dim), nn.Conv2d(cell_dim, vocab, 1))
        self.ink_head = nn.Conv2d(cell_dim, 1, 1)  # aux: the teacher's mean ink per cell
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def cond_index(background="dark", invert="auto", focus="auto") -> int:
        return (BACKGROUNDS.index(background) * len(INVERTS) + INVERTS.index(invert)) * len(FOCUS) + FOCUS.index(focus)

    def forward(self, x, ctx, cond, ckpt: bool = False):
        """x: (B,3,rows*16,cols*8) grid image in [0,1]; ctx: (B,3,Hc,Wc) context image in [0,1];
        cond: (B,) long option index. Returns glyph logits (B,V,rows,cols), subject-map logits
        (B,1,Hc/8,Wc/8) and cell ink (B,1,rows,cols)."""
        R, C = x.shape[-2] // CELL_H, x.shape[-1] // CELL_W
        run = (lambda f, *a: checkpoint(f, *a, use_reentrant=False)) if ckpt else (lambda f, *a: f(*a))
        # context
        h0 = run(self.c_s0, self.c_stem(ctx * 2 - 1))
        h1 = run(self.c_s1, self.c_d1(h0))
        h2 = run(self.c_s2, self.c_d2(h1))
        h3 = self.c_d3(h2) + self.cond(cond)[:, :, None, None]
        h3 = run(self.c_attn, h3)
        p2 = self.f2(self.lat2(h2) + F.interpolate(self.top2(h3), size=h2.shape[-2:], mode="bilinear"))
        p1 = self.f1(self.lat1(h1) + F.interpolate(self.top1(p2), size=h1.shape[-2:], mode="bilinear"))
        matte = self.matte_head(p1)
        # grid
        lum = (0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
        g = run(self.g_s0, self.g_stem(torch.cat([x, lum], 1) * 2 - 1))
        g = run(self.g_s1, self.g_d1(g))
        cell = self.g_d2(g) + self.cell_pix(cells_to_pixels(x) * 2 - 1)
        # fuse
        up = lambda t: F.interpolate(t, size=(R, C), mode="bilinear", align_corners=False)  # noqa: E731
        cell = cell + self.ctx_to_cell(torch.cat([up(p1), up(p2), up(matte)], 1))
        cell = cell + self.cond_cell(cond)[:, :, None, None]
        cell = run(self.cell, cell)
        return self.head(cell), matte, self.ink_head(cell)
