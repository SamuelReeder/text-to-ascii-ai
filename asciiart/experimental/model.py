"""AsciiNet: fully-convolutional image -> per-cell character logits.

The pixel branch sees the whole image at render resolution (CELL_H x CELL_W pixels per
character), a cell-level U-Net adds context (object outlines, continuity of strokes),
and the exact pixels of each cell are injected before the head so glyph shape choice
can follow fine structure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..core import CELL_H, CELL_W, cells_to_pixels


class Block(nn.Module):
    """ConvNeXt-style block."""

    def __init__(self, dim, k=7):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pw1 = nn.Conv2d(dim, 4 * dim, 1)
        self.pw2 = nn.Conv2d(4 * dim, dim, 1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), 1e-2))

    def forward(self, x):
        return x + self.gamma * self.pw2(F.gelu(self.pw1(self.norm(self.dw(x)))))


def stage(dim, n, k=7):
    return nn.Sequential(*[Block(dim, k) for _ in range(n)])


def run(stage_, x, ckpt):
    """Gradient-checkpoint the high-resolution stages; they dominate activation memory."""
    for blk in stage_:
        x = checkpoint(blk, x, use_reentrant=False) if ckpt else blk(x)
    return x


class AsciiNet(nn.Module):
    def __init__(self, vocab: int, in_ch: int = 4, dims=(48, 96, 192, 256, 384), depths=(2, 2, 3, 3, 4)):
        super().__init__()
        d0, d1, d2, d3, d4 = dims
        # 16x8 pixel cell -> 4x4 sub-cells
        self.stem = nn.Conv2d(in_ch, d0, kernel_size=(8, 4), stride=(4, 2), padding=(2, 1))
        self.s0 = stage(d0, depths[0])
        self.down0 = nn.Conv2d(d0, d1, 2, 2)  # -> 2x2 sub-cells
        self.s1 = stage(d1, depths[1])
        self.down1 = nn.Conv2d(d1, d2, 2, 2)  # -> cell grid
        self.cell_pix = nn.Conv2d(CELL_H * CELL_W, d2, 1)
        self.s2 = stage(d2, depths[2])
        self.down2 = nn.Conv2d(d2, d3, 2, 2)
        self.s3 = stage(d3, depths[3])
        self.down3 = nn.Conv2d(d3, d4, 2, 2)
        self.s4 = stage(d4, depths[4])
        self.up3 = nn.Conv2d(d4, d3, 1)
        self.u3 = stage(d3, 1)
        self.up2 = nn.Conv2d(d3, d2, 1)
        self.u2 = stage(d2, 2)
        self.head_pix = nn.Conv2d(CELL_H * CELL_W, d2, 1)
        self.head = nn.Sequential(nn.GroupNorm(1, 2 * d2), nn.Conv2d(2 * d2, d2, 1), nn.GELU(),
                                  nn.Conv2d(d2, vocab, 1))

    def forward(self, ink: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """ink (B,1,H,W) in [0,1], rgb (B,3,H,W) in [0,1]; H=rows*CELL_H, W=cols*CELL_W.
        Returns logits (B, V, rows, cols)."""
        B, _, H, W = ink.shape
        R, C = H // CELL_H, W // CELL_W
        # pad the cell grid to a multiple of 4 for the two cell-level downsamplings
        pr, pc = (-R) % 4, (-C) % 4
        x = torch.cat([ink, rgb], 1) * 2 - 1
        if pr or pc:
            x = F.pad(x, (0, pc * CELL_W, 0, pr * CELL_H), mode="replicate")
            ink = F.pad(ink, (0, pc * CELL_W, 0, pr * CELL_H), mode="replicate")
        pix = cells_to_pixels(ink) * 2 - 1
        ck = self.training and torch.is_grad_enabled()
        h = run(self.s0, self.stem(x), ck)
        h = run(self.s1, self.down0(h), ck)
        h = self.down1(h) + self.cell_pix(pix)
        h2 = self.s2(h)
        h3 = self.s3(self.down2(h2))
        h4 = self.s4(self.down3(h3))
        u = self.u3(h3 + F.interpolate(self.up3(h4), size=h3.shape[-2:], mode="nearest"))
        u = self.u2(h2 + F.interpolate(self.up2(u), size=h2.shape[-2:], mode="nearest"))
        out = self.head(torch.cat([u, self.head_pix(pix)], 1))
        return out[:, :, :R, :C]
