"""Image preprocessing, differentiable ASCII rendering and text <-> id helpers."""
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .glyphs import CELL_H, CELL_W
from .imageio import load_image  # noqa: F401  (re-exported)

POLARITIES = ("dark", "light")  # terminal background: dark => bright pixels get ink


def grid_rows(w: int, h: int, cols: int, max_rows: int = 400) -> int:
    return min(max_rows, max(1, int(round(cols * (h / w) * (CELL_W / CELL_H)))))


def image_to_tensor(img: Image.Image, cols: int, rows: int | None = None) -> torch.Tensor:
    """Resize to exactly (rows*CELL_H, cols*CELL_W) pixels -> (3, H, W) float in [0,1]."""
    if rows is None:
        rows = grid_rows(img.width, img.height, cols)
    img = img.resize((cols * CELL_W, rows * CELL_H), Image.LANCZOS)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of (B,C,H,W) with reflect padding."""
    if sigma <= 0:
        return x
    r = max(1, int(math.ceil(3 * sigma)))
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = k / k.sum()
    C = x.shape[1]
    H, W = x.shape[-2:]
    mode = "reflect" if (r < H and r < W) else "replicate"
    x = F.pad(x, (r, r, 0, 0), mode=mode)
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    x = F.pad(x, (0, 0, r, r), mode=mode)
    x = F.conv2d(x, k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
    return x


def luminance(rgb: torch.Tensor) -> torch.Tensor:
    w = torch.tensor([0.299, 0.587, 0.114], device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    return (rgb * w).sum(1, keepdim=True)


def tone_map(rgb: torch.Tensor, polarity="dark", local_contrast: float = 0.6,
             mask: torch.Tensor | None = None) -> torch.Tensor:
    """(B,3,H,W) rgb -> (B,1,H,W) 'ink' in [0,1]: robust contrast stretch + local contrast boost.

    polarity: "dark" (dark terminal, bright pixels => ink) or "light" (dark pixels => ink),
    or a (B,) bool tensor that is True where the light-background mapping should be used.
    """
    y = luminance(rgb)
    B = y.shape[0]
    flat = y.flatten(1)
    if mask is not None:  # stretch contrast for the subject region only; mask (H,W) or (B,1,H,W)
        m = mask.to(y).view(-1, 1, *mask.shape[-2:])
        if m.shape[-2:] != y.shape[-2:]:
            m = F.interpolate(m, size=y.shape[-2:], mode="bilinear")
        sel = m.flatten(1).expand_as(flat) > 0.5
        flat = torch.where(sel, flat, torch.nan)
        lo = torch.nanquantile(flat, 0.01, dim=1).view(B, 1, 1, 1)
        hi = torch.nanquantile(flat, 0.99, dim=1).view(B, 1, 1, 1)
    else:
        lo = torch.quantile(flat, 0.01, dim=1).view(B, 1, 1, 1)
        hi = torch.quantile(flat, 0.99, dim=1).view(B, 1, 1, 1)
    # Don't blow up nearly-flat images into noise.
    span = (hi - lo).clamp_min(0.15)
    mid = (hi + lo) / 2
    lo = torch.minimum(lo, mid - span / 2)
    t = ((y - lo) / span).clamp(0, 1)
    if local_contrast > 0:
        sigma = max(2.0, min(t.shape[-2:]) / 10)
        t = (t + local_contrast * (t - gaussian_blur(t, sigma))).clamp(0, 1)
    if isinstance(polarity, str):
        return 1 - t if polarity == "light" else t
    inv = polarity.view(B, 1, 1, 1).to(t.dtype)
    return t * (1 - inv) + (1 - t) * inv


def auto_polarity(rgb: torch.Tensor, background: str = "dark") -> torch.Tensor:
    """Decide per image whether to invert so that a flat, uniform border (the likely background)
    maps to empty space rather than solid ink. Returns (B,) bool: True => use 'light' mapping."""
    y = luminance(rgb)[:, 0]
    H, W = y.shape[-2:]
    bh, bw = max(1, H // 12), max(1, W // 12)
    border = torch.cat([y[:, :bh].flatten(1), y[:, -bh:].flatten(1),
                        y[:, :, :bw].flatten(1), y[:, :, -bw:].flatten(1)], 1)
    bright_bg = (border.mean(1) > 0.7) & (border.std(1) < 0.12)
    dark_bg = (border.mean(1) < 0.3) & (border.std(1) < 0.12)
    if background == "dark":
        return bright_bg  # white-background art on a dark terminal: invert
    return ~dark_bg  # light terminal: invert unless the image already has a dark background


def cells_to_pixels(x: torch.Tensor) -> torch.Tensor:
    """(B, C, rows*CELL_H, cols*CELL_W) -> (B, C*CELL_H*CELL_W, rows, cols)."""
    B, C, H, W = x.shape
    x = x.view(B, C, H // CELL_H, CELL_H, W // CELL_W, CELL_W)
    return x.permute(0, 1, 3, 5, 2, 4).reshape(B, C * CELL_H * CELL_W, H // CELL_H, W // CELL_W)


def render(weights: torch.Tensor, atlas: torch.Tensor) -> torch.Tensor:
    """weights (B, V, rows, cols) (one-hot or soft) x atlas (V, ch, cw) -> (B, 1, rows*ch, cols*cw)."""
    B, V, R, C = weights.shape
    _, ch, cw = atlas.shape
    out = torch.einsum("bvrc,vhw->brhcw", weights, atlas.to(weights.dtype))
    return out.reshape(B, 1, R * ch, C * cw)


def render_ids(ids: torch.Tensor, atlas: torch.Tensor) -> torch.Tensor:
    """ids (B, rows, cols) long -> (B,1,H,W)."""
    B, R, C = ids.shape
    _, ch, cw = atlas.shape
    g = atlas.to(ids.device)[ids]  # B,R,C,ch,cw
    return g.permute(0, 1, 3, 2, 4).reshape(B, 1, R * ch, C * cw)


def ids_to_text(ids: torch.Tensor, chars: str) -> str:
    return "\n".join("".join(chars[i] for i in row).rstrip() for row in ids.tolist())


def text_to_ids(text: str, chars: str, cols: int | None = None) -> torch.Tensor:
    lines = text.split("\n")
    cols = cols or max(len(l) for l in lines)
    lut = {c: i for i, c in enumerate(chars)}
    return torch.tensor([[lut.get(ch, 0) for ch in l.ljust(cols)[:cols]] for l in lines], dtype=torch.long)
