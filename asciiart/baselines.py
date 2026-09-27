"""Non-learned converters used as baselines."""
import math
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from .core import CELL_H, CELL_W, cells_to_pixels, gaussian_blur, luminance, text_to_ids

AIC_RAMP = " .:-=+*#%@"


def ramp_ids(rgb: torch.Tensor, chars: str, ramp: str = AIC_RAMP, tone=None) -> torch.Tensor:
    """Classic brightness -> density ramp on per-cell mean. rgb (B,3,H,W) -> ids (B,rows,cols)."""
    y = luminance(rgb) if tone is None else tone
    m = F.avg_pool2d(y, (CELL_H, CELL_W))[:, 0]
    idx = (m * len(ramp)).long().clamp(0, len(ramp) - 1)
    lut = torch.tensor([chars.index(c) for c in ramp], device=rgb.device)
    return lut[idx]


def match_ids(tone: torch.Tensor, atlas: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Per-cell nearest glyph (L2 after light blur); tone (B,1,H,W) in [0,1] scaled to glyph range."""
    atlas = atlas.to(tone.device)
    t = gaussian_blur(tone, sigma) * _scale(atlas) if sigma else tone * _scale(atlas)
    g = gaussian_blur(atlas[:, None], sigma)[:, 0] if sigma else atlas
    tc = cells_to_pixels(t)  # B, P, r, c
    gf = g.flatten(1)  # V, P
    # ||t - g||^2 = |t|^2 - 2 t.g + |g|^2
    d = -2 * torch.einsum("bprc,vp->bvrc", tc, gf) + (gf ** 2).sum(1).view(1, -1, 1, 1)
    return d.argmin(1)


def _scale(atlas: torch.Tensor) -> float:
    """Map tone 1.0 to the per-pixel ink level of the densest glyph (glyphs never fill a cell)."""
    return float(atlas.mean((1, 2)).max())


def aic_text(img_path: str | Path, cols: int, rows: int | None = None, extra: list[str] | None = None,
             bin_name="ascii-image-converter") -> str:
    """Run the real ascii-image-converter CLI (what the original dataset used)."""
    size = ["-d", f"{cols},{rows}"] if rows else ["-W", str(cols)]
    with tempfile.TemporaryDirectory() as d:
        cmd = [bin_name, str(img_path), *size, "--save-txt", d, "--only-save"] + (extra or [])
        subprocess.run(cmd, check=True, capture_output=True)
        txt = next(Path(d).glob("*.txt")).read_text(encoding="utf-8")
    return txt.strip("\n")


def aic_ids(img_path, cols: int, rows: int, chars: str, extra=None) -> torch.Tensor:
    txt = aic_text(img_path, cols, rows, extra)
    lines = (txt.split("\n") + [""] * rows)[:rows]
    return text_to_ids("\n".join(lines), chars, cols)


STROKES = "|/\\_-()<>^,'`.:;"
FILL_RAMP = " .:-=+*#%@"


def edge_map(lum: torch.Tensor, sigma: float = 1.5, pct: float = 0.88, floor: float = 0.06) -> torch.Tensor:
    """Thin binary edges (Canny-like: Sobel on smoothed luminance + non-max suppression). (B,1,H,W)."""
    y = gaussian_blur(lum, sigma)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=y.dtype, device=y.device).view(1, 1, 3, 3) / 8
    gx = F.conv2d(F.pad(y, (1, 1, 1, 1), mode="replicate"), kx)
    gy = F.conv2d(F.pad(y, (1, 1, 1, 1), mode="replicate"), kx.transpose(-1, -2))
    mag = torch.sqrt(gx ** 2 + gy ** 2)
    ang = torch.atan2(gy, gx)  # gradient direction
    q = (torch.round(ang / (math.pi / 4)) % 4).long()  # 0: horiz grad, 1: diag, 2: vert grad, 3: anti-diag
    p = F.pad(mag, (1, 1, 1, 1))
    H, W = mag.shape[-2:]
    nb = {0: ((0, 1), (0, -1)), 1: ((1, 1), (-1, -1)), 2: ((1, 0), (-1, 0)), 3: ((1, -1), (-1, 1))}
    keep = torch.zeros_like(mag, dtype=torch.bool)
    for k, ((dy1, dx1), (dy2, dx2)) in nb.items():
        n1 = p[..., 1 + dy1:1 + dy1 + H, 1 + dx1:1 + dx1 + W]
        n2 = p[..., 1 + dy2:1 + dy2 + H, 1 + dx2:1 + dx2 + W]
        keep |= (q == k) & (mag >= n1) & (mag >= n2)
    thr = torch.quantile(mag.flatten(1), pct, dim=1).clamp_min(floor).view(-1, 1, 1, 1)
    return (keep & (mag > thr)).to(lum.dtype)


def edge_ramp_ids(ink: torch.Tensor, lum: torch.Tensor, chars: str, min_edge_px: int = 6,
                  ramp: str = FILL_RAMP, gamma: float = 1.0) -> torch.Tensor:
    """Acerola-style: luminance ramp fill + stroke characters where the image has contours."""
    from .glyphs import glyph_atlas
    m = F.avg_pool2d(ink, (CELL_H, CELL_W))[:, 0] ** gamma
    fill_lut = torch.tensor([chars.index(c) for c in ramp], device=ink.device)
    ids = fill_lut[(m * len(ramp)).long().clamp(0, len(ramp) - 1)]
    E = edge_map(lum)
    cnt = F.avg_pool2d(E, (CELL_H, CELL_W))[:, 0] * CELL_H * CELL_W
    sub = STROKES
    atlas = glyph_atlas(sub).to(ink.device)
    g = gaussian_blur(atlas[:, None], 1.2)[:, 0]
    g = g / g.flatten(1).norm(dim=1).view(-1, 1, 1)
    e = cells_to_pixels(gaussian_blur(E, 1.2))  # B,P,r,c
    en = e / e.norm(dim=1, keepdim=True).clamp_min(1e-6)
    score = torch.einsum("bprc,vp->bvrc", en, g.flatten(1))  # cosine similarity of shapes
    best = score.argmax(1)
    stroke_lut = torch.tensor([chars.index(c) for c in sub], device=ink.device)
    use = cnt >= min_edge_px
    return torch.where(use, stroke_lut[best], ids)
