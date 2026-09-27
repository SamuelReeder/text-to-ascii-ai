"""Glyph atlas: rasterized coverage maps for each character in a monospace font."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

CELL_W, CELL_H = 8, 16  # render resolution of one character cell (terminal cells are ~1:2)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/Library/Fonts/DejaVuSansMono.ttf",
    "C:/Windows/Fonts/consola.ttf",
]

# All printable ASCII, space first so index 0 is "no ink".
PRINTABLE = " " + "".join(chr(c) for c in range(33, 127))


def find_font() -> str:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return p
    raise FileNotFoundError("No monospace font found; pass font_path explicitly.")


@lru_cache(maxsize=8)
def _atlas_np(chars: str, font_path: str, cell_w: int, cell_h: int, ss: int) -> np.ndarray:
    """Rasterize `chars` at `ss`x supersampling then box-downsample to (cell_h, cell_w) coverage in [0,1]."""
    W, H = cell_w * ss, cell_h * ss
    probe = ImageFont.truetype(font_path, 100)
    adv = probe.getlength("M") / 100.0
    ascent, descent = probe.getmetrics()
    line = (ascent + descent) / 100.0
    size = min(W / adv, H / line)
    font = ImageFont.truetype(font_path, max(1, int(round(size))))
    ascent, descent = font.getmetrics()
    y0 = (H - (ascent + descent)) // 2
    x0 = int(round((W - font.getlength("M")) / 2))
    out = np.zeros((len(chars), cell_h, cell_w), dtype=np.float32)
    for i, ch in enumerate(chars):
        img = Image.new("L", (W, H), 0)
        ImageDraw.Draw(img).text((x0, y0), ch, fill=255, font=font)
        img = img.resize((cell_w, cell_h), Image.BOX)
        out[i] = np.asarray(img, dtype=np.float32) / 255.0
    return out


def glyph_atlas(chars: str = PRINTABLE, font_path: str | None = None,
                cell_w: int = CELL_W, cell_h: int = CELL_H, ss: int = 8) -> torch.Tensor:
    """(V, cell_h, cell_w) float tensor of ink coverage per glyph."""
    return torch.from_numpy(_atlas_np(chars, font_path or find_font(), cell_w, cell_h, ss).copy())
