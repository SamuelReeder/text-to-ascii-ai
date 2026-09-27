"""Contact sheets: original image next to terminal-style renderings of each method's ASCII."""
import numpy as np
import torch
from PIL import Image, ImageDraw

from .core import render_ids
from .glyphs import glyph_atlas


def ids_to_pil(ids: torch.Tensor, atlas=None, polarity="dark", scale=1) -> Image.Image:
    atlas = glyph_atlas() if atlas is None else atlas.cpu()
    img = (render_ids(ids[None].cpu(), atlas)[0, 0] / atlas.max()).clamp(0, 1)
    if polarity == "light":
        img = 1 - img
    im = Image.fromarray((img.numpy() * 255).astype(np.uint8), "L")
    return im.resize((im.width * scale, im.height * scale), Image.NEAREST) if scale > 1 else im


def contact_sheet(rows, titles, pad=8, bg=40) -> Image.Image:
    """rows: list of lists of PIL images (same count per row). Tiles are scaled to a common height."""
    H = max(im.height for r in rows for im in r)
    rows = [[im.convert("RGB").resize((max(1, round(im.width * H / im.height)), H)) for im in r] for r in rows]
    ncol = len(rows[0])
    colw = [max(r[c].width for r in rows) for c in range(ncol)]
    W = sum(colw) + pad * (ncol + 1)
    out = Image.new("RGB", (W, 20 + len(rows) * (H + pad) + pad), (bg, bg, bg))
    d = ImageDraw.Draw(out)
    x = pad
    for c, t in enumerate(titles):
        d.text((x, 4), t, fill=(230, 230, 230))
        x += colw[c] + pad
    y = 20
    for r in rows:
        x = pad
        for c, im in enumerate(r):
            out.paste(im, (x, y))
            x += colw[c] + pad
        y += H + pad
    return out
