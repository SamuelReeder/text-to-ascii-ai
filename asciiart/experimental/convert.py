"""Inference: any image -> ASCII art with a trained AsciiNet checkpoint."""
from pathlib import Path

import torch
import torch.nn.functional as F

from ..core import (CELL_H, CELL_W, auto_polarity, grid_rows, ids_to_text, image_to_tensor, load_image,
                   tone_map)
from ..glyphs import glyph_atlas
from .model import AsciiNet

DEFAULT_CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "asciinet.pt"


class AsciiConverter:
    def __init__(self, ckpt: str | Path = DEFAULT_CKPT, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.chars = ck["chars"]
        self.model = AsciiNet(len(self.chars), **ck.get("model_kwargs", {})).to(self.device).eval()
        self.model.load_state_dict(ck["model"])
        self.atlas = glyph_atlas(self.chars)

    def _prepare(self, img, cols, rows, background, invert):
        img = load_image(img)
        x = image_to_tensor(img, cols, rows)[None].to(self.device)
        if invert == "auto":
            inv = auto_polarity(x, background)
        else:
            flip = invert in (True, "yes", "on", "true")
            inv = torch.tensor([(background == "light") != flip], device=self.device)
        ink = tone_map(x, inv)
        rgb_adj = torch.where(inv.view(-1, 1, 1, 1), 1 - x, x)
        return x, ink, rgb_adj

    @torch.no_grad()
    def ids(self, img, cols: int = 80, rows: int | None = None, background: str = "dark",
            invert="auto", polarity: str | None = None) -> torch.Tensor:
        """Returns (rows, cols) character ids. `polarity` is an alias for `background`."""
        background = polarity or background
        _, ink, rgb_adj = self._prepare(img, cols, rows, background, invert)
        with torch.autocast(self.device if self.device != "mps" else "cpu", dtype=torch.bfloat16,
                            enabled=self.device == "cuda"):
            logits = self.model(ink, rgb_adj)
        return logits.float().argmax(1)[0].cpu()

    def text(self, img, cols: int = 80, **kw) -> str:
        return ids_to_text(self.ids(img, cols, **kw), self.chars)

    @torch.no_grad()
    def ansi(self, img, cols: int = 80, background: str = "dark", invert="auto", rows=None) -> str:
        """24-bit color terminal output: characters from the model, color from the image."""
        x, _, _ = self._prepare(img, cols, rows, background, invert)
        ids = self.ids(img, cols, rows=rows, background=background, invert=invert)
        col = F.avg_pool2d(x, (CELL_H, CELL_W))[0].permute(1, 2, 0).clamp(0, 1)
        # brighten dim colors so thin glyph strokes stay visible
        col = (col / col.amax(-1, keepdim=True).clamp_min(1e-3) * col.amax(-1, keepdim=True).clamp_min(0.35)).cpu()
        lines = []
        for r in range(ids.shape[0]):
            parts = []
            for c in range(ids.shape[1]):
                ch = self.chars[ids[r, c]]
                if ch == " ":
                    parts.append(" ")
                else:
                    R, G, B = (int(v * 255) for v in col[r, c])
                    parts.append(f"\x1b[38;2;{R};{G};{B}m{ch}")
            lines.append("".join(parts).rstrip() + "\x1b[0m")
        return "\n".join(lines)

    def png(self, img, cols: int = 80, background: str = "dark", scale: int = 2, **kw):
        from ..viz import ids_to_pil
        return ids_to_pil(self.ids(img, cols, background=background, **kw), self.atlas, background, scale)
