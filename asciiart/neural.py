"""Image -> ASCII with the trained AsciiNet (asciiart/model.py): one forward pass, no BiRefNet.

Same interface as pipeline.Converter (ids / text / colorize), so the CLI and the evaluation
scripts can use either engine.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .core import grid_rows, image_to_tensor, ids_to_text, load_image
from .glyphs import glyph_atlas
from .model import AsciiNet, ctx_size
from .pipeline import CHARSET, Converter, Options

DEFAULT_CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "asciinet.pt"


class NeuralConverter:
    def __init__(self, ckpt: str | Path = DEFAULT_CKPT, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.chars = ck.get("chars", CHARSET)
        self.model = AsciiNet(len(self.chars))
        self.model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
        self.model.to(self.device).eval()
        self.atlas = glyph_atlas(self.chars).to(self.device)
        self.step = ck.get("step")

    @torch.no_grad()
    def ids(self, img, opt: Options = Options(), rows: int | None = None, return_info=False):
        img = load_image(img)
        rows = rows or grid_rows(img.width, img.height, opt.cols)
        x = image_to_tensor(img, opt.cols, rows)[None].to(self.device)
        hc, wc = ctx_size(rows, opt.cols)
        ctx = torch.from_numpy(np.array(img.resize((wc, hc), Image.LANCZOS), dtype=np.float32) / 255)
        ctx = ctx.permute(2, 0, 1)[None].to(self.device)
        focus = "off" if opt.focus == "off" else "auto"
        cond = torch.tensor([AsciiNet.cond_index(opt.background, opt.invert, focus)], device=self.device)
        # cuDNN compiles bf16 kernels per new grid shape (seconds each); the native kernels need no warm-up
        with torch.backends.cudnn.flags(enabled=False), \
                torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            logits, _, _ = self.model(x, ctx, cond)
        ids = logits.float().argmax(1)[0].cpu()
        return (ids, {"x": x, "rows": rows}) if return_info else ids

    def text(self, img, opt: Options = Options()) -> str:
        return ids_to_text(self.ids(img, opt), self.chars)

    def colorize(self, ids, x, background="dark") -> str:
        return Converter.colorize(self, ids, x, background)

    def png(self, img, opt: Options = Options(), scale: int = 1):
        from .viz import ids_to_pil
        return ids_to_pil(self.ids(img, opt), self.atlas.cpu(), opt.background, scale)

