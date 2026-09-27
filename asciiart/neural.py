"""Image -> ASCII with the trained AsciiNet (asciiart/model.py): one forward pass, no BiRefNet.

Same interface as pipeline.Converter (ids / text / colorize), so the CLI and the evaluation
scripts can use either engine.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .core import grid_rows, image_to_tensor, ids_to_text, load_image
from .glyphs import glyph_atlas
from .model import AsciiNet, ctx_size
from .pipeline import CHARSET, Converter, Options

DEFAULT_CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "asciinet.pt"
HF_REPO = "SamuelReeder/asciinet"  # the released weights: model.safetensors + config.json
HF_REVISION = "55c4588acef906144ab5982049c6adf7c22a54a8"


def download_checkpoint() -> Path:
    """Fetch the config and weights from the same immutable release snapshot."""
    from huggingface_hub import snapshot_download
    snapshot = snapshot_download(
        HF_REPO, revision=HF_REVISION, allow_patterns=["config.json", "model.safetensors"]
    )
    return Path(snapshot) / "model.safetensors"


def find_checkpoint() -> Path | None:
    """checkpoints/asciinet.pt if you trained one, else the released weights (downloaded once, then cached)."""
    if DEFAULT_CKPT.exists():
        return DEFAULT_CKPT
    try:
        return download_checkpoint()
    except Exception as e:  # offline, or the Hub is unreachable
        print(f"warning: could not fetch AsciiNet weights from {HF_REPO} ({type(e).__name__})", file=sys.stderr)
        return None


def load_checkpoint(ckpt: str | Path) -> tuple[dict, str, int | None]:
    """(state dict, charset, training step) from a train.py .pt or a released .safetensors (+ config.json beside it)."""
    ckpt = Path(ckpt)
    if ckpt.suffix == ".safetensors":
        from safetensors.torch import load_file
        cfg = json.loads((ckpt.parent / "config.json").read_text())
        return load_file(ckpt), cfg["chars"], cfg.get("step")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    return ck["ema"] if "ema" in ck else ck["model"], ck.get("chars", CHARSET), ck.get("step")


class NeuralConverter:
    def __init__(self, ckpt: str | Path | None = None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = ckpt or find_checkpoint()
        if ckpt is None:
            raise FileNotFoundError(f"no AsciiNet checkpoint at {DEFAULT_CKPT}, and none downloaded from {HF_REPO}")
        state, self.chars, self.step = load_checkpoint(ckpt)
        self.model = AsciiNet(len(self.chars))
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.atlas = glyph_atlas(self.chars).to(self.device)

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

