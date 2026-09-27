"""Subject focus: a salient-object matte (BiRefNet) lets the converter drop background clutter.

At 60-120 characters wide, busy backgrounds (walls, foliage, crowds) turn into character noise
that hides the subject. When a confident, reasonably sized subject is found, its ink is kept
and the background fades to empty space; otherwise the full image is used unchanged.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BIREFNET = ("ZhengPeng7/BiRefNet", "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class SubjectMatte:
    def __init__(self, device="cuda", size=1024):
        from transformers import AutoModelForImageSegmentation
        self.model = AutoModelForImageSegmentation.from_pretrained(
            BIREFNET[0], revision=BIREFNET[1], trust_remote_code=True)
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = self.model.to(device).eval().to(self.dtype)
        self.device, self.size = device, size

    @torch.no_grad()
    def __call__(self, img: Image.Image) -> torch.Tensor:
        """RGB PIL -> (H, W) matte in [0,1] at the image's resolution."""
        x = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None]
        xi = F.interpolate(x, size=(self.size, self.size), mode="bilinear", antialias=True)
        xi = ((xi - MEAN) / STD).to(self.device, self.dtype)
        m = self.model(xi)[-1].float().sigmoid()
        return F.interpolate(m, size=x.shape[-2:], mode="bilinear")[0, 0].cpu()


def matte_is_useful(m: torch.Tensor, min_area: float = 0.015) -> bool:
    """Focus whenever the matte found a real subject. (With a VLM judging legibility, focusing
    helped even for uncertain/large mattes; it only hurts when there is no subject at all.)"""
    return (m > 0.5).float().mean().item() >= min_area


def apply_focus(ink: torch.Tensor, matte: torch.Tensor, keep_bg: float = 0.0, floor: float = 0.12) -> torch.Tensor:
    """ink (B,1,H,W) in [0,1]; matte (H,W) or (B,1,H,W), resized to ink. Subject keeps a minimum ink
    level so dark subjects still read as a silhouette; background ink is scaled by keep_bg."""
    m = matte.to(ink).view(-1, 1, *matte.shape[-2:])
    if m.shape[-2:] != ink.shape[-2:]:
        m = F.interpolate(m, size=ink.shape[-2:], mode="bilinear", antialias=True)
    subj = floor + (1 - floor) * ink
    return subj * m + ink * keep_bg * (1 - m)
