"""Image loading without torch (so data-preparation workers stay small)."""
import numpy as np
from PIL import Image, ImageOps


def load_image(path_or_img) -> Image.Image:
    """Load any image as 8-bit RGB (first frame of animations, 16-bit/float scaled, alpha composited)."""
    img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:  # noqa: BLE001  (corrupt EXIF shouldn't stop a conversion)
        pass
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.asarray(img, dtype=np.float32)
        lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
        return Image.fromarray((a * 255).astype(np.uint8), "L").convert("RGB")
    if img.mode == "PA" or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA", "La", "RGBa"):
        img = img.convert("RGBA")
        # Composite transparent pixels onto the color that best contrasts the opaque content.
        arr = np.asarray(img, dtype=np.float32) / 255.0  # float32: huge images stay manageable
        a, rgb = arr[..., 3:4], arr[..., :3]
        fg_lum = (rgb.mean(-1, keepdims=True) * a).sum() / max(a.sum(), 1e-6)
        bg = 0.0 if fg_lum > 0.5 else 1.0
        out = rgb * a + bg * (1 - a)
        return Image.fromarray((out * 255).astype(np.uint8), "RGB")
    return img.convert("RGB")
