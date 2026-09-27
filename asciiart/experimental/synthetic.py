"""Procedural training images: big text, digits, geometric shapes, and line art.

These teach the converter to keep lettering, outlines and flat graphics legible, which
photo datasets under-represent.
"""
import glob
import math
import random
import string

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

WORDS = """hello world ascii art cat dog bird fish tree house car sun moon star love home
code python open close stop go yes no fire water earth wind light dark music game play
coffee pizza apple banana robot ghost heart smile happy rain snow cloud boat train plane
rocket space city road bridge mountain river ocean forest flower garden book paper pen
king queen castle dragon sword shield magic hero zero one two three four five six seven
eight nine ten hi ok wow cool nice sale open 24/7 exit hotel cafe bar taxi police bank
news jazz rock pop art lab ai gpu linux unix hack data""".split()

_FONTS = None


def fonts():
    global _FONTS
    if _FONTS is None:
        fs = [f for f in glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
              if not any(k in f.lower() for k in ("emoji", "symbol", "math"))]
        _FONTS = fs or [None]
    return _FONTS


def _font(size):
    f = random.choice(fonts())
    try:
        return ImageFont.truetype(f, size) if f else ImageFont.load_default()
    except OSError:
        return ImageFont.load_default()


def _rand_color(rng, dark=None):
    if dark is None:
        dark = rng.random() < 0.5
    base = rng.randint(0, 80) if dark else rng.randint(170, 255)
    return tuple(int(np.clip(base + rng.randint(-40, 40), 0, 255)) for _ in range(3))


def _background(rng, W, H, photo=None):
    kind = rng.random()
    if photo is not None and kind < 0.25:
        bg = photo.convert("RGB").resize((W, H)).filter(ImageFilter.GaussianBlur(rng.uniform(2, 8)))
        return bg, None
    dark = rng.random() < 0.45
    c1 = _rand_color(rng, dark)
    if kind < 0.7:
        return Image.new("RGB", (W, H), c1), dark
    c2 = _rand_color(rng, dark)
    t = np.linspace(0, 1, H if rng.random() < 0.5 else W)
    grad = np.outer(t, np.ones(W if len(t) == H else H))
    if len(t) == W:
        grad = grad.T
    arr = (np.array(c1)[None, None] * (1 - grad[..., None]) + np.array(c2)[None, None] * grad[..., None])
    return Image.fromarray(arr.astype(np.uint8)), dark


def _bezier(rng, W, H, n=40):
    pts = [(rng.uniform(0, W), rng.uniform(0, H)) for _ in range(4)]
    out = []
    for i in range(n + 1):
        t = i / n
        a = [(1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3]
        out.append((sum(a[k] * pts[k][0] for k in range(4)), sum(a[k] * pts[k][1] for k in range(4))))
    return out


def make_synthetic(rng: random.Random, photo=None) -> Image.Image:
    W = rng.choice([320, 384, 448, 512])
    H = int(W * rng.uniform(0.45, 1.3))
    mode = rng.random()
    if mode < 0.2:  # line art: dark strokes on light paper (or inverse)
        dark_bg = rng.random() < 0.25
        img = Image.new("RGB", (W, H), (15, 15, 15) if dark_bg else (245, 245, 240))
        fg = (235, 235, 235) if dark_bg else (20, 20, 20)
        d = ImageDraw.Draw(img)
        for _ in range(rng.randint(2, 8)):
            r = rng.random()
            lw = rng.randint(2, max(3, W // 60))
            if r < 0.4:
                d.line(_bezier(rng, W, H), fill=fg, width=lw, joint="curve")
            elif r < 0.7:
                x0, y0 = rng.uniform(0, W * 0.7), rng.uniform(0, H * 0.7)
                x1, y1 = x0 + rng.uniform(W * 0.1, W * 0.5), y0 + rng.uniform(H * 0.1, H * 0.5)
                (d.ellipse if rng.random() < 0.5 else d.rectangle)((x0, y0, x1, y1), outline=fg, width=lw)
            else:
                cx, cy, rad = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(W * 0.05, W * 0.3)
                k = rng.randint(3, 8)
                pts = [(cx + rad * math.cos(2 * math.pi * i / k + 0.3), cy + rad * math.sin(2 * math.pi * i / k + 0.3))
                       for i in range(k)]
                d.polygon(pts, outline=fg, width=lw)
        return img
    bg, dark = _background(rng, W, H, photo)
    d = ImageDraw.Draw(bg)
    n = rng.randint(1, 5)
    for _ in range(n):
        r = rng.random()
        fg = _rand_color(rng, dark=(not dark) if dark is not None else None)
        if r < 0.45:  # text
            word = rng.choice(WORDS) if rng.random() < 0.7 else "".join(
                rng.choice(string.ascii_letters + string.digits) for _ in range(rng.randint(1, 6)))
            if rng.random() < 0.3:
                word = word.upper()
            size = int(H * rng.uniform(0.15, 0.45))
            font = _font(size)
            tw = d.textlength(word, font=font)
            if tw > W * 0.95:
                size = max(8, int(size * W * 0.95 / tw))
                font = _font(size)
                tw = d.textlength(word, font=font)
            x = rng.uniform(0, max(1, W - tw))
            y = rng.uniform(-0.1 * size, max(1, H - size * 1.1))
            d.text((x, y), word, fill=fg, font=font,
                   stroke_width=rng.choice([0, 0, 0, 1, 2]), stroke_fill=fg)
        elif r < 0.85:  # filled / outlined shapes
            x0, y0 = rng.uniform(-W * 0.1, W * 0.8), rng.uniform(-H * 0.1, H * 0.8)
            x1, y1 = x0 + rng.uniform(W * 0.1, W * 0.6), y0 + rng.uniform(H * 0.1, H * 0.6)
            fill = fg if rng.random() < 0.6 else None
            lw = rng.randint(2, max(3, W // 40))
            k = rng.random()
            if k < 0.35:
                d.ellipse((x0, y0, x1, y1), fill=fill, outline=fg, width=lw)
            elif k < 0.65:
                d.rectangle((x0, y0, x1, y1), fill=fill, outline=fg, width=lw)
            else:
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                rad = min(x1 - x0, y1 - y0) / 2
                m = rng.randint(3, 6)
                star = rng.random() < 0.4
                pts = []
                for i in range(m * (2 if star else 1)):
                    rr = rad * (0.45 if star and i % 2 else 1.0)
                    a = math.pi * 2 * i / (m * (2 if star else 1)) - math.pi / 2
                    pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
                d.polygon(pts, fill=fill, outline=fg, width=lw)
        else:  # thick strokes
            d.line(_bezier(rng, W, H), fill=fg, width=rng.randint(3, max(4, W // 25)), joint="curve")
    if rng.random() < 0.3:
        bg = bg.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    return bg
