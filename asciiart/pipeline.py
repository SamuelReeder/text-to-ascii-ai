"""Image -> legible ASCII art.

    image ─► subject matte (BiRefNet) ─► focus + tone map ─► per-cell glyph matching ─┐
                                      └► color contours + silhouette ─► stroke chars ─┴► ASCII

* Subject focus: when the matte finds a subject, background clutter becomes empty space, the
  subject keeps a minimum ink level and its silhouette is always drawn as a contour.
* Tone: robust contrast stretch + local contrast boost; polarity is chosen so a plain background
  maps to empty space on the user's terminal.
* Fill: every cell gets the glyph whose (lightly blurred) shape and mean ink best match that cell
  of the tone image, from a letter-free set, so the art never turns into "text soup".
* Contours: color-aware Canny-style edges; contour cells get the stroke character whose shape
  best matches the local edge pattern (| / \\ _ - ( ) < > ^ ...).
"""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .core import (CELL_H, CELL_W, cells_to_pixels, gaussian_blur, grid_rows, ids_to_text, image_to_tensor,
                   load_image, luminance, tone_map)
from .glyphs import glyph_atlas

RAMP = " .:-=+*#%@"
STROKES = "|/\\_-()<>^,'`.:;"
# Fill characters for shape matching: no letters, digits or brackets (those read as text/code).
FILL = " .'`,:;-_~\"^!*+=<>/\\|()#%@"
CHARSET = "".join(dict.fromkeys(RAMP + STROKES + FILL))


@dataclass
class Options:
    cols: int = 100
    background: str = "dark"  # the terminal's background: "dark" or "light"
    invert: str = "auto"  # "auto" | "yes" | "no"
    focus: str = "auto"  # "auto" | "select" | "on" | "off": fade background around the main subject.
    # auto: focus whenever the matte finds a subject (>=1.5% of the frame), else use the full image;
    # select: render with and without focus and keep the one a CLIP model finds closest to the image.
    keep_background: float = 0.0  # 0..1 ink kept for the background when focusing
    local_contrast: float = 0.6
    edge_sensitivity: float = 0.88  # quantile of gradient magnitude that counts as a contour
    edge_sigma: float = 1.5  # smoothing (pixels; a cell is 8x16) before contour detection
    min_edge_px: int = 5  # edge pixels (of 128) for a cell to become a stroke
    min_stroke_match: float = 0.45  # cosine similarity between edge pattern and stroke glyph
    fill: str = "match"  # "match": per-cell glyph shape matching; "ramp": density ramp only
    tone_weight: float = 0.5  # match mode: extra weight on matching each cell's mean ink (0 = shape only)
    strokes: bool = True
    coarse_sigma: float = 0.0  # >0: a contour must also exist at this coarser scale (suppresses texture)
    fill_sigma: float = 1.0  # blur of the tone target before glyph matching


def bright_background(x: torch.Tensor) -> torch.Tensor:
    """(B,) True where the image border is mostly near-white (product shots, line art, logos)."""
    return _border_frac(x, lambda y: y > 0.82) > 0.6


def dark_background(x: torch.Tensor) -> torch.Tensor:
    return _border_frac(x, lambda y: y < 0.18) > 0.6


def _border_frac(x, pred):
    y = luminance(x)[:, 0]
    H, W = y.shape[-2:]
    bh, bw = max(1, H // 16), max(1, W // 16)
    border = torch.cat([y[:, :bh].flatten(1), y[:, -bh:].flatten(1),
                        y[:, :, :bw].flatten(1), y[:, :, -bw:].flatten(1)], 1)
    return pred(border).float().mean(1)


def color_edges(rgb: torch.Tensor, sigma: float = 1.5, pct: float = 0.88, floor: float = 0.05,
                weight: torch.Tensor | None = None, silhouette: torch.Tensor | None = None) -> torch.Tensor:
    """Thin binary contours from a color structure tensor (luminance + opponent chroma). (B,1,H,W)."""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    chans = [luminance(rgb), 0.5 * (r - g), 0.5 * ((r + g) / 2 - b)]
    if silhouette is not None:  # subject matte: its boundary is always a contour
        chans.append(silhouette)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=rgb.dtype, device=rgb.device).view(1, 1, 3, 3) / 8
    jxx = jyy = jxy = 0
    for c in chans:
        c = gaussian_blur(c, sigma)
        p = F.pad(c, (1, 1, 1, 1), mode="replicate")
        gx, gy = F.conv2d(p, kx), F.conv2d(p, kx.transpose(-1, -2))
        jxx, jyy, jxy = jxx + gx * gx, jyy + gy * gy, jxy + gx * gy
    tr, det = jxx + jyy, jxx * jyy - jxy * jxy
    lam = tr / 2 + torch.sqrt((tr / 2) ** 2 - det).nan_to_num(0)
    mag = torch.sqrt(lam.clamp_min(0))
    ang = 0.5 * torch.atan2(2 * jxy, jxx - jyy)  # dominant gradient direction
    q = (torch.round(ang / (math.pi / 4)) % 4).long()
    p = F.pad(mag, (1, 1, 1, 1))
    H, W = mag.shape[-2:]
    keep = torch.zeros_like(mag, dtype=torch.bool)
    for k, ((dy1, dx1), (dy2, dx2)) in {0: ((0, 1), (0, -1)), 1: ((1, 1), (-1, -1)),
                                         2: ((1, 0), (-1, 0)), 3: ((1, -1), (-1, 1))}.items():
        n1 = p[..., 1 + dy1:1 + dy1 + H, 1 + dx1:1 + dx1 + W]
        n2 = p[..., 1 + dy2:1 + dy2 + H, 1 + dx2:1 + dx2 + W]
        keep |= (q == k) & (mag >= n1) & (mag >= n2)
    if weight is not None:
        # keep contours on and just outside the subject; ignore the (faded) background
        mag = mag * (gaussian_blur(weight, 2.0) > 0.05)
        sel = weight.flatten(1) > 0.5
        flat = torch.where(sel, mag.flatten(1), torch.nan)
        thr = torch.nanquantile(flat, pct, dim=1).nan_to_num(1.0)
    else:
        thr = torch.quantile(mag.flatten(1), pct, dim=1)
    thr = thr.clamp_min(floor).view(-1, 1, 1, 1)
    return (keep & (mag > thr)).to(rgb.dtype)


class Converter:
    def __init__(self, device: str | None = None, matte: bool = True, selector: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.chars = CHARSET
        self.atlas = glyph_atlas(self.chars).to(self.device)
        self.ramp_ids = torch.tensor([self.chars.index(c) for c in RAMP], device=self.device)
        self.stroke_ids = torch.tensor([self.chars.index(c) for c in STROKES], device=self.device)
        g = gaussian_blur(glyph_atlas(STROKES).to(self.device)[:, None], 1.2)[:, 0].flatten(1)
        self.stroke_tmpl = g / g.norm(dim=1, keepdim=True)
        self.fill_ids = torch.tensor([self.chars.index(c) for c in FILL], device=self.device)
        fa = glyph_atlas(FILL).to(self.device)
        self.fill_blur = gaussian_blur(fa[:, None], 1.0)[:, 0].flatten(1)  # V,P
        self.fill_mean = fa.flatten(1).mean(1)
        self.ink_scale = float(glyph_atlas(self.chars).mean((1, 2)).max())
        self._matte = None
        self._matte_enabled = matte
        self._selector = None
        self._selector_enabled = selector

    @property
    def selector(self):
        if self._selector is None and self._selector_enabled:
            from .clip import ClipEmbedder
            self._selector = ClipEmbedder(device=self.device)
        return self._selector

    @property
    def matte(self):
        if self._matte is None and self._matte_enabled:
            from .subject import SubjectMatte
            self._matte = SubjectMatte(self.device)
        return self._matte

    def prepare(self, img, opt: Options, rows: int | None = None, matte=None):
        """Returns dict with the rgb grid image, ink target, edge weight and diagnostics."""
        img = load_image(img)
        rows = rows or grid_rows(img.width, img.height, opt.cols)
        x = image_to_tensor(img, opt.cols, rows)[None].to(self.device)
        m = matte
        if opt.focus != "off" and (m is not None or self.matte is not None):
            m = self.matte(img) if m is None else m
            m = F.interpolate(m.to(self.device).view(1, 1, *m.shape[-2:]), size=x.shape[-2:], mode="bilinear",
                              antialias=True)
        p = self.prepare_tensor(x, m, opt)
        p["rows"] = rows
        return p

    def prepare_tensor(self, x: torch.Tensor, m: torch.Tensor | None, opt: Options):
        """Batched core of prepare(): x (B,3,H,W) grid images, m (B,1,H,W) subject mattes at the same
        size (or None). Used directly to label training batches for the neural model."""
        from .subject import apply_focus
        B = x.shape[0]
        if opt.invert == "auto":
            inv = bright_background(x) if opt.background == "dark" else ~dark_background(x)
        else:
            flip = opt.invert in (True, "yes", "on", "true")
            inv = torch.full((B,), (opt.background == "light") != flip, device=x.device)
        focused = torch.zeros(B, dtype=torch.bool, device=x.device)
        if m is not None and opt.focus != "off":
            # focus whenever the matte found a real subject (>= 1.5% of the frame), see matte_is_useful
            focused = torch.ones_like(focused) if opt.focus == "on" else (m > 0.5).float().mean((1, 2, 3)) >= 0.015
        ink = tone_map(x, inv, opt.local_contrast)
        if focused.any():
            ink_f = apply_focus(tone_map(x, inv, opt.local_contrast, mask=m), m, keep_bg=opt.keep_background)
            ink = torch.where(focused.view(B, 1, 1, 1), ink_f, ink)
        return {"x": x, "ink": ink, "matte": m, "inverted": inv, "focused": focused}

    def _edges(self, p, sigma, pct):
        """Contours (B,1,H,W). Focused images: the matte's silhouette is always a contour and the
        (faded) background's texture is ignored; others: contours over the whole image."""
        x, m, f = p["x"], p["matte"], p["focused"]
        E = torch.zeros_like(x[:, :1])
        if (~f).any():
            E[~f] = color_edges(x[~f], sigma=sigma, pct=pct)
        if f.any():
            mf = m[f]
            E[f] = color_edges(x[f] * mf, sigma=sigma, pct=pct, weight=mf, silhouette=mf)
        return E

    @torch.no_grad()
    def ids(self, img, opt: Options = Options(), rows: int | None = None, return_info=False):
        if opt.focus == "select":
            return self._select(img, opt, rows, return_info)
        return self._convert(self.prepare(img, opt, rows), opt, return_info)

    def _select(self, img, opt, rows, return_info):
        from dataclasses import replace

        from .core import render_ids
        img = load_image(img)
        if self.matte is None or self.selector is None:
            return self.ids(img, replace(opt, focus="off"), rows, return_info)
        m = self.matte(img).to(self.device)
        area = m.mean().item()
        cands = [self._convert(self.prepare(img, replace(opt, focus="off"), rows), opt, True)]
        if 0.02 <= area <= 0.95:
            cands.append(self._convert(self.prepare(img, replace(opt, focus="on"), rows, matte=m), opt, True))
        if len(cands) == 1:
            return cands[0] if return_info else cands[0][0]
        x = cands[0][1]["x"]
        target = self.selector.embed(x)
        renders = torch.cat([render_ids(c[0][None].to(self.device), self.atlas) for c in cands]) / self.atlas.max()
        if opt.background == "light":
            renders = 1 - renders
        sims = sum((e @ t.T)[:, 0] for e, t in zip(self.selector.embed(renders.clamp(0, 1)), target))
        best = int(sims.argmax())
        ids, info = cands[best]
        info["candidate_scores"] = sims.tolist()
        return (ids, info) if return_info else ids

    def _fill_match(self, ink, tone_weight, sigma=1.0):
        """Per-cell nearest fill glyph: shape term (lightly blurred L2) + tone term on the cell mean."""
        t = gaussian_blur(ink, sigma) * self.ink_scale
        tc = cells_to_pixels(t)  # B,P,r,c
        P = tc.shape[1]
        d = -2 * torch.einsum("bprc,vp->bvrc", tc, self.fill_blur) + (self.fill_blur ** 2).sum(1).view(1, -1, 1, 1)
        if tone_weight:
            tm = F.avg_pool2d(ink, (CELL_H, CELL_W)) * self.ink_scale  # B,1,r,c
            d = d + tone_weight * P * (self.fill_mean.view(1, -1, 1, 1) - tm) ** 2
        return self.fill_ids[d.argmin(1)]

    def _convert(self, p, opt, return_info=False):
        ids = self.convert_batch(p, opt)[0].cpu()
        return (ids, p) if return_info else ids

    def convert_batch(self, p, opt) -> torch.Tensor:
        """prepare_tensor() output -> ids (B, rows, cols) on the device."""
        ink = p["ink"]
        if opt.fill == "match":
            ids = self._fill_match(ink, opt.tone_weight, opt.fill_sigma)
        else:
            m = F.avg_pool2d(ink, (CELL_H, CELL_W))[:, 0]
            ids = self.ramp_ids[(m * len(RAMP)).long().clamp(0, len(RAMP) - 1)]
        if not opt.strokes:
            return ids
        E = self._edges(p, opt.edge_sigma, opt.edge_sensitivity)
        cnt = F.avg_pool2d(E, (CELL_H, CELL_W))[:, 0] * CELL_H * CELL_W
        e = cells_to_pixels(gaussian_blur(E, 1.2))
        e = e / e.norm(dim=1, keepdim=True).clamp_min(1e-6)
        score = torch.einsum("bprc,vp->bvrc", e, self.stroke_tmpl)
        best = score.max(1)
        use = (cnt >= opt.min_edge_px) & (best.values >= opt.min_stroke_match)
        if opt.coarse_sigma > 0:
            Ec = self._edges(p, opt.coarse_sigma, opt.edge_sensitivity)
            near = F.max_pool2d(F.max_pool2d(Ec, (CELL_H, CELL_W)), 3, stride=1, padding=1)[:, 0] > 0
            use = use & near
        return torch.where(use, self.stroke_ids[best.indices], ids)

    def text(self, img, opt: Options = Options()) -> str:
        return ids_to_text(self.ids(img, opt), self.chars)

    @torch.no_grad()
    def ansi(self, img, opt: Options = Options()) -> str:
        """24-bit color output: characters from the converter, color from the image."""
        ids, p = self.ids(img, opt, return_info=True)
        return self.colorize(ids, p["x"], opt.background)

    def colorize(self, ids: torch.Tensor, x: torch.Tensor, background: str = "dark") -> str:
        """ids (rows, cols) + the grid image x (1,3,H,W) -> ANSI 24-bit colored text."""
        col = F.avg_pool2d(x, (CELL_H, CELL_W))[0].permute(1, 2, 0).clamp(0, 1)
        mx = col.amax(-1, keepdim=True).clamp_min(1e-3)
        # keep colors visible against the terminal: brighten dim ones on dark, darken pale ones on light
        target = mx.clamp_min(0.45) if background == "dark" else mx.clamp_max(0.6)
        col = (col / mx * target).cpu()
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

    def png(self, img, opt: Options = Options(), scale: int = 1):
        from .viz import ids_to_pil
        return ids_to_pil(self.ids(img, opt), self.atlas.cpu(), opt.background, scale)
