"""Robustness: every kind of input image converts without errors into a sane grid.

run: python -m pytest tests -q   (or: python tests/test_inputs.py)
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asciiart.core import ids_to_text, load_image  # noqa: E402
from asciiart.pipeline import Converter, Options  # noqa: E402

_conv = None


def conv():
    global _conv
    if _conv is None:
        _conv = Converter()
    return _conv


def shapes_image(w=320, h=240, mode="RGB"):
    im = Image.new("RGB", (w, h), (250, 250, 250))
    d = ImageDraw.Draw(im)
    d.ellipse((w * 0.2, h * 0.2, w * 0.6, h * 0.8), fill=(200, 40, 40))
    d.rectangle((w * 0.55, h * 0.3, w * 0.9, h * 0.7), outline=(0, 0, 0), width=4)
    return im.convert(mode)


def cases():
    rgba = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    ImageDraw.Draw(rgba).ellipse((40, 40, 216, 216), fill=(30, 30, 200, 255))
    gif = [shapes_image(), shapes_image().transpose(Image.FLIP_LEFT_RIGHT)]
    grad16 = Image.fromarray((np.linspace(0, 65535, 256 * 256).reshape(256, 256)).astype(np.uint16))
    return {
        "rgb": shapes_image(),
        "gray": shapes_image(mode="L"),
        "cmyk": shapes_image(mode="CMYK"),
        "palette": shapes_image(mode="P"),
        "transparent": rgba,
        "16bit": grad16,
        "float": Image.fromarray(np.random.rand(64, 64).astype(np.float32), "F"),
        "tiny": shapes_image(8, 6),
        "panorama": shapes_image(2000, 150),
        "tall": shapes_image(120, 2400),
        "black": Image.new("RGB", (300, 200), 0),
        "white": Image.new("RGB", (300, 200), (255, 255, 255)),
        "noise": Image.fromarray((np.random.rand(300, 300, 3) * 255).astype(np.uint8)),
        "huge": shapes_image(6000, 4000),
        "gif": (gif[0].save("/tmp/_t.gif", save_all=True, append_images=gif[1:]), Image.open("/tmp/_t.gif"))[1],
    }


def test_all_inputs():
    c = conv()
    for name, img in cases().items():
        for cols in (40, 100):
            for focus in ("select", "off"):
                for bg in ("dark", "light"):
                    ids = c.ids(img, Options(cols=cols, focus=focus, background=bg))
                    assert ids.shape[1] == cols, name
                    assert 1 <= ids.shape[0] <= 400, name
                    txt = ids_to_text(ids, c.chars)
                    assert all(32 <= ord(ch) < 127 for ch in txt.replace("\n", "")), name


def test_shapes_have_ink_and_background_is_empty():
    c = conv()
    ids = c.ids(shapes_image(640, 480), Options(cols=80, focus="off"))
    txt = ids_to_text(ids, c.chars).split("\n")
    assert sum(ch != " " for ch in "".join(txt)) > 200  # subject drawn
    corner = "".join(l[:6] for l in txt[:3])
    assert corner.strip() == ""  # white background became empty space on a dark terminal


def test_neural_engine_all_inputs():
    """The trained AsciiNet handles the same odd inputs, at small and large widths."""
    import pytest
    from asciiart.neural import DEFAULT_CKPT, NeuralConverter
    if not DEFAULT_CKPT.exists():
        pytest.skip("no AsciiNet checkpoint (train one with scripts/train.py)")
    c = NeuralConverter()
    for name, img in cases().items():
        for cols in (16, 40, 100):
            for focus in ("auto", "off"):
                for bg in ("dark", "light"):
                    ids = c.ids(img, Options(cols=cols, focus=focus, background=bg))
                    assert ids.shape[1] == cols, name
                    assert 1 <= ids.shape[0] <= 400, name
                    txt = ids_to_text(ids, c.chars)
                    assert all(32 <= ord(ch) < 127 for ch in txt.replace("\n", "")), name
    ids = c.ids(shapes_image(640, 480), Options(cols=80))
    assert (ids != c.chars.index(" ")).sum() > 100  # the shapes are drawn


def test_load_modes():
    for name, img in cases().items():
        out = load_image(img)
        assert out.mode == "RGB", name


if __name__ == "__main__":
    test_load_modes()
    test_shapes_have_ink_and_background_is_empty()
    test_all_inputs()
    test_neural_engine_all_inputs()
    print("ok")
