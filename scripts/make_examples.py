"""Build docs/examples/comparison.png: original | ascii-image-converter | the pipeline (teacher) |
AsciiNet at 80 columns | AsciiNet at 32 columns (small output)."""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asciiart.baselines import aic_ids  # noqa: E402
from asciiart.core import grid_rows, load_image  # noqa: E402
from asciiart.glyphs import PRINTABLE, glyph_atlas  # noqa: E402
from asciiart.neural import NeuralConverter  # noqa: E402
from asciiart.pipeline import Converter, Options  # noqa: E402
from asciiart.viz import ids_to_pil  # noqa: E402

IMAGES = ["data/images/eval/coco/000379.jpg", "data/images/eval/coco/000456.jpg",
          "data/images/eval/ffhq/000012.jpg", "data/images/eval/pokemon/000019.jpg"]


def main(cols=80, small=32, tile_h=420, out="docs/examples/comparison.png"):
    conv = Converter()
    net = NeuralConverter()
    atlas_p = glyph_atlas()
    rows_out = []
    for p in IMAGES:
        img = load_image(p)
        rows = grid_rows(img.width, img.height, cols)
        tiles = [img.resize((cols * 8, rows * 16)),
                 ids_to_pil(aic_ids(p, cols, rows, PRINTABLE), atlas_p),
                 ids_to_pil(conv.ids(img, Options(cols=cols)), conv.atlas.cpu()),
                 ids_to_pil(net.ids(img, Options(cols=cols)), net.atlas.cpu()),
                 ids_to_pil(net.ids(img, Options(cols=small)), net.atlas.cpu())]
        tiles = [t.convert("RGB").resize((round(t.width * tile_h / t.height), tile_h), Image.LANCZOS) for t in tiles]
        rows_out.append(tiles)
    colw = [max(r[c].width for r in rows_out) for c in range(len(rows_out[0]))]
    pad = 8
    W = sum(colw) + pad * (len(colw) + 1)
    H = 26 + len(rows_out) * (tile_h + pad) + pad
    sheet = Image.new("RGB", (W, H), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    x = pad
    for c, t in enumerate(["original", "ascii-image-converter", "pipeline (teacher)",
                           f"AsciiNet, {cols} cols", f"AsciiNet, {small} cols"]):
        d.text((x + 4, 7), t, fill=(220, 220, 220))
        x += colw[c] + pad
    y = 26
    for r in rows_out:
        x = pad
        for c, t in enumerate(r):
            sheet.paste(t, (x + (colw[c] - t.width) // 2, y))
            x += colw[c] + pad
        y += tile_h + pad
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, optimize=True)
    print(out, sheet.size)


if __name__ == "__main__":
    main()
