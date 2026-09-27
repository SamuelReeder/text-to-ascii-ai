#!/usr/bin/env python
"""Turn any image — or a text prompt — into legible ASCII art.

  python ascii.py photo.jpg                     # fits your terminal width
  python ascii.py photo.jpg -w 120 --color      # 24-bit color characters
  python ascii.py https://example.com/cat.png   # URLs work too
  python ascii.py -p "a lighthouse on a cliff"  # text -> image (SANA-Sprint) -> ASCII
"""
import argparse
import io
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_input(src: str):
    from PIL import Image
    if src == "-":
        return Image.open(io.BytesIO(sys.stdin.buffer.read()))
    if src.startswith(("http://", "https://")):
        import requests
        r = requests.get(src, timeout=30, headers={"User-Agent": "asciinet"})
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content))
    return Image.open(src)


def default_width() -> int:
    cols = shutil.get_terminal_size((100, 40)).columns
    return max(40, min(cols - 1, 160))


def quiet_libraries():
    """Hide model-loading progress bars and library warnings; errors and download progress still show."""
    import os
    import warnings
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
    warnings.filterwarnings("ignore")
    import diffusers
    import transformers
    from huggingface_hub.utils import enable_progress_bars
    from huggingface_hub.utils import logging as hub_logging
    hub_logging.set_verbosity_error()  # e.g. "unauthenticated requests" on every run
    transformers.utils.logging.disable_progress_bar()
    diffusers.utils.logging.disable_progress_bar()
    enable_progress_bars()  # the calls above also turn off Hugging Face download bars; keep those


def make_converter(args, Converter):
    """The trained network by default (local checkpoint, else the released weights); the pipeline for
    options only it has, or when no weights are available."""
    from asciiart.neural import HF_REPO, NeuralConverter, find_checkpoint
    pipeline_only = args.fill != "match" or args.no_strokes or args.focus in ("select", "on")
    engine = "pipeline" if args.engine == "auto" and pipeline_only else args.engine
    if engine in ("auto", "net"):
        ckpt = Path(args.checkpoint) if args.checkpoint else find_checkpoint()
        if ckpt is not None and ckpt.exists():
            return NeuralConverter(ckpt, args.device)
        if engine == "net":
            sys.exit(f"error: no AsciiNet checkpoint at {ckpt or 'checkpoints/asciinet.pt'} and none from {HF_REPO} "
                     "(train one with scripts/train.py, or use --engine pipeline)")
    return Converter(args.device, matte=not args.fast, selector=not args.fast)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", help="image path, URL, or '-' for stdin")
    ap.add_argument("-p", "--prompt", help="generate the picture from text instead of an image")
    ap.add_argument("-w", "--width", type=int, default=None, help="characters per line (default: terminal width)")
    ap.add_argument("--background", choices=["dark", "light"], default="dark",
                    help="your terminal/page background (default: dark)")
    ap.add_argument("--invert", choices=["auto", "yes", "no"], default="auto",
                    help="auto: plain white backgrounds become empty space")
    ap.add_argument("--focus", choices=["auto", "select", "on", "off"], default="auto",
                    help="fade background clutter around the main subject (auto: whenever one is found)")
    ap.add_argument("--fill", choices=["match", "ramp"], default="match",
                    help="match: glyphs chosen by shape; ramp: classic density ramp")
    ap.add_argument("--no-strokes", action="store_true", help="don't draw contours with / \\ | _ characters")
    ap.add_argument("--color", action="store_true", help="24-bit ANSI color output")
    ap.add_argument("-o", "--out", help="also write the plain-text art to this file")
    ap.add_argument("--png", help="also save a rendering of the art to this PNG")
    ap.add_argument("--save-image", help="prompt mode: save the generated picture here")
    ap.add_argument("--candidates", type=int, default=4, help="prompt mode: pictures to generate and rank")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--engine", choices=["auto", "net", "pipeline"], default="auto",
                    help="net: our trained AsciiNet (one forward pass); pipeline: BiRefNet matte + glyph "
                         "matching (the net's teacher). auto: net unless an option needs the pipeline")
    ap.add_argument("--checkpoint", default=None,
                    help="AsciiNet .pt or .safetensors (default: checkpoints/asciinet.pt, else the released weights)")
    ap.add_argument("--t2i", choices=["sana-sprint", "sd-turbo"], default="sana-sprint",
                    help="prompt mode: text->image model (sd-turbo needs less VRAM)")
    ap.add_argument("--fast", action="store_true", help="pipeline engine: skip the BiRefNet matte (no focus); the net is already fast")
    args = ap.parse_args()
    if not args.image and not args.prompt:
        ap.error("give an image (path/URL/-) or --prompt TEXT")

    quiet_libraries()
    import torch
    from asciiart.pipeline import Converter, Options

    t0 = time.time()
    opt = Options(cols=args.width or default_width(), background=args.background, invert=args.invert,
                  focus="off" if args.fast else args.focus, fill=args.fill, strokes=not args.no_strokes)
    conv = make_converter(args, Converter)

    if args.prompt:
        from asciiart.text2img import PromptScorer, TextToImage
        t2i = TextToImage(args.t2i, device=conv.device, keep_text_encoder=False)
        pics = t2i(args.prompt, n=args.candidates, seed=args.seed)
        del t2i
        torch.cuda.empty_cache()
        scorer = PromptScorer(device=conv.device)
        results = [conv.ids(p, opt, return_info=True) for p in pics]
        renders = []
        for ids, _ in results:
            from asciiart.core import render_ids
            r = (render_ids(ids[None], conv.atlas.cpu())[0] / conv.atlas.max().cpu()).clamp(0, 1)
            renders.append(r if opt.background == "dark" else 1 - r)
        best = int(scorer(args.prompt, renders).argmax())
        if args.save_image:
            pics[best].save(args.save_image)
        ids, info = results[best]
    else:
        try:
            img = load_input(args.image)
            img.load()
        except Exception as e:  # noqa: BLE001  (missing file, not an image, HTTP error, ...)
            sys.exit(f"error: couldn't read image {args.image!r}: {e}")
        ids, info = conv.ids(img, opt, return_info=True)

    from asciiart.core import CELL_H, ids_to_text
    # drop the empty rows above and below the art (focus often blanks the background there)
    x = info["x"]
    inked = (ids != conv.chars.index(" ")).any(1).nonzero()
    if len(inked):
        a, b = int(inked[0]), int(inked[-1]) + 1
        ids, x = ids[a:b], x[..., a * CELL_H:b * CELL_H, :]
    text = ids_to_text(ids, conv.chars)
    print(conv.colorize(ids, x, opt.background) if args.color else text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    if args.png:
        from asciiart.viz import ids_to_pil
        ids_to_pil(ids, conv.atlas.cpu(), opt.background, 2).save(args.png)
    print(f"[{opt.cols}x{ids.shape[0]} chars, {time.time() - t0:.1f}s]", file=sys.stderr)


if __name__ == "__main__":
    main()
