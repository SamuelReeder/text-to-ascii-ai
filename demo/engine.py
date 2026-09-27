"""Bounded inputs and shared inference for the Gradio demo; no models load on import."""
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from asciiart.core import ids_to_text, load_image, render_ids
from asciiart.pipeline import Options

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000
WIDTHS = (32, 64, 80, 120)


def validate_width(width):
    if width not in WIDTHS:
        raise ValueError("Choose 32, 64, 80, or 120 columns.")
    return int(width)


def validate_prompt(prompt, seed):
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 500:
        raise ValueError("Enter a prompt between 1 and 500 characters.")
    if isinstance(seed, bool) or not isinstance(seed, (int, float)) or not 0 <= seed <= 2**31 - 1 or int(seed) != seed:
        raise ValueError("Seed must be a whole number between 0 and 2147483647.")
    return prompt.strip(), int(seed)


def read_upload(path):
    if not path:
        raise ValueError("Upload an image first.")
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError("Upload an image smaller than 10 MB.")
    try:
        with Image.open(path) as source:
            w, h = source.size
            if w * h > MAX_PIXELS or max(w, h) > 4 * min(w, h):
                raise ValueError("Use an image up to 16 megapixels with an aspect ratio between 1:4 and 4:1.")
            if source.format not in {"JPEG", "PNG", "WEBP", "GIF", "BMP"}:
                raise ValueError("Use a PNG, JPEG, WebP, GIF, or BMP image.")
            # Header checks happen before decoding. Load the first frame and
            # composite transparency using the same rules as the CLI.
            image = load_image(source)
            image.thumbnail((1024, 1024))
            return image
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("That file could not be read as an image.") from exc


def image_to_ascii(converter, path, width):
    width = validate_width(width)
    image = read_upload(path)
    return image, converter.text(image, Options(cols=width))


def prompt_to_ascii(converter, generator, scorer, prompt, width, seed):
    prompt, seed = validate_prompt(prompt, seed)
    width = validate_width(width)
    pictures = generator(prompt, n=2, seed=seed)
    options = Options(cols=width)
    ids = [converter.ids(picture, options) for picture in pictures]
    atlas = converter.atlas.cpu()
    renders = [(render_ids(grid[None], atlas)[0] / atlas.max()).clamp(0, 1) for grid in ids]
    best = int(scorer(prompt, renders).argmax())
    return pictures[best], ids_to_text(ids[best], converter.chars)
