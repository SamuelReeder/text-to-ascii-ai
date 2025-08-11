import argparse
import json
import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm
import torch
from torchvision import datasets, transforms


def run_ascii_converter(
    bin_name: str,
    img_path: Path,
    width: int,
    ascii_txt_dir: Path,
    extra_args: Optional[list] = None,
) -> Tuple[bool, str, str]:
    cmd = [bin_name, str(img_path), "-W", str(width), "--save-txt", str(ascii_txt_dir)]
    if extra_args:
        cmd.extend(extra_args)
    try:
        res = subprocess.run(cmd, check=False, capture_output=True, text=True)
        return (res.returncode == 0), res.stdout, res.stderr
    except Exception as e:
        return False, "", str(e)


def save_manifest_row(
    fh,
    *,
    uid: str,
    split: str,
    label: int,
    orig_image: Path,
    ascii_text: Path,
    width_chars: int,
    ascii_rows: int,
):
    row = {
        "id": uid,
        "split": split,
        "label": int(label),
        "image_path": str(orig_image.as_posix()),
        "ascii_txt_path": str(ascii_text.as_posix()),
        "ascii_width_chars": width_chars,
        "ascii_height_rows": ascii_rows,
    }
    fh.write(json.dumps(row) + "\n")


def prepare_dirs(root: Path):
    for p in [
        root / "train" / "images",
        root / "train" / "ascii_txt",
        root / "test" / "images",
        root / "test" / "ascii_txt",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def load_mnist(data_dir: Path, split: str):
    transform = transforms.ToTensor()
    is_train = (split == "train")
    ds = datasets.MNIST(
        root=str(data_dir),
        train=is_train,
        download=True,
        transform=transform
    )
    return ds


def tensor_to_png(tensor, out_path: Path):
    """
    MNIST tensors are [1, 28, 28] in [0,1]. Save as 8-bit PNG.
    """
    from PIL import Image
    arr = (tensor.squeeze().numpy() * 255).astype("uint8")
    img = Image.fromarray(arr, mode="L")  # grayscale
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Build an MNIST → ASCII dataset using ascii-image-converter (saves TXT files)."
    )
    parser.add_argument("--out", type=Path, required=True,
                        help="Output dir.")
    parser.add_argument("--width", type=int, default=80,
                        help="ASCII width (chars). Preserves aspect ratio.")
    parser.add_argument("--limit-train", type=int, default=None,
                        help="Cap on train samples.")
    parser.add_argument("--limit-test", type=int, default=None,
                        help="Cap on test samples.")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"),
                        help="Where to download/cache MNIST.")
    parser.add_argument("--overwrite", action="store_true",
                        help="If set, remove existing OUT directory first.")
    parser.add_argument("--extra-aic-args", type=str, default="",
                        help="Extra args to pass to ascii-image-converter (as one string).")
    parser.add_argument("--aic-bin", type=str, default=os.environ.get("AIC_BIN", "ascii-image-converter"),
                        help="Binary name/path for ascii-image-converter.")
    args = parser.parse_args()

    out_root: Path = args.out
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    prepare_dirs(out_root)

    manifest_train = (out_root / "train_manifest.jsonl").open("w", encoding="utf-8")
    manifest_test = (out_root / "test_manifest.jsonl").open("w", encoding="utf-8")
    errors_log = (out_root / "errors.log").open("w", encoding="utf-8")

    extra_args = [a for a in args.extra_aic_args.split() if a]

    for split, limit, manifest in [
        ("train", args.limit_train, manifest_train),
        ("test", args.limit_test, manifest_test),
    ]:
        ds = load_mnist(args.data_dir, split)
        n = len(ds) if limit is None else min(limit, len(ds))
        pbar = tqdm(range(n), desc=f"Processing {split}", unit="img")

        ascii_txt_dir = out_root / split / "ascii_txt"
        images_dir = out_root / split / "images"

        for idx in pbar:
            img_t, label = ds[idx]  # tensor [1,28,28], label int
            uid = f"{split}-{idx:06d}"

            img_path = images_dir / f"{uid}.png"
            # The CLI writes <image-stem>-ascii-art.txt into ascii_txt_dir
            ascii_txt_path = ascii_txt_dir / f"{uid}-ascii-art.txt"

            try:
                tensor_to_png(img_t, img_path)
            except Exception as e:
                errors_log.write(f"[{uid}] save_png failed: {e}\n")
                continue

            ok, _stdout, stderr_txt = run_ascii_converter(
                args.aic_bin, img_path, args.width, ascii_txt_dir, extra_args
            )
            if not ok:
                errors_log.write(f"[{uid}] aic failed: {stderr_txt}\n")
                continue

            if not ascii_txt_path.exists():
                errors_log.write(f"[{uid}] expected TXT not found at {ascii_txt_path}\n")
                continue

            try:
                ascii_text = ascii_txt_path.read_text(encoding="utf-8")
            except Exception as e:
                errors_log.write(f"[{uid}] read_txt failed: {e}\n")
                continue

            ascii_rows = ascii_text.strip("\n").count("\n") + 1 if ascii_text.strip("\n") else 0

            save_manifest_row(
                manifest,
                uid=uid,
                split=split,
                label=int(label),
                orig_image=img_path,
                ascii_text=ascii_txt_path,
                width_chars=args.width,
                ascii_rows=ascii_rows,
            )

    manifest_train.close()
    manifest_test.close()
    errors_log.close()

    print("\nDone")
    print(f"- Train manifest: {out_root/'train_manifest.jsonl'}")
    print(f"- Test manifest:  {out_root/'test_manifest.jsonl'}")
    print(f"- Errors:         {out_root/'errors.log'}")


if __name__ == "__main__":
    main()