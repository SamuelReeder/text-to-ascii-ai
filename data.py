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
from PIL import Image

IMAGES_DIR="orig_imgs"
ASCII_DIR="ascii_txt"

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

    res = subprocess.run(cmd, check=False, capture_output=True, text=True)

    if not res.returncode == 0:
        raise Exception(f"aic failed: {res.stderr}\n")


def save_manifest_row(
    fh,
    *,
    uid: str,
    split: str,
    label: int,
    orig_image: Path,
    ascii_text: Path,
    ascii_width: int,
    ascii_height: int,
):
    row = {
        "id": uid,
        "split": split,
        "label": int(label),
        "orig_img_path": str(orig_image.as_posix()),
        "ascii_txt_path": str(ascii_text.as_posix()),
        "ascii_width": ascii_width,
        "ascii_height": ascii_height,
    }
    fh.write(json.dumps(row) + "\n")


def prepare_dirs(root: Path):
    for p in [
        root / "train" / IMAGES_DIR,
        root / "train" / ASCII_DIR,
        root / "test" / IMAGES_DIR,
        root / "test" / ASCII_DIR,
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

    extra_args = [a for a in args.extra_aic_args.split() if a]

    for split, limit, manifest in [
        ("train", args.limit_train, manifest_train),
        ("test", args.limit_test, manifest_test),
    ]:
        ds = load_mnist(args.data_dir, split)
        n = len(ds) if limit is None else min(limit, len(ds))
        pbar = tqdm(range(n), desc=f"Processing {split}", unit="img")

        ascii_txt_dir = out_root / split / ASCII_DIR
        images_dir = out_root / split / IMAGES_DIR

        for idx in pbar:
            img_t, label = ds[idx]  # tensor [1,28,28], label int
            uid = f"{split}-{idx:06d}"

            img_path = images_dir / f"{uid}.png"
            ascii_txt_path = ascii_txt_dir / f"{uid}-ascii-art.txt"

            # Saves original tensors as images
            try:
                tensor_to_png(img_t, img_path)
            except Exception as e:
                print(f"[{uid}] save_png failed: {e}\n")
                return

            try: 
                run_ascii_converter(
                    args.aic_bin, img_path, args.width, ascii_txt_dir, extra_args
                )
            except Exception as e:
                print(f"[{uid}] run_ascii_converter failed: {e}\n")
                return
            

            if not ascii_txt_path.exists():
                print(f"[{uid}] expected TXT not found at {ascii_txt_path}\n")
                return

            try:
                ascii_text = ascii_txt_path.read_text(encoding="utf-8")
            except Exception as e:
                print(f"[{uid}] read_txt failed: {e}\n")
                return

            ascii_rows = ascii_text.strip("\n").count("\n") + 1 if ascii_text.strip("\n") else 0

            save_manifest_row(
                manifest,
                uid=uid,
                split=split,
                label=int(label),
                orig_image=img_path,
                ascii_text=ascii_txt_path,
                ascii_width=args.width,
                ascii_height=ascii_rows,
            )

    manifest_train.close()
    manifest_test.close()

    print("\nDone")
    print(f"- Train manifest: {out_root/'train_manifest.jsonl'}")
    print(f"- Test manifest:  {out_root/'test_manifest.jsonl'}")

if __name__ == "__main__":
    main()