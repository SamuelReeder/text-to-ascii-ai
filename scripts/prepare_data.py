"""Extract downloaded HF shards into data/images/{train,eval}/<source>/*.jpg plus manifests.

Eval images are held out per source so the legibility metrics cover every domain
(photos, scenes, faces, art, cartoons, doodles, sketches, logos, screenshots, ...) without
overlapping training data.

Memory stays bounded (a few GB): tar and parquet sources are indexed first and only the
selected images are read, images are written by a small worker pool in batches of 256, and each
finished source is recorded in data/images/parts/<source>.jsonl so an interrupted run resumes.
"""
import io
import json
import random
import sys
import tarfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asciiart.imageio import load_image  # noqa: E402  (torch-free, keeps workers small)

RAW = Path("data/raw")
OUT = Path("data/images")
PARTS = OUT / "parts"
MAX_SIDE = 512
MAX_SIDE_NEW = 1024  # enough for ~128-column grids without upsampling
IMG_EXT = ("jpg", "jpeg", "png", "webp")
WORKERS = 6

CORE = ["imagenet", "coco", "celeba", "ffhq", "wikiart", "pokemon", "cartoon", "quickdraw", "flickr8k", "caltech101"]
# name: (clip-benchmark dataset, shard pattern, n_train, n_eval, eval keeps labels)
WDS = {
    "sun397": ("sun397", "test/*.tar", 9000, 500, True),
    "imagenet_r": ("imagenet-r", "test/*.tar", 8000, 1000, True),
    "sketch": ("imagenet_sketch", "test/*.tar", 5000, 500, True),
    "food101": ("food101", "train/*.tar", 3000, 0, True),
    "cars": ("cars", "train/*.tar", 1800, 100, False),
    "aircraft": ("fgvc_aircraft", "train/*.tar", 1500, 100, False),
    "pets": ("vtab-pets", "train/*.tar", 2400, 0, True),
    "flowers": ("vtab-flowers", "train/*.tar", 900, 100, False),
    "aerial": ("vtab-resisc45", "train/*.tar", 2000, 100, False),
    "satellite": ("vtab-eurosat", "train/*.tar", 1000, 50, False),
    "texture": ("vtab-dtd", "train/*.tar", 1500, 100, False),
    "traffic_sign": ("gtsrb", "train/*.tar", 1200, 50, False),
    "geo_photo": ("country211", "train/*.tar", 4000, 100, False),
    "fer": ("fer2013", "train/*.tar", 1000, 50, False),
    "mnist": ("mnist", "train/*.tar", 800, 50, False),
    "svhn": ("vtab-svhn", "train/*.tar", 800, 50, False),
    "clevr": ("vtab-clevr_count_all", "train/*.tar", 1500, 50, False),
    "text_render": ("renderedsst2", "train/*.tar", 800, 50, False),
    "voc": ("voc2007", "train/*.tar", 2400, 100, False),
    "game3d": ("vtab-dmlab", "train/*.tar", 1000, 50, False),
    "driving": ("vtab-kitti_closest_vehicle_distance", "train/*.tar", 1200, 50, False),
    "microscopy": ("vtab-pcam", "train/*.tar", 500, 30, False),
}
# labeled eval-only splits
WDS_EVAL = {"food101": ("food101", "test/*.tar", 300), "pets": ("vtab-pets", "test/*.tar", 300)}
# name: (parquet glob, image column, n_train, n_eval)
PARQUET = {
    "logo": ("iamkaikai__amazing_logos_v4/data/*.parquet", "image", 4000, 100),
    "icon": ("likaixin__IconStack-48M-Rendered-Train/*/*.parquet", "image", 4000, 100),
    "emoji": ("Norod78__microsoft-fluentui-emoji-512-whitebg/data/*.parquet", "image", 3000, 100),
    "pixelart": ("Chan-Y__pixelart-308k/data/*.parquet", "image", 4000, 100),
    "product": ("ashraq__fashion-product-images-small/data/*.parquet", "image", 4000, 100),
    "chart": ("HuggingFaceM4__ChartQA/data/*.parquet", "image", 2000, 100),
    "diagram": ("shreyanshu09__Block_Diagram/data/*.parquet", "image", 1500, 100),
    "ink_painting": ("mingyy__chinese_landscape_paintings/data/*.parquet", "target", 540, 50),
}
# name: (raw folder, n_train, n_eval)
LOOSE = {
    "manga": ("Chan-Y__Manga-Drawings", 950, 97),
    "game_screenshot": ("taesiri__steam_screenshots_samples_2", 1400, 100),
    "web_screenshot": ("silatus__1k_Website_Screenshots_and_Metadata", 900, 100),
    "anime_lineart": ("ityizNola__Anime-LineArt-Dataset", 1100, 100),
    "lineart": ("ebykAI__LineArtImageNet100", 1100, 100),
    "anime_wallpaper": ("puruchinera__anime_wallpapers", 700, 100),
}
NEW_SOURCES = set(WDS) | set(PARQUET) | set(LOOSE) | {"anime_face", "generated"}
ORDER = CORE + list(WDS) + list(PARQUET) + list(LOOSE) + ["anime_face", "generated"]


def save(args):
    data, path = args
    if path.exists():
        try:  # a file left half-written by an interrupted run is redone
            Image.open(path).verify()
            return True
        except Exception:  # noqa: BLE001
            pass
    try:
        im = Image.open(io.BytesIO(data))
        if max(im.size) > 2 * MAX_SIDE_NEW and getattr(im, "n_frames", 1) == 1:
            im.draft("RGB", (2 * MAX_SIDE_NEW, 2 * MAX_SIDE_NEW))  # JPEG: decode at reduced size
            im.thumbnail((2 * MAX_SIDE_NEW, 2 * MAX_SIDE_NEW), Image.LANCZOS)  # before compositing (memory)
        max_side = MAX_SIDE
        if path.parts[-2] in NEW_SOURCES:  # composite transparency exactly like inference does
            im = load_image(im)
            max_side = MAX_SIDE_NEW
            if im.height > 2 * im.width:  # full-page screenshots etc.: keep the top (first screen)
                im = im.crop((0, 0, im.width, int(1.5 * im.width)))
        else:
            im = im.convert("RGBA").convert("RGB") if im.mode in ("P", "LA", "RGBA") else im.convert("RGB")
        if max(im.size) > max_side:
            s = max_side / max(im.size)
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
        if max(im.size) < 64:  # doodles are 28x28: upsample so resizing to the grid isn't the bottleneck
            im = im.resize((im.width * 8, im.height * 8), Image.BICUBIC)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.jpg")
        im.save(tmp, quality=92)
        tmp.rename(path)
        return True
    except Exception as e:  # noqa: BLE001
        print("skip", path, e)
        return False


class Writer:
    """Queues (bytes, path) jobs, writes them in batches, records finished sources as parts."""

    def __init__(self, pool):
        self.pool, self.jobs, self.recs, self.bad, self.pending = pool, [], [], set(), 0

    def add(self, split, source, i, data, **meta):
        p = OUT / split / source / f"{i:06d}.jpg"
        self.jobs.append((data, p))
        self.recs.append({"path": str(p), "source": source, "split": split, **meta})
        self.pending += len(data)
        if len(self.jobs) >= 256 or self.pending > 128e6:
            self.flush()

    def flush(self):
        ok = list(self.pool.map(save, self.jobs, chunksize=4))
        self.bad |= {str(p) for (_, p), k in zip(self.jobs, ok) if not k}
        self.jobs.clear()
        self.pending = 0

    def done(self, source):
        self.flush()
        recs = [r for r in self.recs if r["source"] == source and r["path"] not in self.bad]
        self.recs = [r for r in self.recs if r["source"] != source]
        write_part(source, recs)


def write_part(source, recs):
    for r in recs:  # image size lets training avoid heavily upsampled inputs at wide grids
        if "w" not in r:
            r["w"], r["h"] = Image.open(r["path"]).size
    PARTS.mkdir(parents=True, exist_ok=True)
    tmp = PARTS / f"{source}.jsonl.tmp"
    with open(tmp, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    tmp.rename(PARTS / f"{source}.jsonl")
    counts = {s: sum(r["split"] == s for r in recs) for s in ("train", "eval")}
    print(source, counts, flush=True)


def has_part(source):
    return (PARTS / f"{source}.jsonl").exists()


def adopt_existing_manifests():
    """Resume support: sources listed in existing manifests (whose files exist) become parts."""
    by_source = {}
    for split in ("train", "eval"):
        f = OUT / f"{split}.jsonl"
        if f.exists():
            for line in open(f):
                r = json.loads(line)
                r["split"] = split
                by_source.setdefault(r["source"], []).append(r)
    for source, recs in by_source.items():
        if not has_part(source) and all(Path(r["path"]).exists() for r in recs):
            write_part(source, recs)


def rows(pattern):
    for f in sorted(RAW.glob(pattern)):
        yield from pq.read_table(f).to_pylist()


def core_sources(w: Writer):
    """Photos, faces, art, cartoons, doodles and the labeled eval sets (one shared RNG, fixed order)."""
    rng = random.Random(0)
    # --- ImageNet (256px) ---
    for i, r in enumerate(rows("benjamin-paine__imagenet-1k-256x256/data/train-*.parquet")):
        w.add("train", "imagenet", i, r["image"]["bytes"], label=r["label"])
    val = list(rows("benjamin-paine__imagenet-1k-256x256/data/validation-*.parquet"))
    for i, r in enumerate(rng.sample(val, 1000)):
        w.add("eval", "imagenet", i, r["image"]["bytes"], label=r["label"])
    del val
    w.done("imagenet")
    # --- COCO scenes ---
    for i, r in enumerate(rows("detection-datasets__coco/data/train-*.parquet")):
        w.add("train", "coco", i, r["image"]["bytes"])
    for i, r in enumerate(rng.sample(list(rows("detection-datasets__coco/data/val-*.parquet")), 500)):
        w.add("eval", "coco", i, r["image"]["bytes"])
    w.done("coco")
    # --- Faces ---
    celeba = list(rows("nielsr__CelebA-faces/data/*.parquet"))
    rng.shuffle(celeba)
    for i, r in enumerate(celeba[:10000]):
        w.add("train", "celeba", i, r["image"]["bytes"])
    for i, r in enumerate(celeba[10000:10300]):
        w.add("eval", "celeba", i, r["image"]["bytes"])
    del celeba
    w.done("celeba")
    ffhq = list(rows("bitmind__ffhq-256/data/*.parquet"))
    for i, r in enumerate(ffhq[:-300]):
        w.add("train", "ffhq", i, r["image"]["bytes"])
    for i, r in enumerate(ffhq[-300:]):
        w.add("eval", "ffhq", i, r["image"]["bytes"])
    del ffhq
    w.done("ffhq")
    # --- Art / cartoons ---
    art = list(rows("huggan__wikiart/data/*.parquet"))
    rng.shuffle(art)
    for i, r in enumerate(art[:-200]):
        w.add("train", "wikiart", i, r["image"]["bytes"])
    for i, r in enumerate(art[-200:]):
        w.add("eval", "wikiart", i, r["image"]["bytes"])
    del art
    w.done("wikiart")
    for name, pat, n_eval in [("pokemon", "reach-vb__pokemon-blip-captions/data/*.parquet", 100),
                              ("cartoon", "Norod78__cartoon-blip-captions/data/*.parquet", 300)]:
        rs = list(rows(pat))
        rng.shuffle(rs)
        for i, r in enumerate(rs[:-n_eval]):
            w.add("train", name, i, r["image"]["bytes"], caption=r["text"])
        for i, r in enumerate(rs[-n_eval:]):
            w.add("eval", name, i, r["image"]["bytes"], caption=r["text"])
        del rs
        w.done(name)
    # --- Doodles ---
    qd = list(rows("Xenova__quickdraw-small/data/valid-*.parquet"))
    rng.shuffle(qd)
    for i, r in enumerate(qd[:8000]):
        w.add("train", "quickdraw", i, r["image"]["bytes"], label=r["label"])
    for i, r in enumerate(qd[8000:8500]):
        w.add("eval", "quickdraw", i, r["image"]["bytes"], label=r["label"])
    del qd
    w.done("quickdraw")
    # --- Flickr8k captions (eval only) ---
    for i, r in enumerate(rows("jxie__flickr8k/data/test-*.parquet")):
        w.add("eval", "flickr8k", i, r["image"]["bytes"], captions=[r[f"caption_{k}"] for k in range(5)])
    w.done("flickr8k")
    # --- Caltech-101 (eval only, labeled) ---
    cal = []
    for tf in sorted((RAW / "clip-benchmark__wds_vtab-caltech101/test").glob("*.tar")):
        with tarfile.open(tf) as t:
            members = {m.name: m for m in t.getmembers()}
            for name in members:
                if name.endswith(".webp"):
                    key = name[:-5]
                    cal.append((t.extractfile(members[name]).read(),
                                int(t.extractfile(members[key + ".cls"]).read().decode().strip())))
    for i, (b, lab) in enumerate(rng.sample(cal, 1000)):
        w.add("eval", "caltech101", i, b, label=lab)
    del cal
    w.done("caltech101")


def wds_index(name, pattern):
    """[(tar path, image member name, class id)] from clip-benchmark shards, without reading images."""
    idx = []
    for tf in sorted(RAW.glob(f"clip-benchmark__wds_{name}/{pattern}")):
        with tarfile.open(tf) as t:
            cls, imgs = {}, []
            for m in t.getmembers():
                key, ext = m.name.rsplit(".", 1)
                if ext == "cls":
                    cls[key] = int(t.extractfile(m).read())
                elif ext in IMG_EXT:
                    imgs.append((key, m.name))
        idx += [(str(tf), n, cls[k]) for k, n in imgs]
    return idx


def wds_add(w, source, picks, labeled):
    """picks: [(split, i, (tar, member, cls))]; reads each tar once."""
    by_tar = {}
    for split, i, (tf, name, lab) in picks:
        by_tar.setdefault(tf, []).append((split, i, name, lab))
    for tf, items in by_tar.items():
        with tarfile.open(tf) as t:
            members = {m.name: m for m in t.getmembers()}
            for split, i, name, lab in items:
                meta = {"label": lab} if (labeled and split == "eval") else {}
                w.add(split, source, i, t.extractfile(members[name]).read(), **meta)


def parquet_select(pattern, col, n, rng):
    """n random rows of one column across the files, streamed in small batches."""
    files = sorted(RAW.glob(pattern))
    sizes = [pq.ParquetFile(f).metadata.num_rows for f in files]
    want = set(rng.sample(range(sum(sizes)), min(n, sum(sizes))))
    out, base = {}, 0
    for f, size in zip(files, sizes):
        if any(base <= k < base + size for k in want):
            j = base
            for batch in pq.ParquetFile(f).iter_batches(batch_size=64, columns=[col]):
                for v in batch.column(0).to_pylist():
                    if j in want:
                        out[j] = v["bytes"] if isinstance(v, dict) else v
                    j += 1
        base += size
    return [out[k] for k in sorted(out)]


def breadth_sources(w: Writer):
    """Scenes, renditions, sketches, fine-grained objects, aerial, textures, digits, logos, icons, ..."""
    for name, (ds, pattern, n_train, n_eval, labeled) in WDS.items():
        if has_part(name):
            continue
        rng = random.Random(name)
        idx = wds_index(ds, pattern)
        rng.shuffle(idx)
        picks = [("eval", i, it) for i, it in enumerate(idx[:n_eval])]
        picks += [("train", i, it) for i, it in enumerate(idx[n_eval:n_eval + n_train])]
        if name in WDS_EVAL:
            ds_e, pat_e, n_e = WDS_EVAL[name]
            picks += [("eval", i, it) for i, it in enumerate(rng.sample(wds_index(ds_e, pat_e), n_e))]
        wds_add(w, name, picks, labeled)
        w.done(name)
    for name, (pattern, col, n_train, n_eval) in PARQUET.items():
        if has_part(name):
            continue
        rng = random.Random(name)
        rs = parquet_select(pattern, col, n_train + n_eval, rng)
        rng.shuffle(rs)
        for i, b in enumerate(rs[:n_eval]):
            w.add("eval", name, i, b)
        for i, b in enumerate(rs[n_eval:]):
            w.add("train", name, i, b)
        del rs
        w.done(name)
    for name, (folder, n_train, n_eval) in LOOSE.items():
        if has_part(name):
            continue
        rng = random.Random(name)
        files = sorted(f for f in (RAW / folder).rglob("*")
                       if f.suffix.lower().lstrip(".") in IMG_EXT and ".cache" not in f.parts)
        rng.shuffle(files)
        for i, f in enumerate(files[:n_eval]):
            w.add("eval", name, i, f.read_bytes())
        for i, f in enumerate(files[n_eval:n_eval + n_train]):
            w.add("train", name, i, f.read_bytes())
        w.done(name)
    if not has_part("anime_face"):
        rng = random.Random("anime_face")
        with zipfile.ZipFile(RAW / "huggan__anime-faces/data.zip") as z:
            names = rng.sample(sorted(n for n in z.namelist() if n.endswith(".png")), 3100)
            for i, n in enumerate(names[:100]):
                w.add("eval", "anime_face", i, z.read(n))
            for i, n in enumerate(names[100:]):
                w.add("train", "anime_face", i, z.read(n))
        w.done("anime_face")
    # pictures from the text->image model (scripts/generate_images.py); rebuilt every run
    gen = sorted(Path("data/generated").glob("*.jpg")) if Path("data/generated").exists() else []
    if gen:
        for i, f in enumerate(gen):
            meta = f.with_suffix(".json")
            cap = json.loads(meta.read_text())["prompt"] if meta.exists() else ""
            w.add("eval" if i % 50 == 0 else "train", "generated", i, f.read_bytes(), caption=cap)
        w.done("generated")


def main():
    # the pool is created before any data is loaded, so forked workers stay small
    with ProcessPoolExecutor(WORKERS) as pool:
        w = Writer(pool)
        adopt_existing_manifests()
        if not all(has_part(s) for s in CORE):
            core_sources(w)
        breadth_sources(w)
    manifests = {"train": [], "eval": []}
    for source in ORDER:
        if has_part(source):
            for line in open(PARTS / f"{source}.jsonl"):
                r = json.loads(line)
                manifests[r.pop("split")].append(r)
    for split, m in manifests.items():
        with open(OUT / f"{split}.jsonl", "w") as f:
            for r in m:
                f.write(json.dumps(r) + "\n")
        counts = {}
        for r in m:
            counts[r["source"]] = counts.get(r["source"], 0) + 1
        print(split, len(m), counts)


if __name__ == "__main__":
    main()
