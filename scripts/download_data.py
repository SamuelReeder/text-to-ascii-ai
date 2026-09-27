"""Download the dataset shards used for training and evaluation.

Everything comes from public Hugging Face datasets (~30 GB). Afterwards run
`python scripts/prepare_data.py` to extract data/images/{train,eval} + manifests.

JOBS: photos, faces, art, cartoons, doodles (+ the labeled eval sets).
BREADTH: scenes, sketches/renditions, fine-grained objects, aerial/satellite, textures, digits,
signs, 3D renders, games, logos, icons, emoji, pixel art, anime, manga, line art, products,
charts, diagrams, screenshots, paintings, so the model sees most kinds of images people convert.
"""
import random
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi, hf_hub_download

JOBS = [
    ("benjamin-paine/imagenet-1k-256x256", "data/train-00000-of-00040.parquet"),
    ("benjamin-paine/imagenet-1k-256x256", "data/train-00001-of-00040.parquet"),
    ("benjamin-paine/imagenet-1k-256x256", "data/validation-00000-of-00002.parquet"),
    ("detection-datasets/coco", "data/train-00000-of-00040-67e35002d152155c.parquet"),
    ("detection-datasets/coco", "data/train-00001-of-00040-2c2b33b9504aa843.parquet"),
    ("detection-datasets/coco", "data/val-00000-of-00002-c4f2e391ee4aba11.parquet"),
    ("nielsr/CelebA-faces", "data/train-00000-of-00003.parquet"),
    ("bitmind/ffhq-256", "data/train-00000-of-00016.parquet"),
    ("huggan/wikiart", "data/train-00000-of-00072.parquet"),
    ("huggan/wikiart", "data/train-00030-of-00072.parquet"),
    ("reach-vb/pokemon-blip-captions", "data/train-00000-of-00001-566cc9b19d7203f8.parquet"),
    ("Norod78/cartoon-blip-captions", "data/train-00000-of-00001-dfb0d9df7ebab67e.parquet"),
    ("Xenova/quickdraw-small", "data/valid-00000-of-00001-e839906e4a48ea50.parquet"),
    ("clip-benchmark/wds_vtab-caltech101", "test/0.tar"),
    ("clip-benchmark/wds_vtab-caltech101", "test/1.tar"),
    ("clip-benchmark/wds_vtab-caltech101", "classnames.txt"),
    ("clip-benchmark/wds_vtab-caltech101", "zeroshot_classification_templates.txt"),
    ("jxie/flickr8k", "data/test-00000-of-00001-42a2661d12c73e48.parquet"),
]

CB = "clip-benchmark/wds_"
# (clip-benchmark shards are sorted by class, so shards are spread over the class range)
BREADTH = [
    *[(CB + "sun397", f"test/{i}.tar") for i in (0, 1, 2, 6, 12, 18, 24, 30, 36)], (CB + "sun397", "classnames.txt"),
    *[(CB + "imagenet-r", f"test/{i}.tar") for i in range(4)], (CB + "imagenet-r", "classnames.txt"),
    *[(CB + "imagenet_sketch", f"test/{i}.tar") for i in (0, 2, 4, 6, 8)],
    *[(CB + "food101", f"train/{i}.tar") for i in (0, 2, 4, 6)], (CB + "food101", "test/0.tar"), (CB + "food101", "classnames.txt"),
    (CB + "cars", "train/0.tar"), *[(CB + "fgvc_aircraft", f"train/{i}.tar") for i in (0, 2, 4)],
    (CB + "vtab-pets", "train/0.tar"), (CB + "vtab-pets", "test/0.tar"), (CB + "vtab-pets", "classnames.txt"),
    (CB + "vtab-flowers", "train/0.tar"), (CB + "vtab-resisc45", "train/0.tar"), (CB + "vtab-eurosat", "train/0.tar"),
    (CB + "vtab-dtd", "train/0.tar"), *[(CB + "gtsrb", f"train/{i}.tar") for i in (0, 2, 4)],
    *[(CB + "country211", f"train/{i}.tar") for i in (0, 2, 4)],
    *[(CB + "fer2013", f"train/{i}.tar") for i in (0, 2)], (CB + "mnist", "train/0.tar"), (CB + "vtab-svhn", "train/0.tar"),
    (CB + "vtab-clevr_count_all", "train/0.tar"), (CB + "renderedsst2", "train/0.tar"),
    *[(CB + "voc2007", f"train/{i}.tar") for i in range(4)], (CB + "vtab-dmlab", "train/0.tar"),
    (CB + "vtab-kitti_closest_vehicle_distance", "train/0.tar"), (CB + "vtab-pcam", "train/0.tar"),
    ("iamkaikai/amazing_logos_v4", "data/train-00000-of-00014-8fa0be170a1cb1f2.parquet"),
    ("likaixin/IconStack-48M-Rendered-Train", "IconStack-F/iconstack-f_0-100000.parquet"),
    ("Norod78/microsoft-fluentui-emoji-512-whitebg", "data/train-00000-of-00001-c0f95ace4411d3e5.parquet"),
    ("Chan-Y/pixelart-308k", "data/train-00000-of-00041.parquet"),
    ("huggan/anime-faces", "data.zip"),
    ("ashraq/fashion-product-images-small", "data/train-00000-of-00002-6cff4c59f91661c3.parquet"),
    ("HuggingFaceM4/ChartQA", "data/train-00000-of-00003-49492f364babfa44.parquet"),
    ("shreyanshu09/Block_Diagram", "data/train-00001-of-00011-8554169812c55911.parquet"),
    ("mingyy/chinese_landscape_paintings", "data/train-00000-of-00089-97ee939f26da621b.parquet"),
]
# datasets stored as loose image files: a fixed random subset of each
LOOSE = [("Chan-Y/Manga-Drawings", "images/", 1047), ("taesiri/steam_screenshots_samples_2", "", 1500),
         ("silatus/1k_Website_Screenshots_and_Metadata", "", 1000), ("ityizNola/Anime-LineArt-Dataset", "", 1200),
         ("ebykAI/LineArtImageNet100", "", 1200), ("puruchinera/anime_wallpapers", "", 800)]


def get(job):
    repo, fn = job
    for attempt in range(5):
        try:
            path = hf_hub_download(repo, fn, repo_type="dataset", local_dir=f"data/raw/{repo.replace('/', '__')}")
            return path
        except Exception as e:  # noqa: BLE001  (rate limits / transient network errors)
            print("retry", repo, fn, e, flush=True)
            import time
            time.sleep(10 * (attempt + 1))
    print("FAILED", repo, fn, flush=True)


def loose_jobs():
    api = HfApi()
    for repo, prefix, n in LOOSE:
        files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset")
                       if f.startswith(prefix) and f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
        yield from ((repo, f) for f in random.Random(0).sample(files, min(n, len(files))))


if __name__ == "__main__":
    with ThreadPoolExecutor(6) as ex:
        for path in ex.map(get, JOBS + BREADTH):
            print("ok", path, flush=True)
    with ThreadPoolExecutor(8) as ex:
        print("loose files:", sum(p is not None for p in ex.map(get, list(loose_jobs()))), flush=True)
