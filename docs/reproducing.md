# Reproducing AsciiNet

[Back to AsciiNet](../README.md)

Use a 16 GB NVIDIA GPU and `./setup.sh --cuda --profile all`. Activate `.venv`
or set `PYTHON` to an existing environment's interpreter. Stages run sequentially:

```bash
scripts/reproduce.sh prepare
scripts/reproduce.sh phase1
scripts/reproduce.sh phase2
python ascii.py photo.jpg --engine net --checkpoint checkpoints/reproduction/asciinet.pt
```

Preparation downloads datasets and generates 6,000 SANA-Sprint images, prepares
manifests, then caches BiRefNet mattes. It requires substantial disk space (the
original local data directory is about 53 GB). Check available storage first.
Each training stage can continue with `--resume`. The script refuses accidental
replacement of an existing training state; `ASCIINET_TRAIN_DIR` chooses a new
output directory. Existing release checkpoints are not overwritten.

The historical phase-1 checkpoint records 30,000 steps, learning rate 0.0008,
1,000 warm-up steps, 40,000 cells per batch, weight decay 0.05, six workers and a
10.5 GiB allocation cap. Phase 2 initialized from its model/EMA state and reset
the optimizer: 20,000 steps, learning rate 0.0004, 500 warm-up steps, subject BCE
weight 1.0 and Dice weight 0.5. The release is phase 2's EMA, not phase 1.
[Recorded settings](benchmarks/training.json) preserve checkpoint metadata.

**Reproducibility limit:** the historical training run did not record an RNG seed
or pin all dataset revisions. These commands reconstruct its procedure, not its
exact weights or scores. Use the pinned released weights to reproduce inference.
Future training comparisons should record seeds, dataset manifests, code revision,
and environment alongside results. Do not claim a bit-for-bit training rerun.

## Evaluation

The original speed test uses 40 evaluation images selected at stride 250. The
[archived timings](benchmarks/gpu-speed.json) are transcribed from its log.
`python scripts/bench_speed.py` measures the default checkpoint; when comparing
retrained weights, explicitly select them in your experiment environment.

The VLM judge needs a separately running llama.cpp server for
Qwen/Qwen3-VL-8B-Instruct-GGUF (Q4_K_M + mmproj), port 8089:

```bash
python scripts/vlm_judge.py --methods net:checkpoints/reproduction/asciinet.pt --cols 24 32 80 --n 60 --sources all --stage render
python scripts/vlm_judge.py --methods net:checkpoints/reproduction/asciinet.pt --cols 24 32 80 --n 60 --sources all --stage ask
python scripts/t2i_bench.py gen
python scripts/t2i_bench.py convert-net
python scripts/t2i_bench.py judge
python scripts/eval_methods.py --methods aic match pipe net --cols 80
python scripts/make_examples.py
python -m pytest -m integration
```

The text benchmark and example scripts use the default checkpoint. Its external
`ascii-image-converter` baseline is optional and installed separately. See
[research notes](research.md) for scoring details and limitations.

On WSL, an optional memory cap can prevent one heavy job from exhausting host
RAM: `systemd-run --user --collect --wait -p MemoryMax=8G -p MemorySwapMax=1G
--working-directory="$PWD" "$PWD/scripts/reproduce.sh" phase1`. Run only one
training, matting, image-generation, or VLM job at a time on a 16 GB GPU.
