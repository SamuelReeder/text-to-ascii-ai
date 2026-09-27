#!/usr/bin/env bash
# Run explicitly selected stages; never overwrite the original released checkpoint.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
py="${PYTHON:-$ROOT/.venv/bin/python}"
out="${ASCIINET_TRAIN_DIR:-checkpoints/reproduction}"
stage="${1:-help}"
resume=()
if [[ "${2:-}" == --resume ]]; then resume=(--resume); fi
case "$stage" in
  prepare)
    "$py" scripts/download_data.py
    "$py" scripts/generate_images.py --n 6000 --model sana-sprint
    "$py" scripts/prepare_data.py
    "$py" scripts/cache_mattes.py ;;
  phase1)
    mkdir -p "$out"
    if [[ -f "$out/phase1_train.pt" && ${#resume[@]} == 0 ]]; then
      echo 'Phase 1 already exists; choose another ASCIINET_TRAIN_DIR or pass --resume.' >&2; exit 1
    fi
    "$py" scripts/train.py --steps 30000 --warmup 1000 --lr 8e-4 --matte-w 0.3 --dice-w 0       --out "$out/phase1_train.pt" --export "$out/phase1.pt" "${resume[@]}" ;;
  phase2)
    [[ -f "$out/phase1_train.pt" ]] || { echo 'Run phase1 first.' >&2; exit 1; }
    if [[ -f "$out/phase2_train.pt" && ${#resume[@]} == 0 ]]; then
      echo 'Phase 2 already exists; choose another ASCIINET_TRAIN_DIR or pass --resume.' >&2; exit 1
    fi
    "$py" scripts/train.py --init "$out/phase1_train.pt" --steps 20000 --warmup 500 --lr 4e-4       --matte-w 1 --dice-w 0.5 --out "$out/phase2_train.pt" --export "$out/asciinet.pt" "${resume[@]}" ;;
  *) echo 'Usage: scripts/reproduce.sh prepare|phase1|phase2 [--resume]'; exit 0 ;;
esac
