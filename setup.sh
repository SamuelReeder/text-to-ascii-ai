#!/usr/bin/env bash
# Creates the `ascii` conda env (Python 3.12, PyTorch 2.13 + CUDA 13) used by this repo.
# RTX 50-series (Blackwell) GPUs need CUDA >= 12.8 builds of PyTorch; cu130 works for 30/40/50 series.
set -euo pipefail
CONDA="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
"$CONDA" create -y -n ascii python=3.12
ENV_PY="$("$CONDA" run -n ascii python -c 'import sys; print(sys.executable)')"
"$ENV_PY" -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
"$ENV_PY" -m pip install -r "$(dirname "$0")/requirements.txt"
echo "done: conda activate ascii && python ascii.py --help"
