#!/usr/bin/env bash
# An isolated CPU image converter by default; opt into CUDA and larger profiles.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
device=cpu
profile=image
venv="$ROOT/.venv"
while (( $# )); do
  case "$1" in
    --cpu) device=cpu; shift ;;
    --cuda) device=cu130; shift ;;
    --profile|--venv)
      if (( $# < 2 )); then echo "Missing value for $1" >&2; exit 2; fi
      if [[ "$1" == --profile ]]; then profile="$2"; else venv="$2"; fi
      shift 2 ;;
    -h|--help)
      echo 'Usage: ./setup.sh [--cpu|--cuda] [--profile image|pipeline|text|all|demo] [--venv PATH]'
      echo 'Requires Python 3.12+ and a monospace font (Ubuntu: fonts-dejavu-core).'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$profile" in
  image|pipeline|text|demo) requirements="$ROOT/requirements/$profile.txt" ;;
  all) requirements="$ROOT/requirements.txt" ;;
  *) echo "Unknown profile: $profile" >&2; exit 2 ;;
esac
"${PYTHON:-python3}" -m venv "$venv"
py="$venv/bin/python"
"$py" -m pip install --upgrade pip
torch_packages=(torch==2.13.0)
if [[ "$profile" != image ]]; then torch_packages+=(torchvision==0.28.0); fi
"$py" -m pip install "${torch_packages[@]}" --index-url "https://download.pytorch.org/whl/$device"
"$py" -m pip install -r "$requirements"
echo "Ready: source '$venv/bin/activate'"
