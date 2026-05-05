#!/usr/bin/env bash
# Set up .venv for Atari training / data collection.
#
# The default torch wheel on PyPI is +cu130 (CUDA 13.0) which won't run on
# CUDA 12.8 drivers (most current setups) — the runtime fails with
# "NVIDIA driver too old". We pin +cu128 from PyTorch's own index instead.
# This script is idempotent: rerun to repair an existing .venv.
#
# Run from the repo root:
#     bash scripts/setup_env.sh

set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv..."
    pip install --quiet uv
fi

if [ ! -d ".venv" ]; then
    echo "[setup] creating .venv with python 3.10..."
    uv venv --python=3.10 .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# Install everything in one resolve so uv sees torch's local-version pin.
# The cu128 index is primary; PyPI is the fallback for non-torch packages.
echo "[setup] installing pinned torch (+cu128) and Atari deps..."
uv pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple \
    "torch==2.11.0+cu128" \
    "torchvision" \
    "stable-worldmodel[format]" \
    "gymnasium[atari]" \
    "ale-py" \
    "autorom[accept-rom-license]" \
    "opencv-python"

echo
echo "[setup] verifying torch + CUDA..."
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print(f"torch={torch.__version__}  cuda_available={ok}")
if ok:
    print(f"device: {torch.cuda.get_device_name(0)}")
else:
    raise SystemExit("CUDA not available — check NVIDIA driver + torch local-version pin")
PY

echo
echo "[setup] done. activate with:  source .venv/bin/activate"
