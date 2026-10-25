#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${1:-/workspace/.venv-underwater-ml}"
if [ ! -x "$venv_dir/bin/python" ]; then
  python -m venv --system-site-packages "$venv_dir"
fi
"$venv_dir/bin/python" -m pip install --cache-dir /workspace/.cache-underwater-pip -r "$project_root/requirements-advanced.txt"
"$venv_dir/bin/python" -m pip install --cache-dir /workspace/.cache-underwater-pip -r "$project_root/requirements-torch-cpu.txt"
"$venv_dir/bin/python" -c 'import torch, xgboost; print("torch", torch.__version__, "xgboost", xgboost.__version__, "CUDA available", torch.cuda.is_available())'
