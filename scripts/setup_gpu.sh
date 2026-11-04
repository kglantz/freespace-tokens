#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Run on the Linux GPU machine. Override the wheel index for its driver if needed.
fst_python=${FST_PYTHON:-python3}
"$fst_python" -c 'import sys
if sys.version_info < (3, 10):
    sys.exit("Python 3.10 or newer is required. Python 3.11 is recommended.")'
"$fst_python" -m venv .venv-gpu
.venv-gpu/bin/python -m pip install --upgrade pip setuptools wheel
.venv-gpu/bin/python -m pip install torch==2.8.0 --index-url "${FST_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
.venv-gpu/bin/python -m pip install -r requirements-gpu.txt
.venv-gpu/bin/python -m pip install -e .
echo 'Installed. Activate with: source .venv-gpu/bin/activate'
