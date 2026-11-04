#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo 'Usage: bash scripts/run_current_roi.sh SCANNETPP_DATA_ROOT [RUN_DIRECTORY]'
  echo 'Activate .venv-gpu first. This trains on three geometry-only debug questions, not ViCA evaluation.'
  exit 2
fi
fst_data=$1
fst_run=${2:-runs/current_roi}
fst_index=artifacts/current_roi/index.json
if [ ! -f "$fst_index" ]; then
  python -m volumetric_reasoning.prototype.prepare register \
    --index "$fst_index" --scene 1ada7a0617 --scope roi \
    --volume outputs/stage_one_validation/uniform_32/volume.npz \
    --regions outputs/cuboid_validation/uniform_32/token_geometry.npz
fi
python -m volumetric_reasoning.prototype.train \
  --index "$fst_index" --train-qa examples/current_roi_train.jsonl \
  --output "$fst_run" --variant surface_both --preflight-only
python -m volumetric_reasoning.features.extract \
  --index "$fst_index" --data-root "$fst_data" --skip-existing
python -m volumetric_reasoning.prototype.train \
  --index "$fst_index" --train-qa examples/current_roi_train.jsonl \
  --output "$fst_run" --variant surface_both --epochs 10 --max-steps 30 \
  --gradient-accumulation 1 --save-every 10
python -m volumetric_reasoning.prototype.infer \
  --checkpoint "$fst_run/last" --index "$fst_index" --region-id 1ada7a0617 \
  --question 'What is the z-axis extent of the entire mapped region in metres? Answer with a number.'
