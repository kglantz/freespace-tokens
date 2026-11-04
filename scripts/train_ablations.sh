#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 4 ]; then
  echo 'Usage: bash scripts/train_ablations.sh INDEX TRAIN_QA VAL_QA OUTPUT_ROOT [extra train flags]'
  echo 'Set FST_GPUS to the number of GPUs. Pass an empty VAL_QA for training-only debug data.'
  exit 2
fi
fst_index=$1
fst_train=$2
fst_val=$3
fst_output=$4
shift 4
fst_gpus=${FST_GPUS:-1}
if ! [[ "$fst_gpus" =~ ^[1-9][0-9]*$ ]]; then
  echo 'FST_GPUS must be a positive integer.' >&2
  exit 2
fi
for fst_arg in "$@"; do
  case "${fst_arg%%=*}" in
    --variant|--output|--index|--train-qa|--val-qa|--resume|--preflight-only)
      echo "The ablation runner owns ${fst_arg%%=*}. Use the individual training CLI for overrides or resume." >&2
      exit 2
      ;;
  esac
done
fst_variants=(surface surface_free surface_unknown surface_both)
fst_inputs=(--index "$fst_index" --train-qa "$fst_train")
if [ -n "$fst_val" ]; then fst_inputs+=(--val-qa "$fst_val"); fi
shopt -s dotglob nullglob
for fst_variant in "${fst_variants[@]}"; do
  fst_run="$fst_output/$fst_variant"
  if [ -e "$fst_run" ] && [ ! -d "$fst_run" ]; then
    echo "Training output must be a directory: $fst_run" >&2
    exit 2
  fi
  for fst_artifact in "$fst_run"/*; do
    if [ "${fst_artifact##*/}" != '.training.lock' ]; then
      echo "Training output must be empty: $fst_run" >&2
      exit 2
    fi
  done
done
python -m volumetric_reasoning.prototype.train "${fst_inputs[@]}" \
  --variant surface_both --output "$fst_output/surface_both" "$@" --preflight-only
fst_revision=''
for fst_variant in "${fst_variants[@]}"; do
  fst_run="$fst_output/$fst_variant"
  fst_args=("${fst_inputs[@]}" --variant "$fst_variant" --output "$fst_run" "$@")
  if [ -n "$fst_revision" ]; then fst_args+=(--model-revision "$fst_revision"); fi
  torchrun --standalone --nproc_per_node="$fst_gpus" -m volumetric_reasoning.prototype.train "${fst_args[@]}"
  if [ -z "$fst_revision" ]; then
    fst_revision=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["model"]["base_revision"])' "$fst_run/run.json")
  fi
done
