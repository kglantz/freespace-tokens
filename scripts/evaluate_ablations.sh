#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 4 ]; then
  echo 'Usage: bash scripts/evaluate_ablations.sh INDEX TEST_QA RUNS_ROOT PREDICTIONS_DIR [extra inference flags]'
  exit 2
fi
fst_index=$1
fst_test=$2
fst_runs=$3
fst_predictions=$4
shift 4
for fst_arg in "$@"; do
  case "${fst_arg%%=*}" in
    --checkpoint|--index|--qa|--output|--region-id|--question)
      echo "The evaluation runner owns ${fst_arg%%=*}. Use the individual inference CLI for overrides." >&2
      exit 2
      ;;
  esac
done
fst_variants=(surface surface_free surface_unknown surface_both)
fst_checkpoints=()
for fst_variant in "${fst_variants[@]}"; do
  fst_checkpoint="$fst_runs/$fst_variant/best"
  if [ ! -f "$fst_checkpoint/adapter.pt" ]; then fst_checkpoint="$fst_runs/$fst_variant/last"; fi
  if [ ! -f "$fst_checkpoint/adapter.pt" ] || [ ! -f "$fst_checkpoint/model_config.json" ]; then
    echo "Missing checkpoint for $fst_variant: $fst_checkpoint" >&2
    exit 2
  fi
  if [ -e "$fst_predictions/$fst_variant.jsonl" ]; then
    echo "Prediction output already exists: $fst_predictions/$fst_variant.jsonl" >&2
    exit 2
  fi
  fst_checkpoints+=("$fst_checkpoint")
done
for fst_i in "${!fst_variants[@]}"; do
  python -m volumetric_reasoning.prototype.infer \
    --checkpoint "${fst_checkpoints[$fst_i]}" --index "$fst_index" --qa "$fst_test" \
    --output "$fst_predictions/${fst_variants[$fst_i]}.jsonl" "$@"
done
python -m volumetric_reasoning.prototype.compare \
  --predictions-dir "$fst_predictions" --output "$fst_predictions/comparison.json"
