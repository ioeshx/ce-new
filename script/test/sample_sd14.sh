#!/usr/bin/env bash
export HF_ENDPOINT=https://hf-mirror.com

set -euo pipefail
CASU2_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CHECKPOINT:?Set CHECKPOINT to a CASU-2 checkpoint directory.}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
FLAGS=()
case "${ORIGINAL_CACHE:-1}" in
  1) FLAGS+=(--original-cache) ;;
  0) FLAGS+=(--no-original-cache) ;;
  *) echo "ORIGINAL_CACHE must be 0 or 1." >&2; exit 2 ;;
esac
if [[ -n "${ORIGINAL_DIR:-${ORIGINAL_CACHE_DIR:-}}" ]]; then
  FLAGS+=(--original-dir "${ORIGINAL_DIR:-$ORIGINAL_CACHE_DIR}")
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then FLAGS+=(--output-dir "$OUTPUT_DIR"); fi
if [[ -n "${CONTENTS:-}" ]]; then FLAGS+=(--contents "$CONTENTS"); fi
exec "${PYTHON:-python}" "$CASU2_DIR/main.py" sample \
  --checkpoint "$CHECKPOINT" \
  --sample-template-count "${SAMPLE_TEMPLATES:-8}" \
  --num-samples "${NUM_SAMPLES:-1}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --steps "${STEPS:-20}" --seed "${SEED:-0}" \
  "${FLAGS[@]}" "$@"
