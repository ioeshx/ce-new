#!/usr/bin/env bash
export HF_ENDPOINT=https://hf-mirror.com

set -euo pipefail

CASU2_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${1:?Usage: run_sd14.sh instance|style [extra main.py erase arguments]}"
shift

case "$TASK" in
  instance) TARGETS="${TARGETS:-Snoopy}"; ANCHORS="${ANCHORS-dog}" ;;
  style) TARGETS="${TARGETS:-Van Gogh}"; ANCHORS="${ANCHORS-realistic}" ;;
  *) echo "Task must be instance or style." >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
FLAGS=()
case "${CA:-1}" in
  1) FLAGS+=(--contextual-anchor) ;;
  0) FLAGS+=(--no-contextual-anchor) ;;
  *) echo "CA must be 0 or 1." >&2; exit 2 ;;
esac
case "${SU:-0}" in
  1)
    FLAGS+=(--selective-update
      --selection-order "${SELECTION_ORDER:-proxy}"
      --selection-judge "${SELECTION_JUDGE:-manual}")
    ;;
  0) FLAGS+=(--no-selective-update) ;;
  *) echo "SU must be 0 or 1." >&2; exit 2 ;;
esac
if [[ -n "${OUTPUT_DIR:-}" ]]; then FLAGS+=(--output-dir "$OUTPUT_DIR"); fi

exec "${PYTHON:-python}" "$CASU2_DIR/main.py" erase \
  --task "$TASK" \
  --sd-ckpt "${SD_CKPT:-CompVis/stable-diffusion-v1-4}" \
  --target-concepts "$TARGETS" --anchor-concepts "$ANCHORS" \
  --params "${PARAMS:-V}" \
  --context-template-count "${CONTEXT_TEMPLATES:-30}" \
  --retain-count "${RETAIN_COUNT:-128}" \
  --beta "${BETA:-1.0}" \
  --contents "${CONTENTS:-$TARGETS}" \
  --sample-template-count "${SAMPLE_TEMPLATES:-8}" \
  --num-samples "${NUM_SAMPLES:-1}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --steps "${STEPS:-20}" --guidance-scale "${GUIDANCE_SCALE:-7.5}" \
  --seed "${SEED:-0}" \
  "${FLAGS[@]}" "$@"
