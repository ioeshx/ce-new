#!/usr/bin/env bash
set -euo pipefail
CASU2_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CHECKPOINT:?Set CHECKPOINT to the edited CASU-2 checkpoint directory.}"
PYTHON_BIN="${PYTHON:-python}"
CHECKPOINT="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]).expanduser().resolve(); print(p.parent if p.is_file() else p)' "$CHECKPOINT")"
RUN_ID="${RUN_ID:-$(basename -- "$(dirname -- "$CHECKPOINT")")}"
IMAGE_DIR="${IMAGE_DIR:-$CASU2_DIR/runs/eval-$RUN_ID}"
METRIC_DIR="${METRIC_DIR:-$CASU2_DIR/metric/$RUN_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SAMPLE_FLAGS=()
METRIC_FLAGS=()
case "${ORIGINAL_CACHE:-1}" in
  1) SAMPLE_FLAGS+=(--original-cache) ;;
  0) SAMPLE_FLAGS+=(--no-original-cache) ;;
  *) echo "ORIGINAL_CACHE must be 0 or 1." >&2; exit 2 ;;
esac
if [[ -n "${ORIGINAL_DIR:-${ORIGINAL_CACHE_DIR:-}}" ]]; then
  SAMPLE_FLAGS+=(--original-dir "${ORIGINAL_DIR:-$ORIGINAL_CACHE_DIR}")
fi
case "${SKIP_METRICS:-0}" in
  0|1) ;;
  *) echo "SKIP_METRICS must be 0 or 1." >&2; exit 2 ;;
esac
case "${CPU_OFFLOAD:-auto}" in
  1) SAMPLE_FLAGS+=(--model-cpu-offload) ;;
  0) SAMPLE_FLAGS+=(--no-model-cpu-offload) ;;
  auto) ;;
  *) echo "CPU_OFFLOAD must be 0 or 1." >&2; exit 2 ;;
esac
case "${RESUME:-1}" in
  1) SAMPLE_FLAGS+=(--resume) ;;
  0) ;;
  *) echo "RESUME must be 0 or 1." >&2; exit 2 ;;
esac
case "${LOCAL_FILES_ONLY:-0}" in
  1) SAMPLE_FLAGS+=(--local-files-only); METRIC_FLAGS+=(--local-files-only) ;;
  0) ;;
  *) echo "LOCAL_FILES_ONLY must be 0 or 1." >&2; exit 2 ;;
esac
for entry in HEIGHT:height WIDTH:width STEPS:steps GUIDANCE_SCALE:guidance-scale; do
  variable="${entry%%:*}"
  if [[ -n "${!variable:-}" ]]; then SAMPLE_FLAGS+=("--${entry#*:}" "${!variable}"); fi
done
if [[ -n "${CONTENTS:-}" ]]; then SAMPLE_FLAGS+=(--contents "$CONTENTS"); fi
if [[ -n "${PROMPT_CSV:-}" ]]; then SAMPLE_FLAGS+=(--prompt-csv "$PROMPT_CSV"); fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then SAMPLE_FLAGS+=(--dry-run); fi

if [[ "${SKIP_SAMPLE:-0}" != 1 ]]; then
  if [[ "${DRY_RUN:-0}" != 1 && "${SKIP_METRICS:-0}" != 1 ]]; then
    "$PYTHON_BIN" -c 'import torch_fidelity; from transformers import CLIPModel, CLIPProcessor'
  fi
  "$PYTHON_BIN" "$CASU2_DIR/main.py" sample \
    --checkpoint "$CHECKPOINT" --benchmark \
    --output-dir "$IMAGE_DIR" --no-save-comparisons \
    --sample-template-count "${SAMPLE_TEMPLATES:-0}" --num-samples "${NUM_SAMPLES:-10}" \
    --batch-size "${BATCH_SIZE:-1}" --seed "${SEED:-0}" \
    --device "${DEVICE:-cuda}" --dtype "${DTYPE:-auto}" --cpu-threads "${CPU_THREADS:-8}" \
    "${SAMPLE_FLAGS[@]}" "$@"
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "[dry-run] Sampling plan saved; metrics skipped."
  exit 0
fi
if [[ "${SKIP_METRICS:-0}" == 1 ]]; then
  echo "[sample complete] Metric computation skipped (SKIP_METRICS=1)."
  exit 0
fi

# A separate process releases the entire diffusion model before metric loading.
"$PYTHON_BIN" "$CASU2_DIR/main.py" metrics \
  --checkpoint "$CHECKPOINT" --image-dir "$IMAGE_DIR" --output-dir "$METRIC_DIR" \
  --metrics "${METRICS:-clip,fid}" --clip-model "${CLIP_MODEL:-openai/clip-vit-base-patch16}" \
  --fid-method "${FID_METHOD:-auto}" --device "${METRIC_DEVICE:-${DEVICE:-cuda}}" \
  --batch-size "${METRIC_BATCH_SIZE:-32}" --workers "${METRIC_WORKERS:-0}" \
  --cpu-threads "${CPU_THREADS:-8}" "${METRIC_FLAGS[@]}"
