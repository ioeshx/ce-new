#!/usr/bin/env bash
set -euo pipefail
CASU2_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_ENDPOINT=https://hf-mirror.com


# Edit generation and metric settings here. Architecture selects the run directory;
# the model and sampling defaults are read from the checkpoint by main.py.
PYTHON="python"
ARCHITECTURE="sd1"
FLUX_VARIANT="klein"
RUN_ROOT=""
CHECKPOINT=""
IMAGE_DIR=""
METRIC_DIR=""
DEVICE="cuda:0"
DTYPE="auto"
SEED=0
CPU_THREADS=8
SAMPLE_TEMPLATES=0
NUM_SAMPLES=10
BATCH_SIZE=1
SKIP_SAMPLE=0
WITH_METRICS=1
LOCAL_FILES_ONLY=0
METRIC_DEVICE=""  # Empty means use DEVICE.
METRICS="clip,fid"
CLIP_MODEL="openai/clip-vit-base-patch16"
FID_METHOD="auto"
METRIC_BATCH_SIZE=32
METRIC_WORKERS=0
# Add options such as --steps 20, --height 512, --original-dir /path here.
# --save-comparisons:保存对照图片，左边原始右边编辑图片
SAMPLE_OPTIONS=(
  --resume
  --original-cache
  --no-save-comparisons
)

METRIC_OPTIONS=()

EXPERIMENT="${1:?Usage: sample.sh CASE [main.py sample options]}"
shift
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --architecture) ARCHITECTURE="$2"; shift 2 ;;
    --architecture=*) ARCHITECTURE="${1#*=}"; shift ;;
    --flux-variant) FLUX_VARIANT="$2"; shift 2 ;;
    --flux-variant=*) FLUX_VARIANT="${1#*=}"; shift ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --run-root=*) RUN_ROOT="${1#*=}"; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --device=*) DEVICE="${1#*=}"; shift ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --checkpoint=*) CHECKPOINT="${1#*=}"; shift ;;
    --output-dir) IMAGE_DIR="$2"; shift 2 ;;
    --output-dir=*) IMAGE_DIR="${1#*=}"; shift ;;
    --metric-dir) METRIC_DIR="$2"; shift 2 ;;
    --metric-dir=*) METRIC_DIR="${1#*=}"; shift ;;
    --metric-device) METRIC_DEVICE="$2"; shift 2 ;;
    --metric-device=*) METRIC_DEVICE="${1#*=}"; shift ;;
    --local-files-only) LOCAL_FILES_ONLY=1; shift ;;
    --no-metrics) WITH_METRICS=0; shift ;;
    --metrics-only) SKIP_SAMPLE=1; shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

MODEL_TAG="$ARCHITECTURE"
if [[ "$ARCHITECTURE" == flux ]]; then MODEL_TAG="flux-$FLUX_VARIANT"; fi
if [[ -z "$RUN_ROOT" ]]; then RUN_ROOT="$CASU2_DIR/runs/paper-$MODEL_TAG"; fi
if [[ -z "$CHECKPOINT" ]]; then CHECKPOINT="$RUN_ROOT/erase/$EXPERIMENT/checkpoint"; fi
if [[ -z "$IMAGE_DIR" ]]; then IMAGE_DIR="$RUN_ROOT/images/$EXPERIMENT"; fi
if [[ -z "$METRIC_DIR" ]]; then METRIC_DIR="$CASU2_DIR/metric/$(basename -- "$RUN_ROOT")/$EXPERIMENT"; fi
if [[ -z "$METRIC_DEVICE" ]]; then METRIC_DEVICE="$DEVICE"; fi
if [[ "$LOCAL_FILES_ONLY" == 1 ]]; then
  SAMPLE_OPTIONS+=(--local-files-only)
  METRIC_OPTIONS+=(--local-files-only)
fi

if [[ "$SKIP_SAMPLE" == 0 ]]; then
  "$PYTHON" "$CASU2_DIR/main.py" sample \
    --checkpoint "$CHECKPOINT" \
    --benchmark \
    --output-dir "$IMAGE_DIR" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --seed "$SEED" \
    --cpu-threads "$CPU_THREADS" \
    --sample-template-count "$SAMPLE_TEMPLATES" \
    --num-samples "$NUM_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    "${SAMPLE_OPTIONS[@]}" \
    "${EXTRA_ARGS[@]}"
fi

# The sampling process exits before CLIP/FID loads, releasing the diffusion model.
if [[ "$WITH_METRICS" == 1 ]]; then
  "$PYTHON" "$CASU2_DIR/main.py" metrics \
    --checkpoint "$CHECKPOINT" \
    --image-dir "$IMAGE_DIR" \
    --output-dir "$METRIC_DIR" \
    --metrics "$METRICS" \
    --clip-model "$CLIP_MODEL" \
    --fid-method "$FID_METHOD" \
    --device "$METRIC_DEVICE" \
    --cpu-threads "$CPU_THREADS" \
    --batch-size "$METRIC_BATCH_SIZE" \
    --workers "$METRIC_WORKERS" \
    "${METRIC_OPTIONS[@]}"
fi
