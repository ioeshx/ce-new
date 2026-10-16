#!/usr/bin/env bash
export HF_ENDPOINT=https://hf-mirror.com

set -euo pipefail
CASU2_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ARCH="${1:?Usage: run_model.sh sdxl|sd3|flux instance|style [extra arguments]}"
TASK="${2:?Specify instance or style}"
shift 2
VARIANT="${FLUX_VARIANT:-klein}"
DEFAULT_SEQUENCE_LENGTH=256
DEFAULT_ENCODE_BATCH=4
DEFAULT_CPU_OFFLOAD=0
case "$ARCH" in
  sdxl) MODEL="stabilityai/stable-diffusion-xl-base-1.0"; DEFAULT_STEPS=20; DEFAULT_GUIDANCE=5.0 ;;
  sd3) MODEL="stabilityai/stable-diffusion-3-medium-diffusers"; DEFAULT_STEPS=28; DEFAULT_GUIDANCE=7.0 ;;
  flux)
    case "$VARIANT" in
      klein)
        MODEL="black-forest-labs/FLUX.2-klein-4B"; DEFAULT_STEPS=4; DEFAULT_GUIDANCE=1.0
        DEFAULT_SEQUENCE_LENGTH=512; DEFAULT_ENCODE_BATCH=1; DEFAULT_CPU_OFFLOAD=1 ;;
      dev) MODEL="black-forest-labs/FLUX.1-dev"; DEFAULT_STEPS=28; DEFAULT_GUIDANCE=3.5 ;;
      schnell) MODEL="black-forest-labs/FLUX.1-schnell"; DEFAULT_STEPS=4; DEFAULT_GUIDANCE=0 ;;
      *) echo "FLUX_VARIANT must be klein, dev or schnell." >&2; exit 2 ;;
    esac ;;
  *) echo "Architecture must be sdxl, sd3 or flux." >&2; exit 2 ;;
esac
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
  1) FLAGS+=(--selective-update --selection-order "${SELECTION_ORDER:-proxy}" --selection-judge "${SELECTION_JUDGE:-manual}") ;;
  0) FLAGS+=(--no-selective-update) ;;
  *) echo "SU must be 0 or 1." >&2; exit 2 ;;
esac
if [[ "${DEVICE:-cuda}" != cuda* ]]; then DEFAULT_CPU_OFFLOAD=0; fi
case "${CPU_OFFLOAD:-$DEFAULT_CPU_OFFLOAD}" in
  1) FLAGS+=(--model-cpu-offload) ;;
  0) FLAGS+=(--no-model-cpu-offload) ;;
  *) echo "CPU_OFFLOAD must be 0 or 1." >&2; exit 2 ;;
esac
if [[ -n "${OUTPUT_DIR:-}" ]]; then FLAGS+=(--output-dir "$OUTPUT_DIR"); fi
exec "${PYTHON:-python}" "$CASU2_DIR/main.py" erase \
  --architecture "$ARCH" --sd-ckpt "${SD_CKPT:-$MODEL}" \
  --task "$TASK" --target-concepts "$TARGETS" --anchor-concepts "$ANCHORS" \
  --condition-branches "${BRANCHES:-both}" --params "${PARAMS:-V}" \
  --context-template-count "${CONTEXT_TEMPLATES:-30}" \
  --max-fit-tokens "${MAX_FIT_TOKENS:-16}" --encode-batch-size "${ENCODE_BATCH_SIZE:-$DEFAULT_ENCODE_BATCH}" \
  --max-sequence-length "${MAX_SEQUENCE_LENGTH:-$DEFAULT_SEQUENCE_LENGTH}" --flux-variant "$VARIANT" \
  --retain-count "${RETAIN_COUNT:-128}" --beta "${BETA:-1.0}" \
  --contents "${CONTENTS:-$TARGETS}" --sample-template-count "${SAMPLE_TEMPLATES:-8}" \
  --num-samples "${NUM_SAMPLES:-1}" --batch-size "${BATCH_SIZE:-1}" \
  --height "${HEIGHT:-1024}" --width "${WIDTH:-1024}" \
  --steps "${STEPS:-$DEFAULT_STEPS}" --guidance-scale "${GUIDANCE_SCALE:-$DEFAULT_GUIDANCE}" \
  --true-cfg-scale "${TRUE_CFG_SCALE:-1.0}" \
  --dtype "${DTYPE:-auto}" --device "${DEVICE:-cuda}" --seed "${SEED:-0}" \
  "${FLAGS[@]}" "$@"
