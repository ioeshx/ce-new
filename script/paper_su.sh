#!/usr/bin/env bash
# Shared multi-model launcher for paper_su_middle.sh and paper_su_vlm.sh.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CASU2_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
METHOD="${1:?Expected middle or vlm}"
shift
MODE="all"
if [[ $# -gt 0 && "$1" != --* ]]; then MODE="$1"; shift; fi
case "$METHOD" in middle|vlm) ;; *) echo "Unknown SD1 SU method: $METHOD" >&2; exit 2 ;; esac
case "$MODE" in erase|sample|all) ;; *) echo "Usage: paper_su_${METHOD}.sh [erase|sample|all] [options]" >&2; exit 2 ;; esac

# Edit these shell settings to match the server. Erasure always uses one GPU.
# Sampling is dispatched by paper_sample_all.sh; edit NUM_GPUS/DEVICE0/DEVICE1 there.
ARCHITECTURE="sd1"
FLUX_VARIANT="klein"
RUN_ROOT=""
ERASE_DEVICE="cuda:0"
EXPERIMENT="all"  # Or pass --experiment instance_snoopy, style_monet, etc.
ERASE_ARGS=()
SAMPLE_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --run-root=*) RUN_ROOT="${1#*=}"; shift ;;
    --experiment) EXPERIMENT="$2"; shift 2 ;;
    --experiment=*) EXPERIMENT="${1#*=}"; shift ;;
    --device) ERASE_DEVICE="$2"; SAMPLE_ARGS+=("$1" "$2"); shift 2 ;;
    --device=*) ERASE_DEVICE="${1#*=}"; SAMPLE_ARGS+=("$1"); shift ;;
    --architecture) ARCHITECTURE="$2"; shift 2 ;;
    --architecture=*) ARCHITECTURE="${1#*=}"; shift ;;
    --flux-variant) FLUX_VARIANT="$2"; shift 2 ;;
    --flux-variant=*) FLUX_VARIANT="${1#*=}"; shift ;;
    --su-vlm-model|--su-vlm-image-size|--su-prompts|--su-min-fraction|--su-max-fraction|--selection-template-count|--selection-samples|--target-description|--sd-ckpt|--sd_ckpt|--params|--beta|--retain-count|--context-template-count|--condition-branches|--token-mode|--solve-device|--rank-rtol|--preserve-rank|--encode-batch-size)
      ERASE_ARGS+=("$1" "$2"); shift 2 ;;
    --su-vlm-model=*|--su-vlm-image-size=*|--su-prompts=*|--su-min-fraction=*|--su-max-fraction=*|--selection-template-count=*|--selection-samples=*|--target-description=*|--sd-ckpt=*|--sd_ckpt=*|--params=*|--beta=*|--retain-count=*|--context-template-count=*|--condition-branches=*|--token-mode=*|--solve-device=*|--rank-rtol=*|--preserve-rank=*|--encode-batch-size=*)
      ERASE_ARGS+=("$1"); shift ;;
    --overwrite|--generate|--no-generate|--protect-null|--no-protect-null|--include-padding|--no-include-padding)
      ERASE_ARGS+=("$1"); shift ;;
    --no-metrics|--metrics-only|--no-save-comparisons|--save-comparisons|--resume)
      SAMPLE_ARGS+=("$1"); shift ;;
    --metric-dir|--metric-device|--original-dir|--num-samples|--sample-template-count|--batch-size)
      SAMPLE_ARGS+=("$1" "$2"); shift 2 ;;
    --metric-dir=*|--metric-device=*|--original-dir=*|--num-samples=*|--sample-template-count=*|--batch-size=*)
      SAMPLE_ARGS+=("$1"); shift ;;
    --seed|--steps|--height|--width|--guidance-scale|--dtype|--cpu-threads|--max-sequence-length)
      ERASE_ARGS+=("$1" "$2"); SAMPLE_ARGS+=("$1" "$2"); shift 2 ;;
    --seed=*|--steps=*|--height=*|--width=*|--guidance-scale=*|--dtype=*|--cpu-threads=*|--max-sequence-length=*|--local-files-only|--model-cpu-offload|--no-model-cpu-offload)
      ERASE_ARGS+=("$1"); SAMPLE_ARGS+=("$1"); shift ;;
    *) echo "Unknown launcher option: $1. Use erase.sh/sample.sh directly for other options." >&2; exit 2 ;;
  esac
done

case "$ARCHITECTURE" in sd1|sdxl|sd3|flux) ;; *) echo "Unknown architecture: $ARCHITECTURE" >&2; exit 2 ;; esac
MODEL_TAG="$ARCHITECTURE"
if [[ "$ARCHITECTURE" == flux ]]; then MODEL_TAG="flux-$FLUX_VARIANT"; fi
if [[ -z "$RUN_ROOT" ]]; then RUN_ROOT="$CASU2_DIR/runs/paper-$MODEL_TAG-su-$METHOD"; fi
COMMON=(--architecture "$ARCHITECTURE" --flux-variant "$FLUX_VARIANT" --run-root "$RUN_ROOT")
if [[ "$MODE" == erase || "$MODE" == all ]]; then
  # Last options deliberately enforce CA+SU and this wrapper's selection method.
  SU_ARGS=(--contextual-anchor --selective-update --selection-method "$METHOD")
  if [[ "$EXPERIMENT" == all ]]; then
    bash "$SCRIPT_DIR/paper_erase_all.sh" "${COMMON[@]}" --device "$ERASE_DEVICE" "${ERASE_ARGS[@]}" "${SU_ARGS[@]}"
  else
    bash "$SCRIPT_DIR/erase.sh" "$EXPERIMENT" "${COMMON[@]}" --device "$ERASE_DEVICE" "${ERASE_ARGS[@]}" "${SU_ARGS[@]}"
  fi
fi
if [[ "$MODE" == sample || "$MODE" == all ]]; then
  if [[ "$EXPERIMENT" == all ]]; then
    bash "$SCRIPT_DIR/paper_sample_all.sh" "${COMMON[@]}" "${SAMPLE_ARGS[@]}"
  else
    bash "$SCRIPT_DIR/sample.sh" "$EXPERIMENT" "${COMMON[@]}" --device "$ERASE_DEVICE" "${SAMPLE_ARGS[@]}"
  fi
fi
