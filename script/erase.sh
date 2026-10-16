#!/usr/bin/env bash
set -euo pipefail
CASU2_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Edit experiment settings here. Empty MODEL/RUN_ROOT uses the architecture default.
PYTHON="python"

# models
ARCHITECTURE="sd1"
FLUX_VARIANT="klein"
MODEL=""
# general settings
RUN_ROOT=""
DEVICE="cuda:0"
DTYPE="auto"
SEED=0
CPU_THREADS=8

# specific options
PARAMS="V"
BETA=1.0
CONTEXT_TEMPLATES=30
RETAIN_COUNT=128
SNOOPY_ANCHOR="dog"
MICKEY_ANCHOR="mouse"
SPONGEBOB_ANCHOR="cartoon character"
STYLE_ANCHOR="realistic"

# Add options such as --local-files-only, --steps 20, --height 512 here.
ERASE_OPTIONS=(
  --contextual-anchor
  --no-selective-update
)

EXPERIMENT="${1:?Usage: erase.sh CASE [main.py erase options]}"
shift
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --architecture) ARCHITECTURE="$2"; shift 2 ;;
    --architecture=*) ARCHITECTURE="${1#*=}"; shift ;;
    --flux-variant) FLUX_VARIANT="$2"; shift 2 ;;
    --flux-variant=*) FLUX_VARIANT="${1#*=}"; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --device=*) DEVICE="${1#*=}"; shift ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --run-root=*) RUN_ROOT="${1#*=}"; shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

case "$EXPERIMENT" in
  instance_snoopy) TASK="instance"; TARGETS="Snoopy"; ANCHORS="$SNOOPY_ANCHOR" ;;
  instance_snoopy_mickey) TASK="instance"; TARGETS="Snoopy,Mickey"; ANCHORS="$SNOOPY_ANCHOR,$MICKEY_ANCHOR" ;;
  instance_snoopy_mickey_spongebob) TASK="instance"; TARGETS="Snoopy,Mickey,Spongebob"; ANCHORS="$SNOOPY_ANCHOR,$MICKEY_ANCHOR,$SPONGEBOB_ANCHOR" ;;
  style_van_gogh) TASK="style"; TARGETS="Van Gogh"; ANCHORS="$STYLE_ANCHOR" ;;
  style_monet) TASK="style"; TARGETS="Monet"; ANCHORS="$STYLE_ANCHOR" ;;
  style_picasso) TASK="style"; TARGETS="Picasso"; ANCHORS="$STYLE_ANCHOR" ;;
  *) echo "Unknown experiment: $EXPERIMENT" >&2; exit 2 ;;
esac

MODEL_TAG="$ARCHITECTURE"
if [[ "$ARCHITECTURE" == flux ]]; then 
  MODEL_TAG="flux-$FLUX_VARIANT"; 
fi
if [[ -z "$RUN_ROOT" ]]; then 
  RUN_ROOT="$CASU2_DIR/runs/paper-$MODEL_TAG"; 
fi

OUTPUT_DIR="$RUN_ROOT/erase/$EXPERIMENT"
if [[ -n "$MODEL" ]]; then 
ERASE_OPTIONS+=(--sd-ckpt "$MODEL"); 
fi

"$PYTHON" "$CASU2_DIR/main.py" erase \
  --architecture "$ARCHITECTURE" \
  --flux-variant "$FLUX_VARIANT" \
  --task "$TASK" \
  --target-concepts "$TARGETS" \
  --anchor-concepts "$ANCHORS" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --seed "$SEED" \
  --cpu-threads "$CPU_THREADS" \
  --params "$PARAMS" \
  --beta "$BETA" \
  --context-template-count "$CONTEXT_TEMPLATES" \
  --retain-count "$RETAIN_COUNT" \
  --output-dir "$OUTPUT_DIR" \
  --no-generate \
  "${ERASE_OPTIONS[@]}" \
  "${EXTRA_ARGS[@]}"

## no-generate 关闭生成原始和编辑的对照图片，已经有html网页了
