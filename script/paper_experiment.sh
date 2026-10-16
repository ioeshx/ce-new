#!/usr/bin/env bash
# Shared runner for the six paper experiments; use the named wrappers below.
set -euo pipefail
CASU2_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:?Usage: paper_experiment.sh erase|sample CASE [extra CLI arguments]}"
EXPERIMENT="${2:?Specify one of the six paper experiment names.}"
shift 2
ARCHITECTURE="${ARCHITECTURE:-sd1}"

case "$ARCHITECTURE" in
  sd1|sdxl|sd3) MODEL_TAG="$ARCHITECTURE" ;;
  flux) MODEL_TAG="flux-${FLUX_VARIANT:-klein}" ;;
  *) echo "ARCHITECTURE must be sd1, sdxl, sd3 or flux." >&2; exit 2 ;;
esac

case "$EXPERIMENT" in
  instance_snoopy)
    TASK=instance; TARGETS="Snoopy"; CASE_ANCHORS="${SNOOPY_ANCHOR-dog}" ;;
  instance_snoopy_mickey)
    TASK=instance; TARGETS="Snoopy,Mickey"; CASE_ANCHORS="${SNOOPY_ANCHOR-dog},${MICKEY_ANCHOR-mouse}" ;;
  instance_snoopy_mickey_spongebob)
    TASK=instance; TARGETS="Snoopy,Mickey,Spongebob"
    CASE_ANCHORS="${SNOOPY_ANCHOR-dog},${MICKEY_ANCHOR-mouse},${SPONGEBOB_ANCHOR-cartoon character}" ;;
  style_van_gogh) TASK=style; TARGETS="Van Gogh"; CASE_ANCHORS="${STYLE_ANCHOR-realistic}" ;;
  style_monet) TASK=style; TARGETS="Monet"; CASE_ANCHORS="${STYLE_ANCHOR-realistic}" ;;
  style_picasso) TASK=style; TARGETS="Picasso"; CASE_ANCHORS="${STYLE_ANCHOR-realistic}" ;;
  *) echo "Unknown paper experiment: $EXPERIMENT" >&2; exit 2 ;;
esac

if [[ "$TASK" == instance ]]; then
  EVAL_CONTENTS="Hello Kitty,Mickey,Pikachu,Snoopy,Spongebob"
else
  EVAL_CONTENTS="Van Gogh,Monet,Picasso"
fi

PAPER_RUN_ROOT="${PAPER_RUN_ROOT:-$CASU2_DIR/runs/paper-$MODEL_TAG}"
PYTHON_BIN="${PYTHON:-python}"
# Resolve relative roots once, so checkpoint_path.txt remains usable from any cwd.
PAPER_RUN_ROOT="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$PAPER_RUN_ROOT")"
PAPER_IMAGE_ROOT="${PAPER_IMAGE_ROOT:-$PAPER_RUN_ROOT/images}"
PAPER_METRIC_ROOT="${PAPER_METRIC_ROOT:-$CASU2_DIR/metric/$(basename -- "$PAPER_RUN_ROOT")}" 
CASE_DIR="$PAPER_RUN_ROOT/erase/$EXPERIMENT"
DRY_RUN="${DRY_RUN:-0}"

for argument in "$@"; do
  case "$argument" in
    --dry-run) DRY_RUN=1 ;;
    --target-concepts|--target-concepts=*|--target_concepts|--target_concepts=*|\
    --task|--task=*|--erasetype|--erasetype=*|--architecture|--architecture=*|\
    --output-dir|--output-dir=*|--checkpoint|--checkpoint=*|--checkpoint-path-file|--checkpoint-path-file=*)
      echo "Paper case targets/task/paths are fixed; use ARCHITECTURE and PAPER_*_ROOT to configure the experiment." >&2
      exit 2 ;;
  esac
done

case "$DRY_RUN" in
  0|1) ;;
  *) echo "DRY_RUN must be 0 or 1." >&2; exit 2 ;;
esac

echo "[paper:$MODE] $EXPERIMENT; targets=$TARGETS"

case "$MODE" in
  erase)
    FLAGS=(--no-generate --cpu-threads "${CPU_THREADS:-8}")
    if [[ "$DRY_RUN" == 1 ]]; then
      # Plans never occupy the real erasure directories or replace checkpoints.
      CASE_DIR="$PAPER_RUN_ROOT/plans/$EXPERIMENT"
      FLAGS+=(--dry-run --overwrite)
    fi
    case "${LOCAL_FILES_ONLY:-0}" in
      1) FLAGS+=(--local-files-only) ;;
      0) ;;
      *) echo "LOCAL_FILES_ONLY must be 0 or 1." >&2; exit 2 ;;
    esac
    FLAGS+=(--checkpoint-path-file "$CASE_DIR/checkpoint_path.txt")
    if [[ "$ARCHITECTURE" == sd1 ]]; then
      RUNNER=(bash "$CASU2_DIR/script/run_sd14.sh" "$TASK")
      FLAGS+=(--device "${DEVICE:-cuda}" --dtype "${DTYPE:-auto}")
      if [[ -n "${HEIGHT:-}" ]]; then FLAGS+=(--height "$HEIGHT"); fi
      if [[ -n "${WIDTH:-}" ]]; then FLAGS+=(--width "$WIDTH"); fi
    else
      RUNNER=(bash "$CASU2_DIR/script/run_model.sh" "$ARCHITECTURE" "$TASK")
    fi
    # Each call loads SD_CKPT/the architecture's ORIGINAL base model afresh.
    TARGETS="$TARGETS" ANCHORS="$CASE_ANCHORS" CONTENTS="$EVAL_CONTENTS" OUTPUT_DIR="$CASE_DIR" \
      "${RUNNER[@]}" "${FLAGS[@]}" "$@"
    if [[ "$DRY_RUN" != 1 ]]; then
      echo "[paper checkpoint] $CASE_DIR/checkpoint"
    fi
    ;;
  sample)
    CHECKPOINT="$CASE_DIR/checkpoint"
    if [[ -f "$CASE_DIR/checkpoint_path.txt" ]]; then
      CHECKPOINT="$(< "$CASE_DIR/checkpoint_path.txt")"
    fi
    # Refuse to accidentally evaluate a different paper case's checkpoint.
    "$PYTHON_BIN" - "$CHECKPOINT" "$TASK" "$TARGETS" <<'PY'
import json
from pathlib import Path
import sys
path, task, targets = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
if not (path / "metadata.json").is_file() or not (path / "weights.safetensors").is_file():
    raise SystemExit(f"Checkpoint missing: {path}. Run this case's paper_erase script first.")
metadata = json.loads((path / "metadata.json").read_text())
normalize = lambda values: sorted(x.strip().casefold() for x in values)
if metadata.get("task") != task or normalize(metadata.get("targets", [])) != normalize(targets.split(",")):
    raise SystemExit(f"Checkpoint task/targets do not match this paper case: {path}")
PY
    case "${WITH_METRICS:-1}" in
      1) SKIP_METRICS=0 ;;
      0) SKIP_METRICS=1 ;;
      *) echo "WITH_METRICS must be 0 or 1." >&2; exit 2 ;;
    esac
    CHECKPOINT="$CHECKPOINT" CONTENTS="$EVAL_CONTENTS" \
    IMAGE_DIR="$PAPER_IMAGE_ROOT/$EXPERIMENT" METRIC_DIR="$PAPER_METRIC_ROOT/$EXPERIMENT" \
    DRY_RUN="$DRY_RUN" SKIP_METRICS="$SKIP_METRICS" \
      bash "$CASU2_DIR/script/sample_and_metrics.sh" "$@"
    ;;
  *) echo "Mode must be erase or sample." >&2; exit 2 ;;
esac
