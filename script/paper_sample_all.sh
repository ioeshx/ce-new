#!/usr/bin/env bash
# Edit CUDA_VISIBLE_DEVICES and the experiment commands below to change GPU assignments.
export HF_ENDPOINT=https://hf-mirror.com

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Default: one GPU. Set NUM_GPUS=2 when launching, or change the default below.
NUM_GPUS="${NUM_GPUS:-1}"
case "$NUM_GPUS" in
  1)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    bash "$SCRIPT_DIR/paper_sample_instance_snoopy.sh" "$@"
    bash "$SCRIPT_DIR/paper_sample_instance_snoopy_mickey.sh" "$@"
    bash "$SCRIPT_DIR/paper_sample_instance_snoopy_mickey_spongebob.sh" "$@"
    bash "$SCRIPT_DIR/paper_sample_style_van_gogh.sh" "$@"
    bash "$SCRIPT_DIR/paper_sample_style_monet.sh" "$@"
    bash "$SCRIPT_DIR/paper_sample_style_picasso.sh" "$@"
    exit 0
    ;;
  2) ;;
  *) echo "NUM_GPUS must be 1 or 2." >&2; exit 2 ;;
esac

# Give each background shell its own process group, so Ctrl-C stops its descendants too.
set -m
pid_gpu0=""
pid_gpu1=""
stop_generation() {
  trap - INT TERM
  for pid in "$pid_gpu0" "$pid_gpu1"; do
    if [[ -n "$pid" ]]; then kill -TERM -- "-$pid" 2>/dev/null || true; fi
  done
  wait "$pid_gpu0" "$pid_gpu1" 2>/dev/null || true
}
trap 'stop_generation; exit 130' INT
trap 'stop_generation; exit 143' TERM

# GPU 0: instance experiments, sequentially. Change the GPU number/tasks here.
(
  export CUDA_VISIBLE_DEVICES=0
  bash "$SCRIPT_DIR/paper_sample_instance_snoopy.sh" "$@"
  bash "$SCRIPT_DIR/paper_sample_instance_snoopy_mickey_spongebob.sh" "$@"
  bash "$SCRIPT_DIR/paper_sample_instance_snoopy_mickey.sh" "$@"
) &
pid_gpu0=$!

# GPU 1: style experiments, sequentially, in parallel with GPU 0.
(
  export CUDA_VISIBLE_DEVICES=1
  bash "$SCRIPT_DIR/paper_sample_style_van_gogh.sh" "$@"
  bash "$SCRIPT_DIR/paper_sample_style_monet.sh" "$@"
  bash "$SCRIPT_DIR/paper_sample_style_picasso.sh" "$@"
) &
pid_gpu1=$!

# Wait for both groups and preserve a failure from either group.
status=0
wait "$pid_gpu0" || status=$?
wait "$pid_gpu1" || status=$?
trap - INT TERM
exit "$status"
