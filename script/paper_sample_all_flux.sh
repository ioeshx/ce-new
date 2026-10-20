#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HF_ENDPOINT=https://hf-mirror.com

ARICHITECTURE="flux"

# Edit GPU assignments here; generation and metric settings are in sample.sh.
NUM_GPUS=2
DEVICE0="cuda:0"
DEVICE1="cuda:1"
case "$NUM_GPUS" in
  1)
    bash "$SCRIPT_DIR/sample.sh" instance_snoopy --device "$DEVICE0" --architecture "$ARICHITECTURE" "$@" > logs/flux_sample_instance.log
    bash "$SCRIPT_DIR/sample.sh" instance_snoopy_mickey --device "$DEVICE0" --architecture "$ARICHITECTURE" "$@" >> logs/flux_sample_instance.log
    bash "$SCRIPT_DIR/sample.sh" instance_snoopy_mickey_spongebob --device "$DEVICE0" --architecture "$ARICHITECTURE" "$@" >> logs/flux_sample_instance.log
    bash "$SCRIPT_DIR/sample.sh" style_van_gogh --device "$DEVICE0" --architecture "$ARICHITECTURE" "$@"  > logs/flux_sample_style.log
    bash "$SCRIPT_DIR/sample.sh" style_monet --device "$DEVICE0" --architecture "$ARICHITECTURE" "$@" >> logs/flux_sample_style.log
    bash "$SCRIPT_DIR/sample.sh" style_picasso --device "$DEVICE0" --architecture "$ARICHITECTURE" "$@" >> logs/flux_sample_style.log
    exit 0
    ;;
  2) ;;
  *) echo "NUM_GPUS must be 1 or 2." >&2; exit 2 ;;
esac

# Give each background shell its own process group, so Ctrl-C stops its descendants too.
set -m
pid_gpu0=""
pid_gpu1=""
# call stop_generation() to terminate both groups and wait for them to exit.
stop_generation() {
  trap - INT TERM
  for pid in "$pid_gpu0" "$pid_gpu1"; do
    if [[ -n "$pid" ]]; 
      then kill -TERM -- "-$pid" 2>/dev/null || true; 
    fi
  done
  wait "$pid_gpu0" "$pid_gpu1" 2>/dev/null || true
}
trap 'stop_generation; exit 130' INT
trap 'stop_generation; exit 143' TERM


# First GPU: instance experiments, sequentially.
(
  bash "$SCRIPT_DIR/sample.sh" instance_snoopy --device "$DEVICE0" --architecture "$ARICHITECTURE" "$@" > logs/flux_sample_instance.log
  bash "$SCRIPT_DIR/sample.sh" instance_snoopy_mickey_spongebob --device "$DEVICE0" --architecture "$ARICHITECTURE" "$@" >> logs/flux_sample_instance.log
  bash "$SCRIPT_DIR/sample.sh" instance_snoopy_mickey --device "$DEVICE0" --architecture "$ARICHITECTURE" "$@" >> logs/flux_sample_instance.log
) &
pid_gpu0=$!

# Second GPU: style experiments, in parallel with the instance group.
(
  bash "$SCRIPT_DIR/sample.sh" style_van_gogh --device "$DEVICE1" --architecture "$ARICHITECTURE" "$@" > logs/flux_sample_style.log
  bash "$SCRIPT_DIR/sample.sh" style_monet --device "$DEVICE1" --architecture "$ARICHITECTURE" "$@" >> logs/flux_sample_style.log
  bash "$SCRIPT_DIR/sample.sh" style_picasso --device "$DEVICE1" --architecture "$ARICHITECTURE" "$@" >> logs/flux_sample_style.log
) &
pid_gpu1=$!

# Wait for both groups and preserve a failure from either group.
status=0
wait "$pid_gpu0" || status=$?
wait "$pid_gpu1" || status=$?
trap - INT TERM
exit "$status"
