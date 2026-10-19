#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Edit the erasure GPU here; model and erasure settings are in erase.sh.
DEVICE="cuda:0"
export HF_ENDPOINT=https://hf-mirror.com

bash "$SCRIPT_DIR/erase.sh" instance_snoopy --device "$DEVICE" "$@"
bash "$SCRIPT_DIR/erase.sh" instance_snoopy_mickey --device "$DEVICE" "$@"
bash "$SCRIPT_DIR/erase.sh" instance_snoopy_mickey_spongebob --device "$DEVICE" "$@"
bash "$SCRIPT_DIR/erase.sh" style_van_gogh --device "$DEVICE" "$@"
bash "$SCRIPT_DIR/erase.sh" style_monet --device "$DEVICE" "$@"
bash "$SCRIPT_DIR/erase.sh" style_picasso --device "$DEVICE" "$@"
