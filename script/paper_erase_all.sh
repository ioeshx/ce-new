#!/usr/bin/env bash
# Erasure always runs serially. Change 0 here, or set CUDA_VISIBLE_DEVICES when launching.
export HF_ENDPOINT=https://hf-mirror.com

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

bash "$SCRIPT_DIR/paper_erase_instance_snoopy.sh" "$@"
bash "$SCRIPT_DIR/paper_erase_instance_snoopy_mickey.sh" "$@"
bash "$SCRIPT_DIR/paper_erase_instance_snoopy_mickey_spongebob.sh" "$@"
bash "$SCRIPT_DIR/paper_erase_style_van_gogh.sh" "$@"
bash "$SCRIPT_DIR/paper_erase_style_monet.sh" "$@"
bash "$SCRIPT_DIR/paper_erase_style_picasso.sh" "$@"
