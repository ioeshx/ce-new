#!/usr/bin/env bash
export HF_ENDPOINT=https://hf-mirror.com

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi

case "$MODE" in
  erase) exec bash "$SCRIPT_DIR/paper_erase_all.sh" "$@" ;;
  sample) exec bash "$SCRIPT_DIR/paper_sample_all.sh" "$@" ;;
  all)
    if [[ $# -gt 0 ]]; then
      echo "Configure all phases with environment variables; pass CLI arguments to erase/sample separately." >&2
      exit 2
    fi
    bash "$SCRIPT_DIR/paper_erase_all.sh"
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
      echo "[paper dry-run] Six erasure plans prepared; real checkpoints are needed for sampling."
      exit 0
    fi
    exec bash "$SCRIPT_DIR/paper_sample_all.sh"
    ;;
  *) echo "Usage: paper_all.sh [erase|sample|all]" >&2; exit 2 ;;
esac
