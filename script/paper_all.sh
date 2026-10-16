#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi

case "$MODE" in
  erase) exec bash "$SCRIPT_DIR/paper_erase_all.sh" "$@" ;;
  sample) exec bash "$SCRIPT_DIR/paper_sample_all.sh" "$@" ;;
  all)
    # Options in all mode apply to both phases; phase-specific settings live in
    # erase.sh/sample.sh or can be passed using the erase/sample mode separately.
    bash "$SCRIPT_DIR/paper_erase_all.sh" "$@"
    exec bash "$SCRIPT_DIR/paper_sample_all.sh" "$@"
    ;;
  *) echo "Usage: paper_all.sh [erase|sample|all]" >&2; exit 2 ;;
esac
