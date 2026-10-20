#!/usr/bin/env bash
# CA + closed form + the 50% of U-Net attention layers nearest mid_block.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/paper_su.sh" middle "$@"
