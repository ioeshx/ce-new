#!/usr/bin/env bash
# CA + closed form + VLM selection among nested 50%..75% editable layer/module sets.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/paper_su.sh" vlm "$@"
