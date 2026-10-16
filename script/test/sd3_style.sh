#!/usr/bin/env bash
export HF_ENDPOINT=https://hf-mirror.com

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_model.sh" sd3 style "$@"
