#!/usr/bin/env bash
# All arguments are forwarded; PYTHON and CUDA_ID may be set by the caller.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${CUDA_ID:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_ID"
fi
exec "${PYTHON:-python3}" "$SCRIPT_DIR/control_frequency_sweep.py" "$@"
