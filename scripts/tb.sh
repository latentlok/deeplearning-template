#!/usr/bin/env bash
# Every run writes to <run_dir>/tb/, so one logdir shows them all side by side.
set -euo pipefail
exec uv run tensorboard --logdir "${1:-outputs}" --port "${PORT:-6006}"
