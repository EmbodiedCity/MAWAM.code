#!/usr/bin/env bash
# Run 100 independent closed-loop trials on MA-Bench
set -euo pipefail

TASK="${TASK:-open_door_place_pot}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-25300}"
NUM_TRIALS="${NUM_TRIALS:-100}"
OUTPUT="${OUTPUT:-./runs/eval_${TASK}.json}"

python3 -m ma_bench.eval_runner \
  --task "${TASK}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --num-trials "${NUM_TRIALS}" \
  --output "${OUTPUT}" \
  "$@"
