#!/usr/bin/env bash
# Launch MAWAM closed-loop policy server (`benchmark` 25/50 or `realtime` 4/8)
set -euo pipefail

TASK="${TASK:-open_door_place_pot}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to MAWAM Stage 2 checkpoint directory}"
BASE_MODEL="${BASE_MODEL:-./checkpoints/lingbot-va-base}"
TEXT_CACHE="${TEXT_CACHE:?Set TEXT_CACHE to text/text_cache.pt}"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-./third_party/LingBot-VA}"
PRESET="${PRESET:-benchmark}"
PORT="${PORT:-25300}"

python3 -m mawam.inference.policy_server \
  --task "${TASK}" \
  --checkpoint "${CHECKPOINT}" \
  --base-model "${BASE_MODEL}" \
  --text-cache "${TEXT_CACHE}" \
  --upstream-root "${UPSTREAM_ROOT}" \
  --preset "${PRESET}" \
  --port "${PORT}" \
  "$@"
