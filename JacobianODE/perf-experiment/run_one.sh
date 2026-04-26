#!/usr/bin/env bash
# Run one timed training pass with the PerStepTimingCallback.
#
# Usage: bash run_one.sh <label> <config> <max_steps> <seed>
#
# Output: /tmp/perf-exp/<config>/<label>/perf.json (per-step trace)
set -euo pipefail

LABEL="$1"
CONFIG="$2"
MAX_STEPS="${3:-25}"
SEED="${4:-42}"

case "$CONFIG" in
  *wmtask*) BS=4 ;;
  *) BS=8 ;;
esac

OUTBASE=/tmp/perf-exp/${CONFIG}/${LABEL}
mkdir -p "$OUTBASE"

WANDB_MODE=offline uv run --no-sync python \
  /workspaces/JacobianODE/JacobianODE/perf-experiment/run_with_callback.py \
  "$OUTBASE/perf.json" \
  "$CONFIG" \
  "$MAX_STEPS" \
  "$BS" \
  "$SEED" \
  "$OUTBASE" \
  hydra.run.dir="$OUTBASE/hydra" \
  "${@:5}" \
  > "$OUTBASE/stdout.log" 2> "$OUTBASE/stderr.log" || {
    echo "RUN FAILED — last 30 lines of stderr:"
    tail -30 "$OUTBASE/stderr.log"
    exit 1
  }

echo "wrote: $OUTBASE/perf.json"
