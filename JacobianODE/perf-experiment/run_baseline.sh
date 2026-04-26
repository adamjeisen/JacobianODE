#!/usr/bin/env bash
# Pre-flight + baseline runner. Two passes per config with the same seed,
# captures wall-clock time and final-step loss for reproducibility check.
#
# Usage (inside container):
#   bash JacobianODE/perf-experiment/run_baseline.sh <config_name> <seed> <max_steps>
#
# Outputs to /tmp/perf-exp/<config_name>/{run1,run2}/ — each contains the
# hydra run dir + the wandb-offline dir from which we extract metrics.

set -euo pipefail

CONFIG="$1"
SEED="${2:-42}"
MAX_STEPS="${3:-20}"

OUTDIR=/tmp/perf-exp/${CONFIG}
rm -rf "$OUTDIR" && mkdir -p "$OUTDIR"

# Pick batch_size based on config — wmtask uses N1+N2=128 dim and is heavier
case "$CONFIG" in
  *wmtask*) BS=4 ;;
  *) BS=8 ;;
esac

run_one() {
    local label="$1"
    local rundir="$OUTDIR/$label"
    mkdir -p "$rundir"
    local t0=$(date +%s.%N)
    WANDB_MODE=offline uv run --no-sync python -m JacobianODE.jacobians.run_jacobians \
        experiment="$CONFIG" \
        +training.trainer_params.max_steps="$MAX_STEPS" \
        training.trainer_params.limit_train_batches=200 \
        training.trainer_params.limit_val_batches=10 \
        training.batch_size="$BS" \
        training.logger_save_dirs="$rundir/lightning" \
        training.logger.save_dir="$rundir/lightning" \
        data.flow.random_state="$SEED" \
        hydra.run.dir="$rundir/hydra" \
        > "$rundir/stdout.log" 2> "$rundir/stderr.log" || {
            echo "RUN FAILED — see $rundir/{stdout,stderr}.log"
            tail -30 "$rundir/stderr.log"
            return 1
        }
    local t1=$(date +%s.%N)
    local elapsed=$(python3 -c "print(f'{$t1 - $t0:.2f}')")
    echo "$label elapsed=${elapsed}s"
    # Extract per-step loss series from wandb-history.jsonl
    local hist=$(find "$rundir/lightning/wandb" -name 'wandb-history.jsonl' | head -1)
    if [ -n "$hist" ]; then
        python3 -c "
import json, sys
losses = []
for line in open('$hist'):
    d = json.loads(line)
    if 'train/loss' in d or 'loss' in d:
        losses.append((d.get('_step'), d.get('train/loss', d.get('loss'))))
print('  steps:', len(losses))
if losses:
    print('  first 3:', losses[:3])
    print('  last 3 :', losses[-3:])
"
    fi
}

echo "=== ${CONFIG}: pass 1 (seed=${SEED}, max_steps=${MAX_STEPS}, batch_size=${BS}) ==="
run_one run1
echo
echo "=== ${CONFIG}: pass 2 (same seed) ==="
run_one run2
