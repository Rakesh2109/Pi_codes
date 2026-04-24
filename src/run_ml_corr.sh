#!/usr/bin/env bash
# Launch ml_baselines_corr.py in the background.
# Survives terminal/VSCode close — output goes to results_corr/run.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$SCRIPT_DIR/results_corr/run.log"

mkdir -p "$SCRIPT_DIR/results_corr"

echo "Starting ml_baselines_corr.py in background..."
echo "Log: $LOG"
echo "Check progress: tail -f $LOG"

nohup python3 "$SCRIPT_DIR/ml_baselines_corr.py" > "$LOG" 2>&1 &
echo "PID: $!"
echo $! > "$SCRIPT_DIR/results_corr/run.pid"
