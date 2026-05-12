#!/usr/bin/env bash
# One-command Pi 5 benchmark runner.
#
# Usage on the Pi:
#     bash run.sh                # everything (slow on kNN_5/RandomForest)
#     bash run.sh --skip-slow    # drop kNN_5 + RandomForest
#
# What it does:
#   1. Installs Python dependencies into the user site (no sudo needed).
#   2. Runs bench_pi.py.
#   3. Prints results_pi.json at the end.

set -euo pipefail

cd "$(dirname "$0")"

PY=${PY:-python3}
echo "==> Python: $($PY --version)"

# 1. Dependencies — kept minimal. Numba JIT-compiles the TM kernel on
# first call; first run will be slow (~30-60 s extra) for that reason.
DEPS=(numpy scikit-learn xgboost numba zstandard)
echo "==> Ensuring dependencies: ${DEPS[*]}"
$PY -m pip install --user --upgrade --quiet --no-warn-script-location "${DEPS[@]}"

# 2. Benchmark
echo "==> Running bench_pi.py"
$PY bench_pi.py "$@"

# 3. Show output location
if [ -f results_pi.json ]; then
    echo
    echo "==> Done. Results saved to: $(pwd)/results_pi.json"
    echo "==> Size: $(du -h results_pi.json | cut -f1)"
    echo
    echo "Copy the file back with:"
    echo "    scp $(whoami)@\$(hostname -I | awk '{print \$1}'):$(pwd)/results_pi.json ."
fi
