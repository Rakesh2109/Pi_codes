#!/usr/bin/env bash
# Raspberry Pi benchmark runner — training + testing time for ML and TM.
# Covers all 4 datasets: MedSec-25, WUSTL-EHMS, TON_IoT, UNSW-NB15.
#
#   ./run_all.sh                 # all datasets
#   ./run_all.sh medsec          # single dataset (medsec|wustl|toniot|unsw)
#   TM_THREADS=1 ./run_all.sh    # force single-threaded TM (Pi core)
#
# Outputs land in ./results/

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Single-core everywhere (matches glade/benchmark canonical setup)
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TM_THREADS="${TM_THREADS:-1}"
N_BINS="${N_BINS:-15}"

if [[ $# -ge 1 ]]; then
    DATASETS=("$@")
else
    DATASETS=(medsec wustl toniot unsw)
fi

mkdir -p results

run_medsec() {
    echo "================================================================"
    echo "  ML baselines — MedSec-25"
    echo "================================================================"
    python3 ml_baselines.py 2>&1 | tee results/ml_medsec.log

    echo "================================================================"
    echo "  FastKBin prepare — MedSec-25 ($N_BINS bins)"
    echo "================================================================"
    python3 prepare_dataset_fastkbin.py "$N_BINS" 2>&1 | tee results/prep_medsec.log

    echo "================================================================"
    echo "  TM train + inference — MedSec-25"
    echo "================================================================"
    julia --threads="$TM_THREADS" inference_benchmark.jl 2>&1 | tee results/tm_medsec.log
}

run_wustl() {
    echo "================================================================"
    echo "  ML baselines — WUSTL-EHMS"
    echo "================================================================"
    python3 ml_baselines_wustl.py 2>&1 | tee results/ml_wustl.log

    echo "================================================================"
    echo "  FastKBin prepare — WUSTL-EHMS ($N_BINS bins)"
    echo "================================================================"
    python3 prepare_dataset_fastkbin_wustl.py "$N_BINS" 2>&1 | tee results/prep_wustl.log

    echo "================================================================"
    echo "  TM train + inference — WUSTL-EHMS"
    echo "================================================================"
    julia --threads="$TM_THREADS" inference_benchmark_wustl.jl 2>&1 | tee results/tm_wustl.log
}

run_toniot() {
    echo "================================================================"
    echo "  ML baselines — TON_IoT"
    echo "================================================================"
    python3 ml_baselines_toniot.py 2>&1 | tee results/ml_toniot.log

    echo "================================================================"
    echo "  FastKBin prepare — TON_IoT ($N_BINS bins)"
    echo "================================================================"
    python3 prepare_dataset_fastkbin_toniot.py "$N_BINS" 2>&1 | tee results/prep_toniot.log

    echo "================================================================"
    echo "  TM train + inference — TON_IoT"
    echo "================================================================"
    julia --threads="$TM_THREADS" inference_benchmark_toniot.jl 2>&1 | tee results/tm_toniot.log
}

run_unsw() {
    echo "================================================================"
    echo "  ML baselines — UNSW-NB15"
    echo "================================================================"
    python3 ml_baselines_unsw.py 2>&1 | tee results/ml_unsw.log

    echo "================================================================"
    echo "  FastKBin prepare — UNSW-NB15 ($N_BINS bins)"
    echo "================================================================"
    python3 prepare_dataset_fastkbin_unsw.py "$N_BINS" 2>&1 | tee results/prep_unsw.log

    echo "================================================================"
    echo "  TM train + inference — UNSW-NB15"
    echo "================================================================"
    julia --threads="$TM_THREADS" inference_benchmark_unsw.jl 2>&1 | tee results/tm_unsw.log
}

for ds in "${DATASETS[@]}"; do
    case "$ds" in
        medsec) run_medsec ;;
        wustl)  run_wustl  ;;
        toniot) run_toniot ;;
        unsw)   run_unsw   ;;
        *) echo "Unknown dataset: $ds  (use medsec|wustl|toniot|unsw)"; exit 1 ;;
    esac
done

echo
echo "================================================================"
echo "  Final combined tables (ML + GLADE_FPTM, sorted by F1)"
echo "================================================================"
python3 make_table.py

echo
echo "All done. Outputs in results/:"
ls -1 results/
