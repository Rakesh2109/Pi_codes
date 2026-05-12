#!/usr/bin/env python3
"""
Step 1+2: Binarize with GLADE, train TM on pre-binarized data, save rules.

For each dataset:
  1. Load + preprocess
  2. Fit GLADE on X_train
  3. Transform to binary
  4. Save X_bin to /tmp/glade_benchmark/X_*.txt
  5. Call Julia train_tm.jl (single-thread)
  6. Load tm_rules.json + tm_metrics.json
  7. Save GLADE + TM model + report per dataset folder

Timings reported:
  - binarize_fit_time:  GLADE fit+transform (train + test)
  - tm_fit_time:        Julia TM training only
  - tm_predict_time:    TM batch predict
  - total_train_time:   binarize + tm_fit
"""

import os
import json
import time
import subprocess
import numpy as np

from .config import (
    DATASETS,
    GLADE_N_BINS,
    TM_PARAMS_PER_DATASET,
    JULIA_THREADS,
    RESULTS_DIR,
    TMP_DIR,
    TM_JULIA_SRC,
)
from .data_loader import load_and_preprocess
from .booleanizers.glade import GLADEBooleanizer as GLADE
from .utils import write_report, dump_json, save_pickle, timer


def save_bin_for_julia(X_tr_b, X_te_b, y_tr, y_te, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, "X_train.txt"), X_tr_b, fmt="%d")
    np.savetxt(os.path.join(out_dir, "X_test.txt"),  X_te_b, fmt="%d")
    np.savetxt(os.path.join(out_dir, "Y_train.txt"), y_tr, fmt="%d")
    np.savetxt(os.path.join(out_dir, "Y_test.txt"),  y_te, fmt="%d")


def run_tm_julia(train_script, tm_params):
    env = os.environ.copy()
    env.update({
        "TM_CLAUSES":  str(tm_params["CLAUSES"]),
        "TM_T":        str(tm_params["T"]),
        "TM_S":        str(tm_params["S"]),
        "TM_L":        str(tm_params["L"]),
        "TM_LF":       str(tm_params["LF"]),
        "TM_EPOCHS":   str(tm_params["EPOCHS"]),
        "TM_STATES":   str(tm_params["STATES_NUM"]),
        "TM_INCLUDE":  str(tm_params["INCLUDE_LIMIT"]),
        "TMP_DIR":     TMP_DIR,
    })
    t0 = time.perf_counter()
    result = subprocess.run(
        ["julia", f"--threads={JULIA_THREADS}", train_script],
        capture_output=True, text=True, timeout=3600, env=env,
    )
    wall = time.perf_counter() - t0
    if result.returncode != 0:
        print("Julia stderr:", result.stderr[-2000:])
    return result, wall


def process_dataset(loader_name, short_id, human_name):
    tm_params = TM_PARAMS_PER_DATASET[short_id]
    print(f"\n{'=' * 70}")
    print(f"  TM + GLADE — {human_name}")
    print(f"  TM cfg: C={tm_params['CLAUSES']}  T={tm_params['T']}  "
          f"S={tm_params['S']}  L={tm_params['L']}  "
          f"LF={tm_params['LF']}  E={tm_params['EPOCHS']}")
    print(f"{'=' * 70}")

    data = load_and_preprocess(loader_name)
    Xtr, Xte = data["X_train"], data["X_test"]
    ytr, yte = data["y_train"], data["y_test"]
    class_names = data["class_names"]

    print(f"  Train: {Xtr.shape[0]}, Test: {Xte.shape[0]}, Features: {Xtr.shape[1]}, Classes: {len(class_names)}")

    # ── Step 1: GLADE binarize (separately timed) ──
    with timer() as t_bin:
        glade = GLADE(n_bins=GLADE_N_BINS)
        glade.fit(Xtr)
        Xtr_b = glade.transform(Xtr)
        Xte_b = glade.transform(Xte)
    print(f"  Binarization: {t_bin.elapsed:.3f}s, {Xtr_b.shape[1]} bits")

    # Save binarized data + GLADE model
    out_dir = os.path.join(RESULTS_DIR, short_id)
    os.makedirs(out_dir, exist_ok=True)
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    glade_path = os.path.join(models_dir, "glade.json")
    glade.save_json(glade_path)
    glade_size_kb = os.path.getsize(glade_path) / 1024.0

    save_bin_for_julia(Xtr_b, Xte_b, ytr, yte, TMP_DIR)

    # ── Step 2: Train TM via Julia (single-thread) ──
    train_script = os.path.join(os.path.dirname(__file__), "train_tm.jl")
    result, wall = run_tm_julia(train_script, tm_params)
    if result.returncode != 0:
        print(f"  TM training FAILED for {short_id}")
        return None

    # ── Load TM results ──
    with open(os.path.join(TMP_DIR, "tm_rules.json")) as f:
        rules = json.load(f)
    with open(os.path.join(TMP_DIR, "tm_metrics.json")) as f:
        tm_metrics = json.load(f)

    # Save TM rules — JSON (canonical), .pkl (uncompressed pickle),
    # and .zst (zstd-1 compressed pickle) via save_pickle().
    tm_rules_path = os.path.join(models_dir, "tm_rules.json")
    with open(tm_rules_path, "w") as f:
        json.dump(rules, f)
    tm_pkl_path = os.path.join(models_dir, "tm_rules.pkl")
    save_pickle(rules, tm_pkl_path)
    tm_size_kb     = os.path.getsize(tm_rules_path) / 1024.0
    tm_pkl_kb      = os.path.getsize(tm_pkl_path) / 1024.0
    tm_zst_path    = os.path.splitext(tm_pkl_path)[0] + ".zst"
    tm_pkl_zst_kb  = os.path.getsize(tm_zst_path) / 1024.0

    # Compact size estimate (same method as FuzzyPatternTM/examples/glade):
    # TM bitmasks + GLADE thresholds/indices (lossless, no quantization).
    n_bits = Xtr_b.shape[1]
    n_classes = len(class_names)
    tm_bytes = n_classes * tm_params["CLAUSES"] * n_bits * 2 / 8
    glade_bytes = n_bits * 12
    compact_kb = (tm_bytes + glade_bytes) / 1024.0

    # ── Build classification metrics ──
    # Julia returned per-class keyed by class_id integer; align with class_names
    per_class_julia = tm_metrics["per_class"]
    per_class = []
    for i, name in enumerate(class_names):
        pcj = per_class_julia.get(str(i), per_class_julia.get(i, {}))
        per_class.append({
            "class": name,
            "precision": float(pcj.get("precision", 0)),
            "recall": float(pcj.get("recall", 0)),
            "f1": float(pcj.get("f1", 0)),
            "support": int(pcj.get("support", 0)),
        })

    metrics = {
        "accuracy": float(tm_metrics["accuracy"]),
        "macro_f1": float(tm_metrics["macro_f1"]),
        "weighted_f1": float(np.average(
            [pc["f1"] for pc in per_class],
            weights=[pc["support"] for pc in per_class],
        )),
        "per_class": per_class,
    }

    timings = {
        "binarize_fit_time": float(t_bin.elapsed),
        "tm_fit_time": float(tm_metrics["timings"]["fit"]),
        "tm_predict_time": float(tm_metrics["timings"]["predict"]),
        "tm_predict_us_per_sample": float(tm_metrics["timings"]["per_sample_predict_us"]),
        "total_train_time": float(t_bin.elapsed + tm_metrics["timings"]["fit"]),
    }

    extra = {
        "n_bits": Xtr_b.shape[1],
        "glade_size_kb": f"{glade_size_kb:.2f}",
        "tm_rules_json_kb":    f"{tm_size_kb:.2f}",
        "tm_rules_pkl_kb":     f"{tm_pkl_kb:.2f}",
        "tm_rules_pkl_zst_kb": f"{tm_pkl_zst_kb:.2f}",
        "tm_rules_zst_kb":     f"{tm_pkl_zst_kb:.2f}",
        "total_size_kb": f"{glade_size_kb + tm_size_kb:.2f}",
        "compact_size_kb": f"{compact_kb:.2f}",
        "hyperparameters": json.dumps(tm_params),
    }

    reports_dir = os.path.join(out_dir, "reports")
    write_report(reports_dir, "GLADE_FPTM", metrics, timings,
                 compact_kb, extra)

    print(f"  Acc={metrics['accuracy']*100:.2f}% MacroF1={metrics['macro_f1']:.4f} "
          f"bin={t_bin.elapsed:.2f}s tm={timings['tm_fit_time']:.1f}s")
    print(f"  Saved: {reports_dir}/GLADE_FPTM.txt")
    return metrics


def main():
    for loader_name, short_id, human_name in DATASETS:
        try:
            process_dataset(loader_name, short_id, human_name)
        except Exception as e:
            print(f"ERROR {short_id}: {e}")
            import traceback; traceback.print_exc()


def run_for_dataset(short_id):
    for loader_name, sid, human_name in DATASETS:
        if sid == short_id:
            return process_dataset(loader_name, sid, human_name)
    raise ValueError(f"Unknown dataset id: {short_id}")


if __name__ == "__main__":
    main()
