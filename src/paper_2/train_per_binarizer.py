#!/usr/bin/env python3
"""Train a TM per dataset using a configurable binarizer and bundle the
result into a single inference pickle, mirroring what `run_tm.py` does
for GLADE.

For each (dataset, binarizer) it produces in results/<dataset>/models/:
    <bn>_tm_rules.json    Julia TM clause dump (binarizer-specific)
    <bn>_tm_rules.pkl     same, pickled
    model_<bn>.pkl        bundle: {'binarizer', 'tm_rules', 'meta'}

Where <bn> is one of {kbins, standard}.
The GLADE bundle (model.pkl) is unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .booleanizers.kbins import KBinsBooleanizer
from .booleanizers.standard import StandardBinarizer
from .config import (
    DATASETS, GLADE_N_BINS, JULIA_THREADS,
    RESULTS_DIR, TMP_DIR, TM_PARAMS_PER_DATASET,
)
from .data_loader import load_and_preprocess
from .run_tm import run_tm_julia, save_bin_for_julia


# Single source of truth for the binarizer factories; identical pattern to
# the codec table in bench_codecs.py — no per-binarizer branching below.
BINARIZERS: dict[str, Callable[[], object]] = {
    "kbins":    lambda: KBinsBooleanizer(n_bins=GLADE_N_BINS,
                                         strategy="quantile"),
    "standard": lambda: StandardBinarizer(max_bits_per_feature=GLADE_N_BINS),
}


def _bits_of(binarizer, X: np.ndarray) -> np.ndarray:
    """Run binarizer.transform(X), returning a (n, n_bits) uint8 matrix."""
    out = binarizer.transform(X)
    return out.astype(np.uint8, copy=False)


def _train_one(loader_name: str, short_id: str, human: str,
               binarizer_name: str) -> dict:
    tm_params = TM_PARAMS_PER_DATASET[short_id]
    print(f"\n{'=' * 70}")
    print(f"  TM + {binarizer_name.upper()} — {human}")
    print(f"  TM cfg: C={tm_params['CLAUSES']}  T={tm_params['T']}  "
          f"S={tm_params['S']}  L={tm_params['L']}  "
          f"LF={tm_params['LF']}  E={tm_params['EPOCHS']}")
    print(f"{'=' * 70}")

    data = load_and_preprocess(loader_name)
    Xtr, Xte = data["X_train"], data["X_test"]
    ytr, yte = data["y_train"], data["y_test"]

    # Fit binarizer + transform splits
    t0 = time.perf_counter()
    bin_ = BINARIZERS[binarizer_name]()
    bin_.fit(Xtr)
    Xtr_b = _bits_of(bin_, Xtr)
    Xte_b = _bits_of(bin_, Xte)
    bin_time = time.perf_counter() - t0
    print(f"  Binarization: {bin_time:.3f}s, {Xtr_b.shape[1]} bits")

    # Hand off to Julia
    save_bin_for_julia(Xtr_b, Xte_b, ytr, yte, TMP_DIR)
    train_script = os.path.join(os.path.dirname(__file__), "train_tm.jl")
    result, wall = run_tm_julia(train_script, tm_params)
    if result.returncode != 0:
        print(f"  TM training FAILED for {short_id}")
        print(result.stderr[-2000:])
        return None

    rules = json.load(open(os.path.join(TMP_DIR, "tm_rules.json")))
    metrics = json.load(open(os.path.join(TMP_DIR, "tm_metrics.json")))

    # Persist outputs
    models_dir = Path(RESULTS_DIR) / short_id / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    rules_json = models_dir / f"{binarizer_name}_tm_rules.json"
    rules_pkl  = models_dir / f"{binarizer_name}_tm_rules.pkl"
    bundle_pkl = models_dir / f"model_{binarizer_name}.pkl"

    rules_json.write_text(json.dumps(rules))
    with rules_pkl.open("wb") as f:
        pickle.dump(rules, f, protocol=pickle.HIGHEST_PROTOCOL)

    bundle = {
        "binarizer": bin_,                # ready-to-use, has .transform()
        "tm_rules": rules,                # canonical TM clause dump
        "meta": {
            "binarizer_name": binarizer_name,
            "dataset": short_id,
            "n_bits": int(Xtr_b.shape[1]),
            "n_classes": int(rules["n_classes"]),
            "hyperparameters": tm_params,
            "macro_f1": float(metrics["macro_f1"]),
            "accuracy": float(metrics["accuracy"]),
        },
    }
    with bundle_pkl.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  Acc={metrics['accuracy']*100:.2f}% "
          f"MacroF1={metrics['macro_f1']:.4f} "
          f"bin={bin_time:.2f}s tm={wall:.1f}s")
    print(f"  → {bundle_pkl} ({bundle_pkl.stat().st_size/1024:.2f} KB)")

    return {
        "dataset": short_id, "binarizer": binarizer_name,
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "n_bits": int(Xtr_b.shape[1]),
        "bundle_kb": bundle_pkl.stat().st_size / 1024.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binarizer", choices=list(BINARIZERS.keys()),
                    default=None,
                    help="Train one binarizer; default: train all.")
    ap.add_argument("--dataset", default=None,
                    help="Restrict to a specific dataset short id.")
    args = ap.parse_args()

    bin_names = [args.binarizer] if args.binarizer else list(BINARIZERS.keys())
    rows = []
    for bn in bin_names:
        for loader, sid, human in DATASETS:
            if args.dataset and sid != args.dataset:
                continue
            try:
                row = _train_one(loader, sid, human, bn)
                if row:
                    rows.append(row)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"ERROR {sid}/{bn}: {e}")

    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    print(f"  {'Dataset':<10} {'Binarizer':<10} {'Acc':>8} {'MF1':>8} "
          f"{'bits':>5} {'bundle KB':>10}")
    for r in rows:
        print(f"  {r['dataset']:<10} {r['binarizer']:<10} "
              f"{r['accuracy']:>8.4f} {r['macro_f1']:>8.4f} "
              f"{r['n_bits']:>5d} {r['bundle_kb']:>10.2f}")


if __name__ == "__main__":
    main()
