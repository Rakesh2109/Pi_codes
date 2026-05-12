#!/usr/bin/env python3
"""End-to-end inference benchmark across all baseline models + GLADE+FPTM.

For each dataset (WUSTL, NSL-KDD, TON_IoT, MedSec) the bundled
testset (``testset.npz``: ``X_te``, ``Y_te``) is run through:

  - Every saved sklearn / xgboost ``.pkl`` baseline (XGBoost, RandomForest,
    MLP_{tiny, small, med, C, 2C}, LogisticRegression, LinearSVM, kNN_5,
    GaussianNB, DecisionTree)
  - The GLADE+FPTM pipeline loaded from the bundled ``model.fbz`` using
    ``fuzzy_tm_infer.algorithms.TMModel.from_fbz``

A single ``benchmark_one(name, predict_fn)`` measures per-sample latency
the same way for every model — no per-model code paths.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Make fuzzy_tm_infer importable
sys.path.insert(0, "/IoT/GLADE_Enhanced_TM/src")
from fuzzy_tm_infer.algorithms import TMModel
from fuzzy_tm_infer.assets import ensure_assets, ensure_ml_models
from fuzzy_tm_infer.config import ASSETS_DIR, ML_MODELS_DIR

DATASETS = [
    ("wustl",   "WUSTL-EHMS-2020"),
    ("nslkdd",  "NSL-KDD"),
    ("toniot",  "TON_IoT"),
    ("medsec",  "MedSec-25"),
]

ML_MODELS = [
    "XGBoost",
    "MLP_med", "MLP_small", "MLP_tiny", "MLP_C", "MLP_2C",
    "LogisticRegression", "LinearSVM",
    "DecisionTree", "GaussianNB",
]
# kNN_5 and RandomForest are excluded from this latency benchmark:
# both take 10²–10⁴ ms per call on these datasets due to their size,
# which is already reflected in the size-based comparisons. Their
# accuracy/F1 are reported in Tables tab:results_part1, _part2.

WARMUP = 3
REPEATS = 5


def benchmark_one(predict_fn: Callable[[np.ndarray], np.ndarray],
                   X: np.ndarray, y: np.ndarray, repeats: int = REPEATS):
    """Time a predict callable end-to-end. Returns (us/sample, acc, macroF1)."""
    for _ in range(WARMUP):
        predict_fn(X[:128])
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        yhat = predict_fn(X)
        runs.append((time.perf_counter() - t0) * 1e6 / X.shape[0])
    yhat = predict_fn(X)
    return float(np.median(runs)), accuracy_score(y, yhat), \
           f1_score(y, yhat, average="macro", zero_division=0)


def _load_baseline(stem: str, model_name: str):
    path = ML_MODELS_DIR / stem / f"{model_name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        clf = pickle.load(f)
    return clf, path.stat().st_size


def _load_scaler(stem: str):
    p = ML_MODELS_DIR / stem / "scaler.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def main():
    ensure_assets()
    ensure_ml_models()

    rows = []
    for stem, human in DATASETS:
        npz = np.load(ML_MODELS_DIR / stem / "testset.npz")
        X_te, y_te = npz["X_te"], npz["Y_te"]
        scaler = _load_scaler(stem)

        print(f"\n{'='*78}\n  {human}  (test n={X_te.shape[0]}, "
              f"raw features={X_te.shape[1]})\n{'='*78}")
        print(f"  {'Model':<22} {'size KB':>10} {'us/sample':>10} "
              f"{'Acc':>8} {'MacroF1':>9}")
        print(f"  {'-' * 64}")

        # Baselines
        for name in ML_MODELS:
            res = _load_baseline(stem, name)
            if res is None:
                continue
            clf, size_b = res
            X_in = scaler.transform(X_te) if scaler is not None else X_te
            us, acc, mf1 = benchmark_one(clf.predict, X_in, y_te)
            rows.append({"dataset": stem, "human": human, "model": name,
                         "size_kb": size_b / 1024.0,
                         "us_per_sample": us, "accuracy": acc, "macro_f1": mf1})
            print(f"  {name:<22} {size_b/1024:>10.1f} {us:>10.3f} "
                  f"{acc:>8.4f} {mf1:>9.4f}")

        # GLADE+FPTM via fuzzy_tm_infer (use predict_batch — vectorised)
        fbz_path = ASSETS_DIR / "tm_models" / f"{stem}_model.fbz"
        tm = TMModel.from_fbz(fbz_path)
        Xf = X_te.astype(np.float32, copy=False)
        us, acc, mf1 = benchmark_one(tm.predict_batch, Xf, y_te)
        size_b = fbz_path.stat().st_size
        rows.append({"dataset": stem, "human": human, "model": "GLADE+FPTM",
                     "size_kb": size_b / 1024.0,
                     "us_per_sample": us, "accuracy": acc, "macro_f1": mf1})
        print(f"  {'GLADE+FPTM (FBZ)':<22} {size_b/1024:>10.1f} {us:>10.3f} "
              f"{acc:>8.4f} {mf1:>9.4f}")

    out = Path("/IoT/Paper_2/results/SUMMARY_inference_all.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
