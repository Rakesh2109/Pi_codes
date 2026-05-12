#!/usr/bin/env python3
"""Single-file Pi 5 inference benchmark for the Paper_2 IDS pipeline.

Runs every model on every dataset and prints / saves
per-(dataset, model) timing and accuracy. Designed to be self-contained:
needs only numpy, scikit-learn, xgboost, numba, zstandard installed.

Usage:
    python3 bench_pi.py              # everything
    python3 bench_pi.py --skip-slow  # drop kNN_5 + RandomForest

Layout next to this script:
    tm_inference.py
    assets/<dataset>/
        testset.npz            X_te, Y_te
        scaler.pkl
        model.pkl              GLADE+FPTM bundle
        model_kbins.pkl        KBins+FPTM bundle
        model_standard.pkl     Standard+FPTM bundle
        XGBoost.pkl  RandomForest.pkl  kNN_5.pkl
        MLP_med.pkl  MLP_small.pkl  MLP_tiny.pkl
        DecisionTree.pkl  GaussianNB.pkl
        LinearSVM.pkl  LogisticRegression.pkl
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tm_inference import TMModel, build_fbz_from_dicts  # noqa: E402
from booleanizers.glade import GLADEBooleanizer  # noqa: E402
from booleanizers.kbins import KBinsBooleanizer  # noqa: E402
from booleanizers.standard import StandardBinarizer  # noqa: E402

BINARIZER_CLASS = {
    "GLADE":    GLADEBooleanizer,
    "KBins":    KBinsBooleanizer,
    "Standard": StandardBinarizer,
}

DATASETS = [
    ("wustl",   "WUSTL-EHMS-2020"),
    ("nslkdd",  "NSL-KDD"),
    ("ton_iot", "TON-IoT"),
    ("medsec",  "MedSec-25"),
]

ML_MODELS = [
    "XGBoost", "RandomForest", "kNN_5",
    "MLP_med", "MLP_small", "MLP_tiny",
    "DecisionTree", "GaussianNB",
    "LinearSVM", "LogisticRegression",
]

TM_BUNDLES = [
    ("GLADE+FPTM",     "model.pkl"),
    ("KBins+FPTM",     "model_kbins.pkl"),
    ("Standard+FPTM",  "model_standard.pkl"),
]

WARMUP = 3
REPEATS = 5


def time_predict(predict_fn, X, repeats=REPEATS):
    for _ in range(WARMUP):
        predict_fn(X[:128])
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        yhat = predict_fn(X)
        runs.append((time.perf_counter() - t0) * 1e6 / X.shape[0])
    return float(np.median(runs)), yhat


def macro_f1(y_true, y_pred):
    classes = np.unique(y_true)
    f1s = []
    for c in classes:
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        if tp == 0 and fp == 0 and fn == 0:
            f1s.append(0.0); continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * precision * recall / (precision + recall)
                   if (precision + recall) else 0.0)
    return float(np.mean(f1s))


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def bench_dataset(stem: str, human: str, results: list):
    ds_dir = HERE / "assets" / stem
    if not (ds_dir / "testset.npz").exists():
        print(f"  [skip] no testset for {stem}")
        return

    npz = np.load(ds_dir / "testset.npz")
    X_te, y_te = npz["X_te"], npz["Y_te"]
    scaler = None
    if (ds_dir / "scaler.pkl").exists():
        with open(ds_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

    print(f"\n=== {human}  (n_test = {X_te.shape[0]}, "
          f"raw_features = {X_te.shape[1]}) ===")
    print(f"  {'Model':<20} {'us/sample':>10} {'Acc':>8} {'MacroF1':>9}")
    print(f"  {'-' * 50}")

    # ---- ML baselines ----
    for name in ML_MODELS:
        path = ds_dir / f"{name}.pkl"
        if not path.exists():
            continue
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            # Saved pkls are bundles {model, scaler, class_names}
            if isinstance(obj, dict):
                clf = obj.get("model") or obj.get("clf")
                local_scaler = obj.get("scaler", scaler)
            else:
                clf = obj
                local_scaler = scaler
            X_in = (local_scaler.transform(X_te)
                    if local_scaler is not None else X_te)
            us, yhat = time_predict(clf.predict, X_in)
            acc = accuracy(y_te, yhat)
            mf1 = macro_f1(y_te, yhat)
            results.append({"dataset": stem, "model": name,
                            "us_per_sample": us,
                            "accuracy": acc, "macro_f1": mf1})
            print(f"  {name:<20} {us:>10.3f} {acc:>8.4f} {mf1:>9.4f}")
        except Exception as e:
            print(f"  {name:<20} FAILED: {e}")
            results.append({"dataset": stem, "model": name, "error": str(e)})

    # ---- TM bundles via the numba kernel ----
    Xf = np.ascontiguousarray(X_te, dtype=np.float32)
    for name, fn in TM_BUNDLES:
        path = ds_dir / fn
        if not path.exists():
            continue
        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            # Portable schema:
            #   {tm_rules, binarizer_kind, binarizer_state}
            kind = bundle.get("binarizer_kind", "GLADE")
            state = bundle["binarizer_state"]
            if kind == "GLADE":
                # The state dict can be fed straight into build_fbz_from_dicts;
                # however we need feat_idx + thresh as numpy arrays internally.
                glade_dict = state
            else:
                # Reconstruct binarizer from dict to match training-time
                # behavior, then read off (feat_idx, thresh) for the kernel.
                cls = BINARIZER_CLASS[kind]
                bin_ = cls.from_dict(state)
                # Kernel only uses feat_idx + thresh + n_bits;
                # build a glade-shaped dict.
                glade_dict = {
                    "n_features_in": int(bin_._n_features),
                    "feat_idx": _binarizer_feat_idx(bin_),
                    "thresh":   _binarizer_thresh(bin_),
                    "n_bits":   bundle["tm_rules"]["n_bits"],
                }
            tm = TMModel(build_fbz_from_dicts(bundle["tm_rules"], glade_dict))
            us, yhat = time_predict(tm.predict_batch, Xf)
            acc = accuracy(y_te, yhat)
            mf1 = macro_f1(y_te, yhat)
            results.append({"dataset": stem, "model": name,
                            "us_per_sample": us,
                            "accuracy": acc, "macro_f1": mf1})
            print(f"  {name:<20} {us:>10.3f} {acc:>8.4f} {mf1:>9.4f}")
        except Exception as e:
            print(f"  {name:<20} FAILED: {e}")
            results.append({"dataset": stem, "model": name, "error": str(e)})


def _binarizer_feat_idx(bin_):
    """Return the (feature_index per output bit) array for KBins / Standard.
    GLADE already exposes this directly; KBins / Standard build it from
    their bin_edges_ / unique_values_ tables."""
    if hasattr(bin_, "_feature_indices"):           # GLADE
        return bin_._feature_indices
    if hasattr(bin_, "bin_edges_"):                 # KBins
        idx = []
        for f, edges in enumerate(bin_.bin_edges_):
            idx.extend([f] * len(edges))
        return np.asarray(idx, np.int32)
    if hasattr(bin_, "unique_values"):              # Standard
        idx = []
        for f, vals in enumerate(bin_.unique_values):
            idx.extend([f] * len(vals))
        return np.asarray(idx, np.int32)
    raise ValueError("unknown binarizer layout")


def _binarizer_thresh(bin_):
    if hasattr(bin_, "thresholds"):                 # GLADE
        return bin_.thresholds.astype(np.float32, copy=False)
    if hasattr(bin_, "bin_edges_"):                 # KBins
        flat = []
        for edges in bin_.bin_edges_:
            flat.extend(list(edges))
        return np.asarray(flat, np.float32)
    if hasattr(bin_, "unique_values"):              # Standard
        flat = []
        for vals in bin_.unique_values:
            flat.extend(list(vals))
        return np.asarray(flat, np.float32)
    raise ValueError("unknown binarizer layout")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-slow", action="store_true",
                    help="skip kNN_5 and RandomForest (very slow on Pi)")
    args = ap.parse_args()
    if args.skip_slow:
        global ML_MODELS
        ML_MODELS = [m for m in ML_MODELS if m not in ("kNN_5", "RandomForest")]

    results: list = []
    for stem, human in DATASETS:
        try:
            bench_dataset(stem, human, results)
        except Exception as e:
            print(f"ERROR {stem}: {e}")

    out = HERE / "results_pi.json"
    with out.open("w") as f:
        json.dump({"results": results,
                   "platform": _platform_info()}, f, indent=2)
    print(f"\nSaved: {out}")
    print(f"Total rows: {len(results)}")


def _platform_info():
    import platform
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        for line in cpuinfo.splitlines():
            if "Model" in line or "model name" in line:
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    return info


if __name__ == "__main__":
    main()
