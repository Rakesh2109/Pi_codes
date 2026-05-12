#!/usr/bin/env python3
"""
Case 2: Train ML + TinyML baselines on RAW features, but with
hyperparameters derived from the per-dataset tuned FPTM configuration,
so that every baseline is forced to operate at a capacity comparable
to the TM model on that dataset.

Mapping (TM  ->  ML):
    C (clauses per class per polarity) ->  n_estimators (trees)
                                     ->  hidden_layer_sizes = (C,)  (MLP)
    L (literal cap per clause)         ->  max_depth = ceil(log2(L+1))
                                           for trees and DecisionTree
    E (training epochs)                ->  max_iter (MLP, LogReg, LinearSVM)
    T, s, L_f                          ->  no direct analogue (left default)

Each trained artefact is saved as
    results/<short_id>_tmmatched/models/<name>.pkl
and a matching metrics report is written to
    results/<short_id>_tmmatched/reports/<name>.json

Pickle layout (compatible with run_ml.py):
    {
      "model":       fitted sklearn/xgboost estimator,
      "scaler":      StandardScaler or None,
      "class_names": list[str],
      "tm_source":   dict with the TM hyperparameters this model was
                     matched to,
    }
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import math
import time
import numpy as np

from .config import DATASETS, RESULTS_DIR
from .data_loader import load_and_preprocess
from .utils import timer, write_report, metrics_dict, save_pickle, serialized_size_kb

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# Per-dataset tuned FPTM configs you chose earlier
TM_CONFIGS = {
    "wustl":   dict(C=60,  T=8,  s=300, L=50, Lf=15, E=200),
    "nslkdd":  dict(C=90,  T=12, s=200, L=40, Lf=8,  E=80),
    "ton_iot": dict(C=100, T=15, s=20,  L=25, Lf=8,  E=200),
    "medsec":  dict(C=80,  T=12, s=75,  L=30, Lf=8,  E=300),
}

NEEDS_SCALING = {"LogReg", "LinearSVM", "GaussianNB", "kNN_5",
                 "MLP_C", "MLP_2C"}


def _depth_from_L(L):
    return max(2, int(math.ceil(math.log2(L + 1))))


def build_model_suite(tm):
    """Returns list of (name, kind, sklearn_estimator) for a given TM cfg."""
    C  = tm["C"]
    L  = tm["L"]
    E  = tm["E"]
    depth = _depth_from_L(L)
    return [
        # Tree-ensembles: n_estimators = C, max_depth derived from L
        ("XGBoost_Cmatched",
         "xgboost",
         XGBClassifier(n_estimators=C, max_depth=depth,
                       learning_rate=0.1, n_jobs=1, random_state=42,
                       eval_metric="mlogloss", verbosity=0)),
        ("RandomForest_Cmatched",
         "rf",
         RandomForestClassifier(n_estimators=C, max_depth=depth,
                                n_jobs=1, random_state=42)),
        ("DecisionTree_Lmatched",
         "dt",
         DecisionTreeClassifier(max_depth=depth, random_state=42)),
        # Linear and distance-based baselines: max_iter = E where applicable
        ("LogReg",
         "logreg",
         LogisticRegression(C=1.0, max_iter=E,
                            random_state=42, solver="lbfgs")),
        ("LinearSVM",
         "svm",
         LinearSVC(C=1.0, max_iter=E, random_state=42)),
        ("GaussianNB",
         "nb",
         GaussianNB()),
        ("kNN_5",
         "knn",
         KNeighborsClassifier(n_neighbors=5, n_jobs=1)),
        # TinyML: one hidden layer of size C, max_iter = E
        ("MLP_C",
         "mlp",
         MLPClassifier(hidden_layer_sizes=(C,), max_iter=E,
                       random_state=42)),
        # TinyML: two-layer MLP mirroring the 2*C positive+negative
        # clause budget of the TM
        ("MLP_2C",
         "mlp",
         MLPClassifier(hidden_layer_sizes=(2*C, C), max_iter=E,
                       random_state=42)),
    ]


def process_dataset(loader_name, short_id, human_name):
    tm = TM_CONFIGS[short_id]
    depth = _depth_from_L(tm["L"])
    print(f"\n{'=' * 70}")
    print(f"  [Case 2] ML + TinyML on RAW, TM-matched — {human_name}")
    print(f"  TM cfg: C={tm['C']}  T={tm['T']}  s={tm['s']}  "
          f"L={tm['L']}  Lf={tm['Lf']}  E={tm['E']}")
    print(f"  -> n_estimators={tm['C']}  max_depth={depth}  "
          f"max_iter={tm['E']}  MLP hidden=({tm['C']},) and (2C,C)")
    print(f"{'=' * 70}")

    data = load_and_preprocess(loader_name)
    Xtr, Xte = data["X_train"], data["X_test"]
    ytr, yte = data["y_train"], data["y_test"]
    class_names = data["class_names"]

    out_dir = os.path.join(RESULTS_DIR, f"{short_id}_tmmatched")
    reports_dir = os.path.join(out_dir, "reports")
    models_dir  = os.path.join(out_dir, "models")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir,  exist_ok=True)

    suite = build_model_suite(tm)
    for (name, kind, model) in suite:
        try:
            scaler = None
            if name in NEEDS_SCALING:
                scaler = StandardScaler()
                Xtr_s = scaler.fit_transform(Xtr)
                Xte_s = scaler.transform(Xte)
            else:
                Xtr_s, Xte_s = Xtr, Xte

            with timer() as t_fit:
                model.fit(Xtr_s, ytr)
            with timer() as t_pred:
                y_pred = model.predict(Xte_s)
            m = metrics_dict(yte, y_pred, class_names)

            model_path = os.path.join(models_dir, f"{name}.pkl")
            save_pickle({
                "model": model,
                "scaler": scaler,
                "class_names": class_names,
                "tm_source": tm,
            }, model_path, compress=False)
            bundle_size_kb = os.path.getsize(model_path) / 1024.0
            model_size_kb = serialized_size_kb(model)

            timings = {
                "fit_time": float(t_fit.elapsed),
                "predict_time": float(t_pred.elapsed),
                "per_sample_predict_us": float(t_pred.elapsed / len(yte) * 1e6),
                "total_train_time": float(t_fit.elapsed),
            }
            extra = {
                "kind": kind,
                "feature_space": "raw",
                "tm_source": tm,
                "derived_hp": {
                    "n_estimators": tm["C"],
                    "max_depth": depth,
                    "max_iter": tm["E"],
                    "hidden_layer_sizes": f"({tm['C']},)" if name == "MLP_C"
                                          else f"({2*tm['C']}, {tm['C']})",
                },
                "scaling": "yes" if scaler else "no",
                "bundle_pickle_kb": f"{bundle_size_kb:.2f}",
                "model_pickle_kb": f"{model_size_kb:.2f}",
            }
            # Add MLP param count if applicable
            if hasattr(model, "coefs_"):
                n_params = sum(w.size for w in model.coefs_) + \
                           sum(b.size for b in model.intercepts_)
                extra["n_neural_params"] = int(n_params)

            write_report(reports_dir, name, m, timings, model_size_kb, extra)
            print(f"  {name:<22} Acc={m['accuracy']*100:>6.2f}%  "
                  f"F1={m['macro_f1']:.4f}  fit={t_fit.elapsed:.2f}s  "
                f"size={model_size_kb:.1f}KB")
        except Exception as e:
            print(f"  {name:<22} FAILED: {e}")


def _all_datasets():
    all_datasets = list(DATASETS)
    if not any(sid == "wustl" for _, sid, _ in all_datasets):
        all_datasets = [("load_wustl", "wustl", "WUSTL-EHMS-2020")] + all_datasets
    return all_datasets


def run_for_dataset(short_id):
    for loader_name, sid, human_name in _all_datasets():
        if sid == short_id:
            process_dataset(loader_name, sid, human_name)
            return
    raise ValueError(f"Unknown dataset id: {short_id}")


def main():
    for loader_name, short_id, human_name in _all_datasets():
        process_dataset(loader_name, short_id, human_name)


if __name__ == "__main__":
    main()
