#!/usr/bin/env python3
"""
Train classic ML models (trees, linear, k-NN, NB) single-core.

For each (dataset, model) pair:
  - Load + preprocess (same as run_tm.py)
  - Fit model (single-core, via env vars + n_jobs=1)
  - Predict on test
  - Save model to models/<name>.pkl
  - Write reports/<name>.txt with metrics + timings + size
"""

import os
from typing import Iterable, Optional

from .config import DATASETS, ML_TREE_MODELS, ML_OTHER_MODELS, RESULTS_DIR
from .data_loader import load_and_preprocess
from .utils import timer, write_report, metrics_dict, save_pickle, serialized_size_kb

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


MODEL_FACTORY = {
    "xgboost": lambda p: XGBClassifier(
        **p, n_jobs=1, random_state=42, eval_metric="mlogloss", verbosity=0
    ),
    "rf":      lambda p: RandomForestClassifier(**p, n_jobs=1, random_state=42),
    "logreg":  lambda p: LogisticRegression(**p, random_state=42),
    "svm":     lambda p: LinearSVC(**p, random_state=42),
    "dt":      lambda p: DecisionTreeClassifier(**p, random_state=42),
    "nb":      lambda p: GaussianNB(**p),
    "knn":     lambda p: KNeighborsClassifier(**p, n_jobs=1),
}

NEEDS_SCALING = {"logreg", "svm", "nb", "knn"}


def _filter_models(models, only_names: Optional[Iterable[str]]):
    if not only_names:
        return models
    wanted = {m.lower() for m in only_names}
    return [m for m in models if m[0].lower() in wanted]


def process_dataset(loader_name, short_id, human_name, only_models=None):
    print(f"\n{'=' * 70}")
    print(f"  ML Models — {human_name}")
    print(f"{'=' * 70}")

    data = load_and_preprocess(loader_name)
    Xtr, Xte = data["X_train"], data["X_test"]
    ytr, yte = data["y_train"], data["y_test"]
    class_names = data["class_names"]

    out_dir = os.path.join(RESULTS_DIR, short_id)
    reports_dir = os.path.join(out_dir, "reports")
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    all_models = _filter_models(ML_TREE_MODELS + ML_OTHER_MODELS, only_models)
    for (name, kind, params) in all_models:
        try:
            # Build model
            model = MODEL_FACTORY[kind](params)
            scaler = None
            if kind in NEEDS_SCALING:
                scaler = StandardScaler()
                Xtr_s = scaler.fit_transform(Xtr)
                Xte_s = scaler.transform(Xte)
            else:
                Xtr_s, Xte_s = Xtr, Xte

            # Fit (timed)
            with timer() as t_fit:
                model.fit(Xtr_s, ytr)
            # Predict (timed)
            with timer() as t_pred:
                y_pred = model.predict(Xte_s)

            m = metrics_dict(yte, y_pred, class_names)

            # Save model + scaler for MCU/Pi inference
            model_path = os.path.join(models_dir, f"{name}.pkl")
            save_pickle({"model": model, "scaler": scaler,
                         "class_names": class_names}, model_path, compress=False)
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
                "params": str(params),
                "scaling": "yes" if scaler else "no",
                "bundle_pickle_kb": f"{bundle_size_kb:.2f}",
                "model_pickle_kb": f"{model_size_kb:.2f}",
            }
            write_report(reports_dir, name, m, timings, model_size_kb, extra)

            print(f"  {name:<22} Acc={m['accuracy']*100:>6.2f}%  "
                  f"F1={m['macro_f1']:.4f}  fit={t_fit.elapsed:.2f}s  size={model_size_kb:.1f}KB")
        except Exception as e:
            print(f"  {name:<22} FAILED: {e}")


def run_for_dataset(short_id, only_models=None):
    for loader_name, sid, human_name in DATASETS:
        if sid == short_id:
            process_dataset(loader_name, sid, human_name, only_models)
            return
    raise ValueError(f"Unknown dataset id: {short_id}")


def main():
    for loader_name, short_id, human_name in DATASETS:
        process_dataset(loader_name, short_id, human_name)


if __name__ == "__main__":
    main()
