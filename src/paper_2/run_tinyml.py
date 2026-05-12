#!/usr/bin/env python3
"""
Train TinyML models (MLPs of varying size), single-core.

Same pipeline as run_ml.py but for neural networks.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from .config import DATASETS, TINYML_MODELS, RESULTS_DIR
from .data_loader import load_and_preprocess
from .utils import timer, write_report, metrics_dict, save_pickle, serialized_size_kb

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def process_dataset(loader_name, short_id, human_name, only_models=None):
    print(f"\n{'=' * 70}")
    print(f"  TinyML — {human_name}")
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

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    models = TINYML_MODELS
    if only_models:
        wanted = {m.lower() for m in only_models}
        models = [m for m in TINYML_MODELS if m[0].lower() in wanted]

    for (name, kind, params) in models:
        try:
            model = MLPClassifier(**params, random_state=42)

            with timer() as t_fit:
                model.fit(Xtr_s, ytr)
            with timer() as t_pred:
                y_pred = model.predict(Xte_s)

            m = metrics_dict(yte, y_pred, class_names)

            model_path = os.path.join(models_dir, f"{name}.pkl")
            save_pickle({"model": model, "scaler": scaler,
                         "class_names": class_names}, model_path, compress=False)
            bundle_size_kb = os.path.getsize(model_path) / 1024.0
            model_size_kb = serialized_size_kb(model)

            # Count parameters for MCU footprint
            n_params = sum(w.size for w in model.coefs_) + \
                       sum(b.size for b in model.intercepts_)

            timings = {
                "fit_time": float(t_fit.elapsed),
                "predict_time": float(t_pred.elapsed),
                "per_sample_predict_us": float(t_pred.elapsed / len(yte) * 1e6),
                "total_train_time": float(t_fit.elapsed),
            }
            extra = {
                "kind": kind,
                "params": str(params),
                "n_neural_params": n_params,
                "param_size_kb_float32": f"{n_params * 4 / 1024:.2f}",
                "param_size_kb_int8": f"{n_params / 1024:.2f}",
                "bundle_pickle_kb": f"{bundle_size_kb:.2f}",
                "model_pickle_kb": f"{model_size_kb:.2f}",
            }
            write_report(reports_dir, name, m, timings, model_size_kb, extra)

            print(f"  {name:<12} Acc={m['accuracy']*100:>6.2f}%  "
                  f"F1={m['macro_f1']:.4f}  params={n_params}  "
                  f"fit={t_fit.elapsed:.1f}s  size={model_size_kb:.1f}KB")
        except Exception as e:
            print(f"  {name:<12} FAILED: {e}")


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
