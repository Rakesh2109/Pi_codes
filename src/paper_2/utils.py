#!/usr/bin/env python3
"""
Utilities: timing, model saving, reports.
"""

import os
import time
import json
import pickle
import numpy as np
import zstandard as zstd
from contextlib import contextmanager
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    precision_recall_fscore_support,
)


@contextmanager
def timer():
    """Return elapsed time via .elapsed after context exit."""
    class T:
        def __init__(self): self.elapsed = 0.0
    t = T()
    t0 = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed = time.perf_counter() - t0


def serialized_size_kb(obj):
    """Measure model size by pickling to bytes."""
    try:
        return len(pickle.dumps(obj)) / 1024.0
    except Exception:
        return -1.0


def metrics_dict(y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    per_class = []
    for i, name in enumerate(class_names):
        per_class.append({
            "class": name,
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        })
    return {
        "accuracy": float(acc),
        "macro_f1": float(mf1),
        "weighted_f1": float(wf1),
        "per_class": per_class,
    }


def write_report(outdir, model_name, metrics, timings, size_kb, extra_info=None):
    """Write a .txt + .json report for one model on one dataset."""
    os.makedirs(outdir, exist_ok=True)
    txt_path = os.path.join(outdir, f"{model_name}.txt")
    json_path = os.path.join(outdir, f"{model_name}.json")

    with open(txt_path, "w") as f:
        f.write(f"{'='*75}\n")
        f.write(f"  MODEL: {model_name}\n")
        f.write(f"{'='*75}\n\n")
        f.write(f"Accuracy:         {metrics['accuracy']*100:.2f}%\n")
        f.write(f"Macro F1:         {metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1:      {metrics['weighted_f1']:.4f}\n\n")
        f.write(f"Timing:\n")
        for k, v in timings.items():
            f.write(f"  {k:<22} {v:.4f}s\n")
        f.write(f"\nModel size:       {size_kb:.2f} KB\n")
        if extra_info:
            f.write(f"\nExtra:\n")
            for k, v in extra_info.items():
                f.write(f"  {k}: {v}\n")
        f.write(f"\n{'='*75}\n")
        f.write(f"  PER-CLASS REPORT\n")
        f.write(f"{'='*75}\n")
        f.write(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}\n")
        f.write("-" * 70 + "\n")
        for pc in metrics["per_class"]:
            f.write(f"{pc['class']:<25} {pc['precision']:>10.4f} {pc['recall']:>10.4f} "
                    f"{pc['f1']:>10.4f} {pc['support']:>10}\n")

    payload = {
        "model": model_name,
        "metrics": metrics,
        "timings": timings,
        "size_kb": size_kb,
        "extra": extra_info or {},
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)


ZSTD_LEVEL = 1  # zstd level 1 — fast encode/decode for TM artefacts


def _zst_path_for(path):
    if path.endswith(".pkl"):
        return path[:-4] + ".zst"
    return path + ".zst"


def save_pickle(obj, path, compress=True):
    """Save a Python object as .pkl with optional .zst compression."""
    blob = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path, "wb") as f:
        f.write(blob)
    if compress:
        zst_path = _zst_path_for(path)
        with open(zst_path, "wb") as f:
            f.write(zstd.ZstdCompressor(level=ZSTD_LEVEL).compress(blob))


def load_pickle(path):
    """Load pickle from .pkl or .zst transparently."""
    with open(path, "rb") as f:
        blob = f.read()
    if path.endswith(".zst"):
        blob = zstd.ZstdDecompressor().decompress(blob)
    return pickle.loads(blob)


def dump_json(obj, path):
    """Save to JSON with numpy-safe default."""
    def _default(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.ndarray,)): return o.tolist()
        raise TypeError(f"Not serializable: {type(o)}")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)
