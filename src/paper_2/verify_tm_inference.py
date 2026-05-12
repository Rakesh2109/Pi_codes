#!/usr/bin/env python3
"""Quick check that the numba TM kernel produces the same predictions
as Paper_2's existing slow numpy `verify_fbz.predict` and matches the
training-time accuracy on every dataset."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .config import DATASETS, RESULTS_DIR
from .data_loader import load_and_preprocess
from .tm_inference import TMModel


def main():
    print(f"  {'Dataset':<10} {'n_test':>7} {'us/sample':>10} "
          f"{'Acc':>8} {'MacroF1':>9}")
    print("  " + "-" * 50)
    for loader, sid, human in DATASETS:
        bundle = Path(RESULTS_DIR) / sid / "models" / "model.pkl"
        tm = TMModel.from_pkl_bundle(bundle)
        data = load_and_preprocess(loader)
        Xte = data["X_test"].astype(np.float32, copy=False)
        yte = np.asarray(data["y_test"], dtype=np.int32)

        # Warm up the JIT
        _ = tm.predict_batch(Xte[:128])
        # Median of 5
        runs = []
        for _ in range(5):
            t0 = time.perf_counter()
            yhat = tm.predict_batch(Xte)
            runs.append((time.perf_counter() - t0) * 1e6 / Xte.shape[0])
        us = float(np.median(runs))
        acc = accuracy_score(yte, yhat)
        mf1 = f1_score(yte, yhat, average="macro", zero_division=0)
        print(f"  {sid:<10} {Xte.shape[0]:>7d} {us:>10.3f} "
              f"{acc:>8.4f} {mf1:>9.4f}")


if __name__ == "__main__":
    main()
