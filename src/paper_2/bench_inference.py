#!/usr/bin/env python3
"""End-to-end inference latency for the GLADE+FPTM pipeline.

Loads ``model.fbz`` (zstd-1, GLADE + TM clauses bundled), then
measures per-sample wall-clock cost of:

    raw input  ──► GLADE binarise ──► TM vote ──► predicted label

on the host PC. The same script can be re-run on the Raspberry Pi 5
to fill the second column of Table~\\ref{tab:edge_inference}.

Reports mean and median per-sample latencies in microseconds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import fbz as fbz_mod
from .config import DATASETS, RESULTS_DIR
from .data_loader import load_and_preprocess
from .verify_fbz import predict


def _bench_dataset(loader_name: str, short_id: str, human: str) -> dict:
    fbz_path = Path(RESULTS_DIR) / short_id / "models" / "model.fbz"
    blob = fbz_path.read_bytes()
    data = load_and_preprocess(loader_name)
    Xte = data["X_test"].astype(np.float64, copy=False)
    n_samples = Xte.shape[0]

    # ── one-time costs (load + decompress) ──
    t0 = time.perf_counter()
    model = fbz_mod.unpack(blob)
    load_us = (time.perf_counter() - t0) * 1e6

    # ── per-sample latency over the full test split, batched ──
    # warm up
    _ = predict(model, fbz_mod.transform(model, Xte[:128]))
    t0 = time.perf_counter()
    Xb = fbz_mod.transform(model, Xte)
    yhat = predict(model, Xb)
    batch_total = time.perf_counter() - t0
    per_sample_us = batch_total * 1e6 / n_samples

    # repeat to get a tighter mean
    runs = []
    for _ in range(5):
        t0 = time.perf_counter()
        Xb = fbz_mod.transform(model, Xte)
        _ = predict(model, Xb)
        runs.append((time.perf_counter() - t0) * 1e6 / n_samples)
    mean_us = float(np.mean(runs))
    median_us = float(np.median(runs))

    return {
        "dataset": short_id, "human": human,
        "n_samples": n_samples,
        "load_us": load_us,
        "mean_us_per_sample": mean_us,
        "median_us_per_sample": median_us,
        "fbz_bytes": fbz_path.stat().st_size,
    }


def main():
    rows = []
    print(f"{'Dataset':<18} {'n_test':>8} {'load us':>10} "
          f"{'mean us/s':>11} {'median us/s':>13}")
    print("-" * 66)
    for loader, sid, human in DATASETS:
        r = _bench_dataset(loader, sid, human)
        rows.append(r)
        print(f"{r['human']:<18} {r['n_samples']:>8d} "
              f"{r['load_us']:>10.0f} {r['mean_us_per_sample']:>10.2f} "
              f"{r['median_us_per_sample']:>12.2f}")

    out = Path(RESULTS_DIR) / "SUMMARY_inference_host.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
