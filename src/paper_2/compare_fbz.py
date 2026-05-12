#!/usr/bin/env python3
"""Side-by-side comparison of two `.fbz` containers:

  (A) Original FBZ (zstd-22) — fcm_bitmask_zstd.py from
      /IoT/FuzzyPatternTM/examples/glade/benchmark/.
      Layout:
        20-byte header
        feat_idx  (i32 × n_bits)         uncompressed
        thresh    (f32 × n_bits)         uncompressed
        feat_names[n] (utf-8)            uncompressed
        class_names[K] (utf-8)           uncompressed
        zstd-22 compressed clause block

  (B) Paper_2 FBZ v2 (zstd-1) — src/paper_2/fbz.py.
      Layout:
        24-byte header
        single zstd-1 compressed payload containing:
            GLADE feat_idx + thresh
            class label table (i32 or utf-8)
            clause bitmasks

For each dataset this script:
  1. Packs the same trained model with both formats.
  2. Times pack and unpack (200 iterations).
  3. Runs end-to-end inference on the test split.
  4. Reports bytes, latency, accuracy, macro-F1.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from . import fbz as fbz2
from .config import DATASETS, RESULTS_DIR
from .data_loader import load_and_preprocess


def _load_original_fbz_module():
    """Import fcm_bitmask_zstd.py from the FuzzyPatternTM repo by path."""
    src = Path("/IoT/FuzzyPatternTM/examples/glade/benchmark/fcm_bitmask_zstd.py")
    spec = importlib.util.spec_from_file_location("fcm_bitmask_zstd", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fcm_bitmask_zstd"] = mod
    spec.loader.exec_module(mod)
    return mod


def _avg_us(fn, iters=200, warmup=3):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1e6 / iters


def _predict_v2(model: dict, Xte: np.ndarray) -> np.ndarray:
    """Inference using Paper_2 fbz.transform + verify_fbz.predict."""
    from .verify_fbz import predict
    Xb = fbz2.transform(model, Xte)
    return predict(model, Xb)


def compare_one(loader_name: str, short_id: str, human_name: str,
                fcm_mod) -> dict:
    print(f"\n{'=' * 72}")
    print(f"  {human_name}")
    print(f"{'=' * 72}")

    results = Path(RESULTS_DIR)
    rules = json.loads((results / short_id / "models" / "tm_rules.json").read_text())
    glade = json.loads((results / short_id / "models" / "glade.json").read_text())

    data = load_and_preprocess(loader_name)
    Xte, yte = data["X_test"], data["y_test"]
    class_names = [str(c) for c in data["class_names"]]

    # ---- (A) original zstd-22 FBZ ----
    fbz_a = results / short_id / "models" / "model_zstd22.fbz"
    bytes_a = fcm_mod.write_fbz(str(fbz_a), rules, glade, class_names=class_names)
    pack_a_us = _avg_us(
        lambda: fcm_mod.write_fbz(str(fbz_a), rules, glade, class_names=class_names),
        iters=20)  # zstd-22 is slow — fewer iters

    t0 = time.perf_counter()
    model_a = fcm_mod.read_fbz(str(fbz_a))
    unpack_a_us = (time.perf_counter() - t0) * 1e6
    # Also do a small averaged read benchmark
    unpack_a_us = _avg_us(lambda: fcm_mod.read_fbz(str(fbz_a)), iters=10)

    yhat_a = model_a.predict(Xte.astype(np.float32))
    acc_a = accuracy_score(yte, yhat_a)
    f1_a = f1_score(yte, yhat_a, average="macro", zero_division=0)

    # ---- (B) Paper_2 v2 zstd-1 FBZ ----
    fbz_b = results / short_id / "models" / "model_zstd1.fbz"
    bytes_b = fbz2.write(fbz_b, rules, glade=glade, level=1)
    pack_b_us = _avg_us(
        lambda: fbz2.write(fbz_b, rules, glade=glade, level=1), iters=200)
    unpack_b_us = _avg_us(lambda: fbz2.read(fbz_b), iters=200)

    model_b = fbz2.read(fbz_b)
    yhat_b = _predict_v2(model_b, Xte)
    acc_b = accuracy_score(yte, yhat_b)
    f1_b = f1_score(yte, yhat_b, average="macro", zero_division=0)

    # ---- contents inventory ----
    n_pos = sum(len(model_b['rules'][l]['positive']) for l in model_b['labels'])
    n_neg = sum(len(model_b['rules'][l]['negative']) for l in model_b['labels'])

    print(f"  inputs        : {data['X_test'].shape[1]} raw features → "
          f"{model_b['n_bits']} boolean bits  ({len(class_names)} classes)")
    print(f"  clauses       : {n_pos} positive + {n_neg} negative "
          f"= {n_pos + n_neg} total")
    print()
    print(f"  {'metric':<20} {'(A) zstd-22':>14} {'(B) zstd-1':>14} "
          f"{'B/A':>8}")
    print(f"  {'-' * 60}")
    print(f"  {'file size (B)':<20} {bytes_a:>14d} {bytes_b:>14d} "
          f"{bytes_b / bytes_a:>7.2f}x")
    print(f"  {'pack time (us)':<20} {pack_a_us:>14.0f} {pack_b_us:>14.0f} "
          f"{pack_b_us / pack_a_us:>7.3f}x")
    print(f"  {'unpack time (us)':<20} {unpack_a_us:>14.0f} {unpack_b_us:>14.0f} "
          f"{unpack_b_us / unpack_a_us:>7.3f}x")
    print(f"  {'accuracy':<20} {acc_a:>14.6f} {acc_b:>14.6f}")
    print(f"  {'macro F1':<20} {f1_a:>14.6f} {f1_b:>14.6f}")

    return {
        "dataset": short_id, "human": human_name,
        "bytes_a": bytes_a, "bytes_b": bytes_b,
        "pack_a": pack_a_us, "pack_b": pack_b_us,
        "unpack_a": unpack_a_us, "unpack_b": unpack_b_us,
        "f1_a": f1_a, "f1_b": f1_b,
        "acc_a": acc_a, "acc_b": acc_b,
    }


def main():
    fcm_mod = _load_original_fbz_module()
    rows = []
    for loader, sid, human in DATASETS:
        rows.append(compare_one(loader, sid, human, fcm_mod))

    print("\n" + "=" * 78)
    print("  SUMMARY  —  (A) original zstd-22 FBZ   vs   (B) Paper_2 zstd-1 FBZ v2")
    print("=" * 78)
    print(f"  {'dataset':<10}"
          f"  {'A bytes':>9} {'B bytes':>9} {'B/A':>6}"
          f"  {'A pack':>8} {'B pack':>8}"
          f"  {'A unpk':>7} {'B unpk':>7}"
          f"  {'A F1':>7} {'B F1':>7}")
    for r in rows:
        print(f"  {r['dataset']:<10}"
              f"  {r['bytes_a']:>9d} {r['bytes_b']:>9d} "
              f"{r['bytes_b']/r['bytes_a']:>5.2f}x"
              f"  {r['pack_a']:>7.0f}u {r['pack_b']:>7.0f}u"
              f"  {r['unpack_a']:>6.0f}u {r['unpack_b']:>6.0f}u"
              f"  {r['f1_a']:>7.4f} {r['f1_b']:>7.4f}")


if __name__ == "__main__":
    main()
