from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..algorithms import TMModel
from ..algorithms.c import FuzzyTMModel
from ..algorithms.py import DecisionTreeModel
from ..assets import ensure_assets, ensure_ml_models
from ..config import DATASETS
from ..data import (
    Float32Array,
    Int32Array,
    dt_model_path,
    load_dt_testset,
    load_X,
    load_y,
    model_path,
)
from ..metrics import macro_f1


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    name: str
    h_words: int
    n_literals: int
    n_classes: int
    us_per_sample: float
    accuracy: float
    f1: float


@dataclass(frozen=True, slots=True)
class CompareRow:
    name: str
    tm_us: float
    tm_accuracy: float
    tm_f1: float
    dt_us: float
    dt_accuracy: float
    dt_f1: float


@dataclass(frozen=True, slots=True)
class FullCompareRow:
    name: str
    python_us: float
    v17_us: float
    v19_us: float
    dt_us: float
    python_accuracy: float
    v17_accuracy: float
    v19_accuracy: float
    dt_accuracy: float
    python_f1: float
    v17_f1: float
    v19_f1: float
    dt_f1: float


def run_real_assets() -> None:
    print("=" * 78)
    print("  TM INFERENCE BENCHMARK (assets inside fuzzy_tm_infer)")
    print("=" * 78)
    ensure_assets()

    rows: list[BenchmarkRow] = []
    print(f"\n  {'Dataset':<10}  {'H':>2}  {'N':>4}  {'K':>3}"
          f"  {'us/sample':>10}  {'Acc':>7}  {'F1':>7}")
    print("  " + "-" * 55)

    for stem, name in DATASETS:
        path = model_path(stem)
        model = TMModel.from_fbz(path)
        X = load_X(name)
        y = load_y(name)
        us, acc, f1 = _benchmark_model(model, X, y, model.layout.n_classes)

        row = BenchmarkRow(
            name=name,
            h_words=model.layout.h_words,
            n_literals=model.layout.n_literals,
            n_classes=model.layout.n_classes,
            us_per_sample=us,
            accuracy=acc,
            f1=f1,
        )
        rows.append(row)
        print(f"  {name:<10}  {row.h_words:>2}  {row.n_literals:>4}  {row.n_classes:>3}"
              f"  {us:>9.3f}   {acc:>6.4f}   {f1:>6.4f}")

    print("\n  SUMMARY (sorted by us/sample)")
    print("  " + "-" * 55)
    for row in sorted(rows, key=lambda item: item.us_per_sample):
        print(f"  {row.name:<10}  {row.h_words:>2}  {row.n_literals:>4}  {row.n_classes:>3}"
              f"  {row.us_per_sample:>9.3f}   {row.accuracy:>6.4f}   {row.f1:>6.4f}")


def run_tm_vs_dt() -> None:
    print("=" * 96)
    print("  FUZZY TM VS DECISION TREE BENCHMARK")
    print("=" * 96)
    ensure_assets()
    ensure_ml_models()

    rows: list[CompareRow] = []
    print(
        f"\n  {'Dataset':<10}  {'TM us':>9}  {'TM Acc':>7}  {'TM F1':>7}"
        f"  {'DT us':>9}  {'DT Acc':>7}  {'DT F1':>7}  {'TM/DT':>7}"
    )
    print("  " + "-" * 86)

    for stem, name in DATASETS:
        tm_model = TMModel.from_fbz(model_path(stem))
        tm_X = load_X(name)
        tm_y = load_y(name)

        dt_model = DecisionTreeModel.from_pickle(dt_model_path(stem))
        dt_X, dt_y = load_dt_testset(stem)

        tm_us, tm_acc, tm_f1 = _benchmark_model(tm_model, tm_X, tm_y, tm_model.layout.n_classes)
        dt_us, dt_acc, dt_f1 = _benchmark_model(dt_model, dt_X, dt_y, dt_model.n_classes)

        row = CompareRow(
            name=name,
            tm_us=tm_us,
            tm_accuracy=tm_acc,
            tm_f1=tm_f1,
            dt_us=dt_us,
            dt_accuracy=dt_acc,
            dt_f1=dt_f1,
        )
        rows.append(row)
        print(
            f"  {name:<10}  {tm_us:>9.3f}  {tm_acc:>7.4f}  {tm_f1:>7.4f}"
            f"  {dt_us:>9.3f}  {dt_acc:>7.4f}  {dt_f1:>7.4f}  {tm_us / dt_us:>7.2f}"
        )

    print("\n  SUMMARY (sorted by TM/DT ratio; lower means TM is faster)")
    print("  " + "-" * 86)
    for row in sorted(rows, key=lambda item: item.tm_us / item.dt_us):
        ratio = row.tm_us / row.dt_us
        faster = "TM" if ratio < 1.0 else "DT"
        print(f"  {row.name:<10}  {ratio:>7.2f}x  faster={faster:<2}")


def run_full_comparison() -> None:
    print("=" * 118)
    print("  FUZZY TM PYTHON/NATIVE VS DECISION TREE BENCHMARK")
    print("=" * 118)
    ensure_assets()
    ensure_ml_models()

    rows: list[FullCompareRow] = []
    print(
        f"\n  {'Dataset':<10}  {'Py us':>8}  {'v17 us':>8}  {'v19 us':>8}  {'DT us':>8}"
        f"  {'Py F1':>7}  {'v17 F1':>7}  {'v19 F1':>7}  {'DT F1':>7}"
        f"  {'Best':>6}"
    )
    print("  " + "-" * 106)

    for stem, name in DATASETS:
        X = load_X(name)
        y = load_y(name)
        py_model = TMModel.from_fbz(model_path(stem))
        py_us, py_acc, py_f1 = _benchmark_model(
            py_model, X, y, py_model.layout.n_classes
        )

        v17_us, v17_acc, v17_f1 = _benchmark_native("v17", stem, X, y)
        v19_us, v19_acc, v19_f1 = _benchmark_native("v19", stem, X, y)

        dt_model = DecisionTreeModel.from_pickle(dt_model_path(stem))
        dt_X, dt_y = load_dt_testset(stem)
        dt_us, dt_acc, dt_f1 = _benchmark_batch_model(
            dt_model, dt_X, dt_y, dt_model.n_classes
        )

        speeds = {"py": py_us, "v17": v17_us, "v19": v19_us, "dt": dt_us}
        best = min(speeds, key=speeds.__getitem__)
        rows.append(
            FullCompareRow(
                name=name,
                python_us=py_us,
                v17_us=v17_us,
                v19_us=v19_us,
                dt_us=dt_us,
                python_accuracy=py_acc,
                v17_accuracy=v17_acc,
                v19_accuracy=v19_acc,
                dt_accuracy=dt_acc,
                python_f1=py_f1,
                v17_f1=v17_f1,
                v19_f1=v19_f1,
                dt_f1=dt_f1,
            )
        )
        print(
            f"  {name:<10}  {py_us:>8.3f}  {v17_us:>8.3f}  {v19_us:>8.3f}  {dt_us:>8.3f}"
            f"  {py_f1:>7.4f}  {v17_f1:>7.4f}  {v19_f1:>7.4f}  {dt_f1:>7.4f}"
            f"  {best:>6}"
        )

    print("\n  ACCURACY")
    print("  " + "-" * 66)
    print(f"  {'Dataset':<10}  {'Py Acc':>8}  {'v17 Acc':>8}  {'v19 Acc':>8}  {'DT Acc':>8}")
    for row in rows:
        print(
            f"  {row.name:<10}  {row.python_accuracy:>8.4f}  {row.v17_accuracy:>8.4f}"
            f"  {row.v19_accuracy:>8.4f}  {row.dt_accuracy:>8.4f}"
        )


def _benchmark_model(
    model: object,
    X: Float32Array,
    y: Int32Array,
    n_classes: int,
) -> tuple[float, float, float]:
    if hasattr(model, "predict_batch"):
        return _benchmark_batch_model(model, X, y, n_classes)

    n = len(X)
    for i in range(min(5_000, n)):
        model.predict(X[i % n])

    y_pred: NDArray[np.int32] = np.empty(n, np.int32)
    for i in range(n):
        y_pred[i] = model.predict(X[i])

    n_time = min(3_000, n)
    t0 = time.perf_counter()
    for i in range(n_time):
        model.predict(X[i % n])
    us = (time.perf_counter() - t0) / n_time * 1e6

    return (
        us,
        float((y_pred == y).mean()),
        macro_f1(y, y_pred, n_classes),
    )


def _benchmark_native(
    version: str,
    stem: str,
    X: Float32Array,
    y: Int32Array,
) -> tuple[float, float, float]:
    with FuzzyTMModel.from_fbz(model_path(stem), version=version) as model:
        model.calibrate(X[: min(3_000, len(X))])
        model.predict_batch(X[: min(1_000, len(X))])
        y_pred = model.predict_batch(X)

        t0 = time.perf_counter()
        model.predict_batch(X)
        us = (time.perf_counter() - t0) / len(X) * 1e6

        return (
            us,
            float((y_pred == y).mean()),
            macro_f1(y, y_pred, model.n_classes),
        )


def _benchmark_batch_model(
    model: object,
    X: Float32Array,
    y: Int32Array,
    n_classes: int,
) -> tuple[float, float, float]:
    model.predict_batch(X[: min(1_000, len(X))])
    y_pred = model.predict_batch(X)

    t0 = time.perf_counter()
    model.predict_batch(X)
    us = (time.perf_counter() - t0) / len(X) * 1e6

    return (
        us,
        float((y_pred == y).mean()),
        macro_f1(y, y_pred, n_classes),
    )

if __name__ == "__main__":
    run_real_assets()
