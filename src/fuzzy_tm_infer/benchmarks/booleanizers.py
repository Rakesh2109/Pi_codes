from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..assets import ensure_assets
from ..booleanizers import (
    GLADEBooleanizer,
    KBinsBooleanizer,
    StandardBinarizer,
    ThermometerBinarizer,
)
from ..config import DATASETS
from ..data import Float32Array, load_X

try:
    from ..algorithms.c import FastGLADEBooleanizer
except (ImportError, OSError):
    FastGLADEBooleanizer = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class BooleanizerSpec:
    name: str
    factory: Callable[[], object]


@dataclass(frozen=True, slots=True)
class BooleanizerBenchmarkRow:
    dataset: str
    booleanizer: str
    n_rows: int
    n_features: int
    n_bits: int
    fit_ms: float
    transform_us: float
    packed_us: float


def run_booleanizer_comparison(repeats: int = 3) -> None:
    ensure_assets()
    specs = (
        BooleanizerSpec("glade15", lambda: GLADEBooleanizer(n_bins=15)),
        BooleanizerSpec("standard25", lambda: StandardBinarizer(max_bits_per_feature=25)),
        BooleanizerSpec(
            "kbins8_quantile", lambda: KBinsBooleanizer(n_bins=8, strategy="quantile")
        ),
        BooleanizerSpec(
            "kbins8_uniform", lambda: KBinsBooleanizer(n_bins=8, strategy="uniform")
        ),
        BooleanizerSpec("thermo8", lambda: ThermometerBinarizer(resolution=8)),
    )
    if FastGLADEBooleanizer is not None:
        specs = (
            BooleanizerSpec("fast_glade15", lambda: FastGLADEBooleanizer(n_bins=15)),
            *specs,
        )

    rows: list[BooleanizerBenchmarkRow] = []
    print("=" * 112)
    print("  BOOLEANIZER SPEED BENCHMARK")
    print("=" * 112)
    print(
        f"\n  {'Dataset':<10}  {'Booleanizer':<17}  {'Rows':>7}  {'Feat':>5}"
        f"  {'Bits':>6}  {'Fit ms':>9}  {'Xform us':>9}  {'Pack us':>9}"
    )
    print("  " + "-" * 100)

    for _, dataset_name in DATASETS:
        X = load_X(dataset_name)
        for spec in specs:
            row = _benchmark_booleanizer(spec, X, dataset_name, repeats)
            rows.append(row)
            print(
                f"  {row.dataset:<10}  {row.booleanizer:<17}  {row.n_rows:>7}"
                f"  {row.n_features:>5}  {row.n_bits:>6}  {row.fit_ms:>9.2f}"
                f"  {row.transform_us:>9.3f}  {row.packed_us:>9.3f}"
            )

    print("\n  FASTEST UNPACKED TRANSFORM PER DATASET")
    print("  " + "-" * 58)
    for _, dataset_name in DATASETS:
        candidates = [row for row in rows if row.dataset == dataset_name]
        best = min(candidates, key=lambda row: row.transform_us)
        print(
            f"  {dataset_name:<10}  {best.booleanizer:<17}"
            f"  {best.transform_us:>9.3f} us/sample  bits={best.n_bits}"
        )


def _benchmark_booleanizer(
    spec: BooleanizerSpec,
    X: Float32Array,
    dataset_name: str,
    repeats: int,
) -> BooleanizerBenchmarkRow:
    fit_times: list[float] = []
    fitted = None
    for _ in range(max(1, repeats)):
        model = spec.factory()
        start = time.perf_counter()
        fitted = model.fit(X)  # type: ignore[attr-defined]
        fit_times.append(time.perf_counter() - start)

    assert fitted is not None
    warmup_n = min(256, X.shape[0])
    _consume(fitted.transform(X[:warmup_n]))  # type: ignore[attr-defined]
    _consume(fitted.transform(X[:warmup_n], pack_bits=True))  # type: ignore[attr-defined]

    transform_time = _best_transform_time(fitted, X, pack_bits=False, repeats=repeats)
    packed_time = _best_transform_time(fitted, X, pack_bits=True, repeats=repeats)
    sample = fitted.transform(X[:1])  # type: ignore[attr-defined]

    return BooleanizerBenchmarkRow(
        dataset=dataset_name,
        booleanizer=spec.name,
        n_rows=int(X.shape[0]),
        n_features=int(X.shape[1]),
        n_bits=_n_bits(fitted, sample),
        fit_ms=min(fit_times) * 1_000.0,
        transform_us=transform_time / X.shape[0] * 1_000_000.0,
        packed_us=packed_time / X.shape[0] * 1_000_000.0,
    )


def _best_transform_time(
    model: object,
    X: Float32Array,
    pack_bits: bool,
    repeats: int,
) -> float:
    best = float("inf")
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        out = model.transform(X, pack_bits=pack_bits)  # type: ignore[attr-defined]
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)
        _consume(out)
    return best


def _n_bits(model: object, sample: np.ndarray) -> int:
    if hasattr(model, "n_bits"):
        try:
            return int(model.n_bits)  # type: ignore[attr-defined]
        except RuntimeError:
            pass
    return int(np.prod(sample.shape[1:]))


def _consume(out: np.ndarray) -> None:
    if out.size:
        int(out.reshape(-1)[0])
