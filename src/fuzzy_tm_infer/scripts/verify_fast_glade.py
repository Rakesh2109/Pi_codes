#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fuzzy_tm_infer._logging import add_logging_args, configure_from_args, logger
    from fuzzy_tm_infer.algorithms.c import FastGLADEBooleanizer
    from fuzzy_tm_infer.assets import ensure_assets
    from fuzzy_tm_infer.booleanizers import GLADEBooleanizer
    from fuzzy_tm_infer.config import DATASETS
    from fuzzy_tm_infer.data import load_X
else:
    from .._logging import add_logging_args, configure_from_args, logger
    from ..algorithms.c import FastGLADEBooleanizer
    from ..assets import ensure_assets
    from ..booleanizers import GLADEBooleanizer
    from ..config import DATASETS
    from ..data import load_X


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify fast GLADE v2 transform against the Python reference."
    )
    parser.add_argument("--rows", type=int, default=1024, help="max rows per dataset")
    parser.add_argument("--n-bins", type=int, default=15)
    add_logging_args(parser)
    args = parser.parse_args()
    configure_from_args(args)

    _verify_random(args.n_bins)
    ensure_assets()
    for _, dataset in DATASETS:
        x = load_X(dataset)[: args.rows]
        _verify_matrix(dataset, x, args.n_bins)
    logger.success("fast GLADE v2 verification passed")


def _verify_random(n_bins: int) -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(257, 19)).astype(np.float32)
    x[::13, 0] = 0.0
    x[::17, 1] = np.nan
    _verify_matrix("random", x, n_bins)


def _verify_matrix(name: str, x: np.ndarray, n_bins: int) -> None:
    py_model = GLADEBooleanizer(n_bins=n_bins).fit(x)
    native = FastGLADEBooleanizer.from_booleanizer(py_model)
    py_u8 = py_model.transform(x, pack_bits=False)
    native_u8 = native.transform(x, pack_bits=False)
    native_u8_into = native.empty_output(x.shape[0], pack_bits=False)
    native.transform_into(x, native_u8_into, pack_bits=False)
    py_packed = py_model.transform(x, pack_bits=True)
    native_packed = native.transform(x, pack_bits=True)
    native_packed_into = native.empty_output(x.shape[0], pack_bits=True)
    native.transform_into(x, native_packed_into, pack_bits=True)
    native_u8_stream = np.vstack(
        list(native.transform_stream(x, batch_size=127, pack_bits=False))
    )
    native_packed_stream = np.vstack(
        list(native.transform_stream(x, batch_size=127, pack_bits=True))
    )

    same_u8 = bool(
        np.array_equal(py_u8, native_u8)
        and np.array_equal(py_u8, native_u8_into)
        and np.array_equal(py_u8, native_u8_stream)
    )
    same_packed = bool(
        np.array_equal(py_packed, native_packed)
        and np.array_equal(py_packed, native_packed_into)
        and np.array_equal(py_packed, native_packed_stream)
    )
    logger.info(
        "{} backend={} bits={} unpacked={} packed={}",
        name,
        native.backend,
        native.n_bits,
        same_u8,
        same_packed,
    )
    if not same_u8:
        diff = np.argwhere(py_u8 != native_u8)[0]
        raise SystemExit(
            f"{name}: unpacked mismatch at row={diff[0]} bit={diff[1]} "
            f"python={py_u8[tuple(diff)]} native={native_u8[tuple(diff)]}"
        )
    if not same_packed:
        diff = np.argwhere(py_packed != native_packed)[0]
        raise SystemExit(
            f"{name}: packed mismatch at row={diff[0]} byte={diff[1]} "
            f"python={py_packed[tuple(diff)]} native={native_packed[tuple(diff)]}"
        )


if __name__ == "__main__":
    main()
