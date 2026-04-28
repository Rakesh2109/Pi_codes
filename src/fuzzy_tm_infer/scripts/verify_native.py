#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fuzzy_tm_infer._logging import add_logging_args, configure_from_args, logger
    from fuzzy_tm_infer.algorithms import TMModel
    from fuzzy_tm_infer.algorithms.c import FuzzyTMModel, available_versions
    from fuzzy_tm_infer.config import DATASETS
    from fuzzy_tm_infer.data import load_X, model_path
else:
    from .._logging import add_logging_args, configure_from_args, logger
    from ..algorithms import TMModel
    from ..algorithms.c import FuzzyTMModel, available_versions
    from ..config import DATASETS
    from ..data import load_X, model_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify native Fuzzy TM backend predictions against Python."
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_from_args(args, default_level="INFO")

    all_ok = True
    for version in available_versions():
        logger.info(version)
        for stem, name in DATASETS:
            x = load_X(name)
            py_model = TMModel.from_fbz(model_path(stem))
            py_pred = np.array([py_model.predict(x[i]) for i in range(len(x))], dtype=np.int32)
            with FuzzyTMModel.from_fbz(model_path(stem), version=version) as native_model:
                if version == "v17":
                    native_model.calibrate(x[: min(len(x), 3000)])
                native_pred = native_model.predict_batch(x)
            same = np.array_equal(py_pred, native_pred)
            diffs = int((py_pred != native_pred).sum())
            logger.info(f"  {name:<10} same={same} rows={len(x)} diffs={diffs}")
            all_ok = all_ok and same

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
