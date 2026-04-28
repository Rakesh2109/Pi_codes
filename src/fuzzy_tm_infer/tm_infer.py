#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fuzzy_tm_infer._logging import add_logging_args, configure_from_args
    from fuzzy_tm_infer.benchmarks import (
        run_booleanizer_comparison,
        run_full_comparison,
        run_real_assets,
        run_tm_vs_dt,
    )
else:
    from ._logging import add_logging_args, configure_from_args
    from .benchmarks import (
        run_booleanizer_comparison,
        run_full_comparison,
        run_real_assets,
        run_tm_vs_dt,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Fuzzy TM inference benchmarks.")
    add_logging_args(parser)
    parser.add_argument(
        "--compare-dt",
        action="store_true",
        help="compare Fuzzy TM against DecisionTree.pkl models from ml_models.zip",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="compare Python Fuzzy TM, native v17/v19, and Decision Tree",
    )
    parser.add_argument(
        "--compare-booleanizers",
        action="store_true",
        help="compare booleanizer fit/transform speed on bundled datasets",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="benchmark repeats for timing-oriented commands",
    )
    args = parser.parse_args()
    configure_from_args(args)

    if args.compare_booleanizers:
        run_booleanizer_comparison(repeats=args.repeats)
    elif args.compare_all:
        run_full_comparison()
    elif args.compare_dt:
        run_tm_vs_dt()
    else:
        run_real_assets()


if __name__ == "__main__":
    main()
