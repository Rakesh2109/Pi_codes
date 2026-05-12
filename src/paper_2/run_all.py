#!/usr/bin/env python3
"""
Run all model families. This can be long and requires Julia for TM.
"""

import os
import argparse

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from . import run_ml, run_tinyml, run_tm, run_ml_tmmatched


def main():
    parser = argparse.ArgumentParser(description="Run all model families.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset id to run. Default: all datasets",
    )
    parser.add_argument(
        "--skip-tm",
        action="store_true",
        help="Skip TM + GLADE (Julia).",
    )
    parser.add_argument(
        "--skip-ml",
        action="store_true",
        help="Skip ML baselines.",
    )
    parser.add_argument(
        "--skip-tinyml",
        action="store_true",
        help="Skip TinyML baselines.",
    )
    parser.add_argument(
        "--skip-ml-tmmatched",
        action="store_true",
        help="Skip TM-matched ML baselines.",
    )
    args = parser.parse_args()

    if args.dataset:
        if not args.skip_ml:
            run_ml.run_for_dataset(args.dataset)
        if not args.skip_tinyml:
            run_tinyml.run_for_dataset(args.dataset)
        if not args.skip_ml_tmmatched:
            run_ml_tmmatched.run_for_dataset(args.dataset)
        if not args.skip_tm:
            run_tm.run_for_dataset(args.dataset)
        return

    if not args.skip_ml:
        run_ml.main()
    if not args.skip_tinyml:
        run_tinyml.main()
    if not args.skip_ml_tmmatched:
        run_ml_tmmatched.main()
    if not args.skip_tm:
        run_tm.main()


if __name__ == "__main__":
    main()
