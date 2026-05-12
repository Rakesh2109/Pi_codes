#!/usr/bin/env python3
"""
Run one model family on one dataset (or all datasets).
"""

import argparse

from .config import DATASETS
from . import run_ml, run_tinyml, run_tm, run_ml_tmmatched


TASKS = {
    "ml": run_ml,
    "tinyml": run_tinyml,
    "tm": run_tm,
    "ml_tmmatched": run_ml_tmmatched,
}


def _dataset_ids():
    return [sid for _, sid, _ in DATASETS]


def main():
    parser = argparse.ArgumentParser(
        description="Run one model family on a dataset."
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASKS.keys()),
        required=True,
        help="Model family to run.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset id (nslkdd, ton_iot, medsec, wustl, iotm). Default: all",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model names (ML/TinyML only).",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List supported dataset ids.",
    )
    args = parser.parse_args()

    if args.list_datasets:
        print("\n".join(_dataset_ids()))
        return

    only_models = None
    if args.models:
        only_models = [m.strip() for m in args.models.split(",") if m.strip()]

    runner = TASKS[args.task]
    if args.dataset:
        if args.dataset not in _dataset_ids():
            raise ValueError(f"Unknown dataset id: {args.dataset}")
        if args.task == "ml":
            runner.run_for_dataset(args.dataset, only_models)
        elif args.task == "tinyml":
            runner.run_for_dataset(args.dataset, only_models)
        elif args.task == "tm":
            runner.run_for_dataset(args.dataset)
        else:
            runner.run_for_dataset(args.dataset)
    else:
        runner.main()


if __name__ == "__main__":
    main()
