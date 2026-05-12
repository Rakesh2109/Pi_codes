#!/usr/bin/env python3
"""GLADE ablation runner: turn each of the four stages off in isolation,
re-fit + re-train the FPTM, and write bits/acc/macro-F1 to JSON.

The GLADE stages this script can toggle:

* "budget"   — adaptive per-feature bit budget (`_hybrid_budget`'s sparse branch)
* "gap"      — gap-aware quantile-edge snapping (`_snap_structural_gaps`)
* "perturb"  — local refinement of each edge   (`_local_perturb`)

(The earlier "entropy" dead-bit pruning stage was removed from GLADE — the
ablation showed it never fired on the evaluated datasets — so it is no longer a
toggle.)

A "variant" is a set of disabled stages. The variants the ablation table reports
are: ``full`` (none disabled), ``no_budget``, ``no_gap``, ``no_perturb``
(exactly one stage disabled each).

The class subclasses ``paper_2.booleanizers.glade.GLADEBooleanizer``, which is
behaviour-equivalent to ``paper_2.glade_v2.GLADEv2`` (identical thresholds and
bit counts on the four datasets at ``n_bins = 15``).

CLI usage:

    python -m paper_2.glade_ablation --dataset wustl --variant full \\
        --out results/ablation/wustl_full.json

Run many in parallel by launching one process per (dataset, variant) cell.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

import numpy as np

# Allow running both as a module ("python -m paper_2.glade_ablation")
# and as a script.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paper_2.booleanizers.glade import GLADEBooleanizer
    from paper_2.config import TM_PARAMS_PER_DATASET
    from paper_2.data_loader import load_and_preprocess
else:
    from .booleanizers.glade import GLADEBooleanizer
    from .config import TM_PARAMS_PER_DATASET
    from .data_loader import load_and_preprocess


TRAIN_JL = str(Path(__file__).resolve().parent / "train_tm.jl")
N_BINS = 15
SEED = 42

VARIANTS: dict[str, frozenset[str]] = {
    "full":        frozenset(),
    "no_budget":   frozenset({"budget"}),
    "no_gap":      frozenset({"gap"}),
    "no_perturb":  frozenset({"perturb"}),
}

DATASETS = {
    "wustl":   ("load_wustl",   "WUSTL-EHMS-2020"),
    "nslkdd":  ("load_nslkdd",  "NSL-KDD"),
    "ton_iot": ("load_toniot",  "TON_IoT"),
    "medsec":  ("load_medsec",  "MedSec-25"),
}


# ─────────────────────────────────────────────────────────────────────────────
# GLADE with toggle-able stages
# ─────────────────────────────────────────────────────────────────────────────
class GLADEAblation(GLADEBooleanizer):
    """GLADEBooleanizer with one or more of the four stages turned off.

    ``disable`` is a subset of ``{"budget","gap","entropy","perturb"}``.  When a
    stage is in ``disable`` the corresponding code path falls back to a no-op:

    * "budget"  — the sparse-column log-scaled budget is skipped; dense and
      sparse columns alike receive the full ``n_bins`` budget (categorical
      handling, which is a separate code path, is unchanged).
    * "gap"     — structural-gap snapping is bypassed; raw quantile edges are
      used unchanged.
    * "perturb" — the ±¼ candidate refinement is bypassed.
    """

    _VALID = frozenset({"budget", "gap", "perturb"})

    def __init__(self, n_bins: int = 15, disable: Iterable[str] = ()):
        super().__init__(n_bins=n_bins)
        d = frozenset(disable)
        unknown = d - self._VALID
        if unknown:
            raise ValueError(
                f"unknown disable keys: {sorted(unknown)} "
                f"(valid: {sorted(self._VALID)})"
            )
        self.disable = d

    # Stage 1 — adaptive bit budget (only the sparse branch is "adaptive")
    def _hybrid_budget(self, col):
        n_unique = np.unique(col).size
        if n_unique <= 1:
            return 0
        if n_unique <= self.n_bins:
            return max(1, n_unique - 1)
        if "budget" in self.disable:
            return self.n_bins
        zero_fraction = float(np.mean(col == 0))
        if zero_fraction > 0.3:
            nonzero = col[col != 0]
            nonzero_unique = np.unique(nonzero).size if nonzero.size else n_unique
            density = max(1.0 - zero_fraction, 0.01)
            effective = max(nonzero_unique * density**2, 2.0)
            budget = int(np.ceil(np.log2(effective)) + 2)
            return max(1, min(budget, self.n_bins))
        return self.n_bins

    # Stage 2 — gap-aware snap
    def _snap_structural_gaps(self, unique_values, raw_edges):
        if "gap" in self.disable:
            return np.unique(raw_edges)
        return GLADEBooleanizer._snap_structural_gaps(unique_values, raw_edges)

    # Stage — local refinement
    def _local_perturb(self, col, edges):
        if "perturb" in self.disable:
            return np.asarray(edges, dtype=np.float64)
        return GLADEBooleanizer._local_perturb(col, edges)


# ─────────────────────────────────────────────────────────────────────────────
# Train + measure one (dataset, variant) cell
# ─────────────────────────────────────────────────────────────────────────────
def run_one(dataset_sid: str, variant: str, work_root: Path) -> dict:
    loader, human = DATASETS[dataset_sid]
    disable = VARIANTS[variant]
    cfg = TM_PARAMS_PER_DATASET[dataset_sid]

    d = load_and_preprocess(loader)
    xtr, xte = d["X_train"], d["X_test"]
    ytr, yte = d["y_train"], d["y_test"]

    b = GLADEAblation(n_bins=N_BINS, disable=disable)
    xtr_b = b.fit_transform(xtr).astype(np.uint8)
    xte_b = b.transform(xte).astype(np.uint8)
    if xtr_b.ndim != 2:
        xtr_b = xtr_b.reshape(xtr_b.shape[0], -1)
        xte_b = xte_b.reshape(xte_b.shape[0], -1)
    bits = int(xtr_b.shape[1])

    work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{dataset_sid}_{variant}_", dir=str(work_root)))
    try:
        np.savetxt(work / "X_train.txt", xtr_b, fmt="%d")
        np.savetxt(work / "X_test.txt",  xte_b, fmt="%d")
        np.savetxt(work / "Y_train.txt", ytr,   fmt="%d")
        np.savetxt(work / "Y_test.txt",  yte,   fmt="%d")
        env = os.environ.copy()
        env.update({
            "TM_CLAUSES": str(cfg["CLAUSES"]), "TM_T": str(cfg["T"]),
            "TM_S": str(cfg["S"]),  "TM_L": str(cfg["L"]),
            "TM_LF": str(cfg["LF"]), "TM_EPOCHS": str(cfg["EPOCHS"]),
            "TM_STATES": str(cfg["STATES_NUM"]),
            "TM_INCLUDE": str(cfg["INCLUDE_LIMIT"]),
            "TM_SEED": str(SEED), "TMP_DIR": str(work),
        })
        t0 = time.perf_counter()
        r = subprocess.run(
            ["julia", "--threads=1", TRAIN_JL],
            capture_output=True, text=True, timeout=14400, env=env,
        )
        wall = time.perf_counter() - t0
        if r.returncode != 0:
            return {
                "dataset": human, "variant": variant, "disable": sorted(disable),
                "bits": bits, "ok": False, "wall": wall, "err": r.stderr[-500:],
            }
        m = json.loads((work / "tm_metrics.json").read_text())
        return {
            "dataset": human, "variant": variant, "disable": sorted(disable),
            "bits": bits, "ok": True, "wall": wall,
            "acc": float(m["accuracy"]),
            "macro_f1": float(m["macro_f1"]),
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m paper_2.glade_ablation",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    p.add_argument("--out", required=True, help="JSON output path for this cell")
    p.add_argument("--work-root", default="/tmp/glade_ablation",
                   help="scratch directory for per-cell Julia working dirs")
    args = p.parse_args(argv)

    work_root = Path(args.work_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    res = run_one(args.dataset, args.variant, work_root=work_root)
    out_path.write_text(json.dumps(res, indent=2))
    if res["ok"]:
        print(f"{res['dataset']:<18} {args.variant:<12} bits={res['bits']:>5} "
              f"acc={res['acc']:.4f} macro_f1={res['macro_f1']:.4f} "
              f"wall={res['wall']:.1f}s", flush=True)
    else:
        print(f"{res['dataset']:<18} {args.variant:<12} FAILED ({res['wall']:.1f}s)",
              flush=True)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
