"""Per-dataset Step 2/3 trigger rates for gated GLADE."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paper_2.booleanizers.glade import GLADEBooleanizer
    from paper_2.data_loader import load_and_preprocess
else:
    from .booleanizers.glade import GLADEBooleanizer
    from .data_loader import load_and_preprocess


def loader_name(sid: str) -> str:
    return "load_toniot" if sid == "ton_iot" else f"load_{sid}"


def main() -> int:
    from paper_2.config import TM_PARAMS_PER_DATASET

    b = GLADEBooleanizer(n_bins=15, gap_ratio=1.0)
    print("Final GLADE — conditional Steps 2 & 3 (same code, all datasets)\n")
    print("Step 2: snap if g_i > median(g) and u_i < tau < u_{i+1}")
    print("Step 3: refine if min(k,n-k) < s(n); strict variance rise\n")
    print(f"{'dataset':<10} {'snap%':>8} {'perturb%':>10} {'floor':>6}")
    for sid in TM_PARAMS_PER_DATASET:
        d = load_and_preprocess(loader_name(sid))
        X = d["X_train"]
        n = X.shape[0]
        floor = b._support_floor(n)
        snap_n = snap_chg = pert_n = pert_chg = 0
        for j in range(X.shape[1]):
            col = X[:, j]
            budget = b._hybrid_budget(col)
            if budget <= 0:
                continue
            work = col
            if float(np.mean(col == 0)) > 0.3:
                nz = col[col > 0]
                if nz.size > 10 and np.unique(nz).size > 1:
                    work = nz
            wu = np.sort(np.unique(work))
            if wu.size <= budget:
                continue
            pct = np.linspace(
                100 / (budget + 1), 100 * budget / (budget + 1), budget
            )
            raw = np.percentile(work, pct)
            snapped = b._snap_structural_gaps(wu, raw)
            for a, c in zip(raw, snapped):
                snap_n += 1
                if abs(a - c) > 1e-9:
                    snap_chg += 1
            post = b._local_perturb(work, snapped)
            for a, c in zip(snapped, post):
                pert_n += 1
                if abs(a - c) > 1e-9:
                    pert_chg += 1
        print(
            f"{sid:<10} "
            f"{100*snap_chg/max(snap_n,1):7.1f}% "
            f"{100*pert_chg/max(pert_n,1):9.1f}% "
            f"{floor:>6}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
