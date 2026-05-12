#!/usr/bin/env python3
"""Binariser comparison (paper Table III): the FPTM is trained with a fixed
per-dataset configuration on the Boolean features produced by three binarisers:

* ``glade``       — the proposed 3-stage GLADE binariser (``GLADEv2``)
* ``standard``    — TMU-style ``StandardBinarizer`` (one bit per non-minimum unique value)
* ``thermometer`` — uniform per-feature thermometer (``KBinsBooleanizer(strategy="uniform")``)

For each (dataset, binariser) cell it records bits / accuracy / macro-F1 /
macro-precision / macro-recall and writes one JSON file under
``results/binarizer_comparison/``.

Usage::

    python -m paper_2.run_binarizer_comparison                    # all 4 datasets x 3 binarisers
    python -m paper_2.run_binarizer_comparison --dataset wustl    # one dataset, all binarisers
    python -m paper_2.run_binarizer_comparison --binariser glade  # all datasets, one binariser
    python -m paper_2.run_binarizer_comparison --render-latex     # print Table III from cached JSON

Requires Julia and the FuzzyPatternTM source (set ``PAPER2_TSETLIN_PATH`` if not
at the default location). Run from ``src/``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paper_2.config import DATASETS, GLADE_N_BINS, RESULTS_DIR, TM_JULIA_SRC, TM_PARAMS_PER_DATASET
    from paper_2.data_loader import load_and_preprocess
    from paper_2.glade_v2 import GLADEv2
    from paper_2.booleanizers import StandardBinarizer, KBinsBooleanizer
else:
    from .config import DATASETS, GLADE_N_BINS, RESULTS_DIR, TM_JULIA_SRC, TM_PARAMS_PER_DATASET
    from .data_loader import load_and_preprocess
    from .glade_v2 import GLADEv2
    from .booleanizers import StandardBinarizer, KBinsBooleanizer

TRAIN_JL = str(Path(__file__).resolve().parent / "train_tm.jl")
OUT_DIR = Path(RESULTS_DIR) / "binarizer_comparison"
_SID = {sid: human for _, sid, human in DATASETS}        # short id -> human name
_LOADER = {sid: loader for loader, sid, _ in DATASETS}    # short id -> loader name

BINARISERS = {
    "glade":       lambda: GLADEv2(n_bins=GLADE_N_BINS),
    "standard":    lambda: StandardBinarizer(max_bits_per_feature=GLADE_N_BINS),
    "thermometer": lambda: KBinsBooleanizer(n_bins=GLADE_N_BINS, strategy="uniform"),
}


def _run_cell(sid: str, bname: str) -> dict:
    human = _SID[sid]
    cfg = TM_PARAMS_PER_DATASET[sid]
    d = load_and_preprocess(_LOADER[sid])
    xtr, xte = np.asarray(d["X_train"], np.float64), np.asarray(d["X_test"], np.float64)
    ytr, yte = d["y_train"], d["y_test"]

    b = BINARISERS[bname]()
    xtr_b = np.asarray(b.fit_transform(xtr)).astype(np.uint8)
    xte_b = np.asarray(b.transform(xte)).astype(np.uint8)
    if xtr_b.ndim != 2:
        xtr_b = xtr_b.reshape(xtr_b.shape[0], -1)
        xte_b = xte_b.reshape(xte_b.shape[0], -1)
    bits = int(xtr_b.shape[1])

    work = Path(tempfile.mkdtemp(prefix=f"{sid}_{bname}_"))
    try:
        np.savetxt(work / "X_train.txt", xtr_b, fmt="%d")
        np.savetxt(work / "X_test.txt", xte_b, fmt="%d")
        np.savetxt(work / "Y_train.txt", ytr, fmt="%d")
        np.savetxt(work / "Y_test.txt", yte, fmt="%d")
        env = os.environ.copy()
        env.update({
            "TM_CLAUSES": str(cfg["CLAUSES"]), "TM_T": str(cfg["T"]), "TM_S": str(cfg["S"]),
            "TM_L": str(cfg["L"]), "TM_LF": str(cfg["LF"]), "TM_EPOCHS": str(cfg["EPOCHS"]),
            "TM_STATES": str(cfg["STATES_NUM"]), "TM_INCLUDE": str(cfg["INCLUDE_LIMIT"]),
            "TM_SEED": "42", "TMP_DIR": str(work), "PAPER2_TSETLIN_PATH": TM_JULIA_SRC,
        })
        t0 = time.perf_counter()
        r = subprocess.run(["julia", "--threads=1", TRAIN_JL],
                           capture_output=True, text=True, timeout=14400, env=env)
        wall = time.perf_counter() - t0
        if r.returncode != 0:
            return {"dataset": human, "binariser": bname, "bits": bits, "ok": False,
                    "wall": wall, "err": r.stderr[-600:]}
        m = json.loads((work / "tm_metrics.json").read_text())
        pc = list(m.get("per_class", {}).values())
        return {
            "dataset": human, "binariser": bname, "bits": bits, "ok": True, "wall": wall,
            "acc": float(m["accuracy"]), "macro_f1": float(m["macro_f1"]),
            "macro_precision": float(sum(x["precision"] for x in pc) / len(pc)) if pc else None,
            "macro_recall": float(sum(x["recall"] for x in pc) / len(pc)) if pc else None,
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _render_latex() -> None:
    rows = {}
    for f in OUT_DIR.glob("*.json"):
        d = json.loads(f.read_text())
        if d.get("ok"):
            rows[(d["dataset"], d["binariser"])] = d
    order = ["NSL-KDD", "TON_IoT", "MedSec-25", "WUSTL-EHMS-2020"]
    print(r"\begin{tabular}{|l|ccc|ccc|ccc|}")
    print(r"\hline")
    print(r" & \multicolumn{3}{c|}{\textbf{GLADE}} & \multicolumn{3}{c|}{\textbf{\texttt{StandardBinarizer}}}"
          r" & \multicolumn{3}{c|}{\textbf{Thermometer}} \\ \hline")
    print(r"\textbf{Dataset} & Bits & Acc & F1 & Bits & Acc & F1 & Bits & Acc & F1 \\ \hline")
    for ds in order:
        cells = []
        for bn in ("glade", "standard", "thermometer"):
            c = rows.get((ds, bn))
            cells += ([str(c["bits"]), f"{c['acc']:.4f}", f"{c['macro_f1']:.4f}"] if c else ["--", "--", "--"])
        print(f"{ds.replace('_', chr(92) + '_')} & " + " & ".join(cells) + r" \\ \hline")
    print(r"\end{tabular}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m paper_2.run_binarizer_comparison",
                                description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(_SID), help="run only this dataset (default: all)")
    p.add_argument("--binariser", choices=sorted(BINARISERS), help="run only this binariser (default: all)")
    p.add_argument("--render-latex", action="store_true", help="print Table III from cached results and exit")
    args = p.parse_args(argv)

    if args.render_latex:
        _render_latex()
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sids = [args.dataset] if args.dataset else list(_SID)
    bns = [args.binariser] if args.binariser else list(BINARISERS)
    for sid in sids:
        for bn in bns:
            print(f">>> {_SID[sid]} / {bn} ...", flush=True)
            res = _run_cell(sid, bn)
            (OUT_DIR / f"{sid}_{bn}.json").write_text(json.dumps(res, indent=2))
            if res["ok"]:
                print(f"    {res['dataset']:<18} {bn:<12} bits={res['bits']:>5} "
                      f"acc={res['acc']:.4f} macro_f1={res['macro_f1']:.4f} ({res['wall']:.0f}s)", flush=True)
            else:
                print(f"    {res['dataset']} {bn} FAILED ({res['wall']:.0f}s)\n{res['err']}", flush=True)
    print(f"\nWrote per-cell JSON to {OUT_DIR}. Run with --render-latex for the LaTeX table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
