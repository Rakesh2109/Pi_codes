#!/usr/bin/env python3
"""
Compare GLADE v1, GLADE v2, StandardBinarizer, KBins (uniform/quantile),
and Thermometer under the tuned FPTM configs used for the paper.
"""

import os
import json
import time
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .data_loader import load_and_preprocess
from .glade_v1 import GLADE as GLADEV1
from .glade_v2 import GLADEv2
from .booleanizers import (
    StandardBinarizer,
    KBinsBooleanizer,
    ThermometerBinarizer,
)

DATASETS = [
    ("load_wustl", "wustl", "WUSTL", 15,
        dict(CLAUSES=60, T=8, S=300, L=50, LF=15, EPOCHS=200,
             STATES_NUM=256, INCLUDE_LIMIT=128)),
    ("load_nslkdd", "nslkdd", "NSL-KDD", 8,
        dict(CLAUSES=90, T=12, S=200, L=40, LF=8, EPOCHS=80,
             STATES_NUM=256, INCLUDE_LIMIT=128)),
    ("load_toniot", "ton_iot", "TON_IoT", 15,
        dict(CLAUSES=100, T=15, S=20, L=25, LF=8, EPOCHS=200,
             STATES_NUM=256, INCLUDE_LIMIT=128)),
    ("load_medsec", "medsec", "MedSec-25", 15,
        dict(CLAUSES=80, T=12, S=75, L=30, LF=8, EPOCHS=300,
             STATES_NUM=256, INCLUDE_LIMIT=128)),
]

BINS = [
    ("GLADE_v1", lambda nb: GLADEV1(n_bins=nb)),
    ("GLADE_v2", lambda nb: GLADEv2(n_bins=nb)),
    ("StandardBin", lambda nb: StandardBinarizer(max_bits_per_feature=nb)),
    ("KBinsUnif", lambda nb: KBinsBooleanizer(n_bins=nb, strategy="uniform")),
    ("KBinsQuant", lambda nb: KBinsBooleanizer(n_bins=nb, strategy="quantile")),
    ("Thermometer", lambda nb: ThermometerBinarizer(resolution=nb)),
]

N_WORKERS = 32
TRAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_tm.jl")
ROOT = "/tmp/paper2_glade_compare_chosen"
OUT_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    "results",
    "binarizers",
)
OUT_FILE = os.path.join(OUT_DIR, "compare_chosen.json")


def _run(args):
    sid, bname, cfg, xtr_b, xte_b, ytr, yte = args
    work = tempfile.mkdtemp(prefix=f"{sid}_{bname}_", dir=ROOT)
    try:
        np.savetxt(os.path.join(work, "X_train.txt"), xtr_b, fmt="%d")
        np.savetxt(os.path.join(work, "X_test.txt"), xte_b, fmt="%d")
        np.savetxt(os.path.join(work, "Y_train.txt"), ytr, fmt="%d")
        np.savetxt(os.path.join(work, "Y_test.txt"), yte, fmt="%d")
        env = os.environ.copy()
        env.update({
            "TM_CLAUSES": str(cfg["CLAUSES"]),
            "TM_T": str(cfg["T"]),
            "TM_S": str(cfg["S"]),
            "TM_L": str(cfg["L"]),
            "TM_LF": str(cfg["LF"]),
            "TM_EPOCHS": str(cfg["EPOCHS"]),
            "TM_STATES": str(cfg["STATES_NUM"]),
            "TM_INCLUDE": str(cfg["INCLUDE_LIMIT"]),
            "TMP_DIR": work,
        })
        t0 = time.perf_counter()
        result = subprocess.run(
            ["julia", "--threads=1", TRAIN],
            capture_output=True,
            text=True,
            timeout=14400,
            env=env,
        )
        wall = time.perf_counter() - t0
        if result.returncode != 0:
            return (sid, bname, None, wall, result.stderr[-300:])
        with open(os.path.join(work, "tm_metrics.json")) as f:
            metrics = json.load(f)
        pc = list(metrics.get("per_class", {}).values())
        prec = sum(x["precision"] for x in pc) / len(pc) if pc else None
        rec = sum(x["recall"] for x in pc) / len(pc) if pc else None
        return (sid, bname, {
            "acc": float(metrics["accuracy"]),
            "mf1": float(metrics["macro_f1"]),
            "mp": prec,
            "mr": rec,
            "bits": int(xtr_b.shape[1]),
        }, wall, "")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    os.makedirs(ROOT, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    prepared = []
    for loader, sid, human, nb, cfg in DATASETS:
        print(f"=== {human} (n_bins={nb}) ===", flush=True)
        d = load_and_preprocess(loader)
        xtr, xte = d["X_train"], d["X_test"]
        ytr, yte = d["y_train"], d["y_test"]
        for bname, ctor in BINS:
            b = ctor(nb)
            try:
                xtr_b = b.fit_transform(xtr).astype(np.uint8)
                xte_b = b.transform(xte).astype(np.uint8)
            except Exception as exc:
                print(f"  [{bname}] FIT FAIL: {exc}", flush=True)
                continue
            prepared.append((sid, human, bname, cfg, xtr_b, xte_b, ytr, yte))
            print(f"  {bname:<14} bits={xtr_b.shape[1]:>5}", flush=True)
    print(f"\n{len(prepared)} jobs x {N_WORKERS} workers\n", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(_run, (sid, bname, cfg, xtr, xte, ytr, yte)):
            (sid, human, bname)
            for sid, human, bname, cfg, xtr, xte, ytr, yte in prepared
        }
        for future in as_completed(futures):
            sid, human, bname = futures[future]
            _, _, metrics, wall, err = future.result()
            if metrics is None:
                print(f"  [{human}/{bname}] FAIL {err[:120]}", flush=True)
                continue
            metrics["ds"] = human
            metrics["bin"] = bname
            metrics["wall"] = wall
            results.append(metrics)
            print(
                f"  [{human:<10}/{bname:<14}] bits={metrics['bits']:>5}  "
                f"acc={metrics['acc']:.4f} f1={metrics['mf1']:.4f} "
                f"p={metrics['mp']:.4f} r={metrics['mr']:.4f} wall={wall:5.1f}s",
                flush=True,
            )
    by_ds = {}
    for row in results:
        by_ds.setdefault(row["ds"], []).append(row)
    with open(OUT_FILE, "w") as f:
        json.dump(by_ds, f, indent=2, default=float)
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()
