#!/usr/bin/env python3
"""Copy trained Paper_2 models + test data into the GLADE_Enhanced_TM
inference package so the bench scripts there can use them.

Layout produced (under ``DEST_ASSETS``):

  tm_models/
    <stem>_model.fbz                # GLADE+FPTM (existing format)

  ml_models/<stem>/
    testset.npz                     # raw X_te, Y_te (NumPy)
    scaler.pkl                      # sklearn fitted StandardScaler
    XGBoost.pkl  MLP_*.pkl  ...     # all baseline classifiers
    model.pkl                       # GLADE+FPTM bundle (uncompressed)
    model.fbz                       # GLADE+FPTM bundle (zstd-1)
    model_kbins.pkl                 # KBins+FPTM bundle
    model_standard.pkl              # Standard+FPTM bundle

The ``ton_iot`` ↔ ``toniot`` naming mismatch between the two repos
is handled by ``DATASET_STEM_MAP``.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from .config import DATASETS, RESULTS_DIR
from .data_loader import load_and_preprocess


def _legacy_fbz_module():
    """Import the FBZ1 writer used by fuzzy_tm_infer's reader."""
    src = Path("/IoT/FuzzyPatternTM/examples/glade/benchmark/fcm_bitmask_zstd.py")
    spec = importlib.util.spec_from_file_location("fcm_bitmask_zstd", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fcm_bitmask_zstd"] = mod
    spec.loader.exec_module(mod)
    return mod

SRC_RESULTS = Path(RESULTS_DIR)
DEST_ASSETS = Path("/IoT/GLADE_Enhanced_TM/src/fuzzy_tm_infer/assets")
DATASET_STEM_MAP = {"ton_iot": "toniot"}        # paper2 → infer-repo name

ML_FILES = [
    "XGBoost.pkl", "RandomForest.pkl", "kNN_5.pkl",
    "MLP_med.pkl", "MLP_small.pkl", "MLP_tiny.pkl",
    "DecisionTree.pkl", "GaussianNB.pkl",
    "LinearSVM.pkl", "LogisticRegression.pkl",
]
TM_BUNDLE_FILES = ["model.pkl", "model.fbz",
                   "model_kbins.pkl", "model_standard.pkl"]


def _save_testset(loader_name: str, dest_npz: Path) -> tuple[int, int]:
    """Persist (X_test, Y_test) as a single .npz next to the models."""
    data = load_and_preprocess(loader_name)
    Xte = np.asarray(data["X_test"], dtype=np.float32)
    yte = np.asarray(data["y_test"], dtype=np.int32)
    np.savez(dest_npz, X_te=Xte, Y_te=yte)
    return Xte.shape


def _save_scaler(loader_name: str, dest_pkl: Path) -> None:
    """Fit a StandardScaler on X_train and pickle it (matches what the
    bundled ml_models/scaler.pkl in the original assets contains)."""
    data = load_and_preprocess(loader_name)
    scaler = StandardScaler().fit(data["X_train"])
    with dest_pkl.open("wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main():
    DEST_ASSETS.mkdir(parents=True, exist_ok=True)
    (DEST_ASSETS / "tm_models").mkdir(exist_ok=True)
    legacy_fbz = _legacy_fbz_module()

    summary = []
    for loader_name, sid, human in DATASETS:
        stem = DATASET_STEM_MAP.get(sid, sid)
        src_models = SRC_RESULTS / sid / "models"
        dst_dir = DEST_ASSETS / "ml_models" / stem
        dst_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {human}  ({sid} → {stem}) ===")
        copied = 0

        # 1. ML pickles
        for fn in ML_FILES:
            ok = _copy_if_exists(src_models / fn, dst_dir / fn)
            if ok:
                copied += 1
                print(f"  · {fn}")

        # 2. TM bundles (GLADE / KBins / Standard) — sit alongside ML
        for fn in TM_BUNDLE_FILES:
            ok = _copy_if_exists(src_models / fn, dst_dir / fn)
            if ok:
                copied += 1
                print(f"  · {fn}")

        # 3. Canonical FBZ for fuzzy_tm_infer.
        # That reader expects the *legacy* FBZ1 layout (20-byte header,
        # uncompressed GLADE prelude, zstd-22 clauses), not Paper_2's new
        # lean-header v3. Re-encode through fcm_bitmask_zstd.write_fbz.
        rules_json = src_models / "tm_rules.json"
        glade_json = src_models / "glade.json"
        if rules_json.exists() and glade_json.exists():
            rules = json.loads(rules_json.read_text())
            glade_payload = json.loads(glade_json.read_text())
            # write_fbz keys on glade_state["thresholds" or "thresh"];
            # Paper_2's glade.json already uses "thresh".
            class_names_str = [str(c) for c in rules.get("classes", [])]
            out_fbz = DEST_ASSETS / "tm_models" / f"{stem}_model.fbz"
            n_bytes = legacy_fbz.write_fbz(
                str(out_fbz), rules, glade_payload,
                class_names=class_names_str,
            )
            # Mirror to ml_models/<stem>/model.fbz so both code paths find it
            shutil.copy2(out_fbz, dst_dir / "model.fbz")
            print(f"  · tm_models/{stem}_model.fbz  (legacy FBZ1, "
                  f"{n_bytes/1024:.2f} KB)")
            copied += 1

        # 4. testset + scaler — let bench_inference_all.py find them
        npz = dst_dir / "testset.npz"
        scaler = dst_dir / "scaler.pkl"
        shape = _save_testset(loader_name, npz)
        _save_scaler(loader_name, scaler)
        print(f"  · testset.npz  shape={shape}")
        print(f"  · scaler.pkl")

        summary.append((human, stem, copied, shape[0]))

    print(f"\n{'='*60}\n  COPY SUMMARY\n{'='*60}")
    for human, stem, n, n_test in summary:
        print(f"  {human:<18} → assets/ml_models/{stem:<8}  "
              f"{n} files, {n_test} test samples")
    print(f"\nDestination root: {DEST_ASSETS}")


if __name__ == "__main__":
    main()
