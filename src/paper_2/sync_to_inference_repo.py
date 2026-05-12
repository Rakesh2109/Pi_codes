#!/usr/bin/env python3
"""Replace the old models in
``/IoT/GLADE_Enhanced_TM/src/fuzzy_tm_infer/assets/ml_models/<stem>/``
with freshly-built copies from this Paper_2 results tree.

The destination layout is normalised so every consumer of
``fuzzy_tm_infer`` finds the same shapes:

    ml_models/<stem>/
        scaler.pkl                   StandardScaler
        testset.npz                  X_te, Y_te
        XGBoost.pkl                  raw XGBClassifier
        RandomForest.pkl             raw RandomForestClassifier
        kNN_5.pkl                    raw KNeighborsClassifier
        MLP_med.pkl  ...             raw MLPClassifier
        DecisionTree.pkl             raw DecisionTreeClassifier
        GaussianNB.pkl               raw GaussianNB
        LinearSVM.pkl                raw LinearSVC
        LogisticRegression.pkl       raw LogisticRegression
        model.pkl                    {tm_rules, binarizer_kind="GLADE",    binarizer_state}
        model_kbins.pkl              {tm_rules, binarizer_kind="KBins",    binarizer_state}
        model_standard.pkl           {tm_rules, binarizer_kind="Standard", binarizer_state}
        <stem>_model.fbz             legacy FBZ1 (zstd-22) for TMModel.from_fbz

(The TM bundles use a *flat dict* rather than pickled binarizer objects
so the consumer doesn't need to import ``src.paper_2.booleanizers``.)
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

DEST_ROOT = Path("/IoT/GLADE_Enhanced_TM/src/fuzzy_tm_infer/assets")
ML_DEST   = DEST_ROOT / "ml_models"
TM_DEST   = DEST_ROOT / "tm_models"
DATASET_STEM_MAP = {"ton_iot": "toniot"}

# Paper_2 ML pkls saved as {model, scaler, class_names}; we strip to the
# raw classifier and save the scaler once next to it.
ML_FILES = [
    "XGBoost.pkl", "RandomForest.pkl", "kNN_5.pkl",
    "MLP_med.pkl", "MLP_small.pkl", "MLP_tiny.pkl",
    "DecisionTree.pkl", "GaussianNB.pkl",
    "LinearSVM.pkl", "LogisticRegression.pkl",
]

# TM bundle source → target name + binarizer kind.
TM_BUNDLES = [
    ("model.pkl",          "model.pkl",          "GLADE"),
    ("model_kbins.pkl",    "model_kbins.pkl",    "KBins"),
    ("model_standard.pkl", "model_standard.pkl", "Standard"),
]


def _legacy_fbz_module():
    src = Path("/IoT/FuzzyPatternTM/examples/glade/benchmark/fcm_bitmask_zstd.py")
    spec = importlib.util.spec_from_file_location("fcm_bitmask_zstd", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fcm_bitmask_zstd"] = mod
    spec.loader.exec_module(mod)
    return mod


def _save_raw_estimator(src_path: Path, dst_path: Path) -> tuple[bool, str]:
    """Copy `src_path` into `dst_path` as a raw classifier.

    Paper_2 saves ML models as ``{"model": clf, "scaler": ..., "class_names": ...}``.
    fuzzy_tm_infer expects the raw classifier object, so we unwrap.
    """
    if not src_path.exists():
        return False, "missing"
    with src_path.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        clf = obj.get("model") or obj.get("clf")
        if clf is None:
            return False, "no 'model' key"
        with dst_path.open("wb") as f:
            pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)
        return True, type(clf).__name__
    # Already raw
    shutil.copy2(src_path, dst_path)
    return True, type(obj).__name__


def _save_scaler(loader_name: str, dst_path: Path) -> None:
    data = load_and_preprocess(loader_name)
    sc = StandardScaler().fit(data["X_train"])
    with dst_path.open("wb") as f:
        pickle.dump(sc, f, protocol=pickle.HIGHEST_PROTOCOL)


def _save_testset(loader_name: str, dst_path: Path) -> tuple[int, int]:
    data = load_and_preprocess(loader_name)
    X = data["X_test"].astype(np.float32, copy=False)
    y = np.asarray(data["y_test"], dtype=np.int32)
    np.savez(dst_path, X_te=X, Y_te=y)
    return X.shape


def _binarizer_to_state_dict(b) -> dict:
    """Return a portable dict that captures everything needed to redo
    `transform(X)` later, without pickling the class. ``GLADEBooleanizer``
    already has ``to_dict``; KBins / Standard expose ``to_dict`` too."""
    if hasattr(b, "to_dict"):
        d = b.to_dict()
        # Convert any numpy arrays in to_dict output to lists for portability
        return {k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in d.items()}
    raise TypeError(f"binarizer {type(b).__name__} has no to_dict()")


def _save_tm_bundle(src_path: Path, dst_path: Path, kind: str) -> bool:
    if not src_path.exists():
        return False
    with src_path.open("rb") as f:
        old = pickle.load(f)

    if "glade" in old:
        # Existing GLADE bundle stores GLADE state directly as a dict.
        state = dict(old["glade"])
        state = {k: (v.tolist() if hasattr(v, "tolist") else v)
                 for k, v in state.items()}
    elif "binarizer" in old:
        state = _binarizer_to_state_dict(old["binarizer"])
    else:
        return False

    new = {
        "tm_rules":        old["tm_rules"],
        "binarizer_kind":  kind,
        "binarizer_state": state,
    }
    with dst_path.open("wb") as f:
        pickle.dump(new, f, protocol=pickle.HIGHEST_PROTOCOL)
    return True


def main():
    legacy_fbz = _legacy_fbz_module()
    ML_DEST.mkdir(parents=True, exist_ok=True)
    TM_DEST.mkdir(parents=True, exist_ok=True)

    summary = []
    for loader, sid, human in DATASETS:
        stem = DATASET_STEM_MAP.get(sid, sid)
        src = Path(RESULTS_DIR) / sid / "models"
        dst = ML_DEST / stem

        # Wipe the dataset's old ml_models dir so we don't leave stale
        # files mixed with the new ones.
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)

        print(f"\n=== {human}  ({sid} → {stem}) ===")

        # 1. Raw ML estimators
        ok_count = 0
        for fn in ML_FILES:
            ok, msg = _save_raw_estimator(src / fn, dst / fn)
            if ok:
                print(f"  · {fn:<24} {msg}")
                ok_count += 1
            else:
                print(f"  · {fn:<24} SKIP ({msg})")

        # 2. Scaler + testset
        _save_scaler(loader, dst / "scaler.pkl")
        shape = _save_testset(loader, dst / "testset.npz")
        print(f"  · scaler.pkl")
        print(f"  · testset.npz  {shape}")

        # 3. TM bundles in portable dict-only schema
        for in_name, out_name, kind in TM_BUNDLES:
            ok = _save_tm_bundle(src / in_name, dst / out_name, kind)
            if ok:
                print(f"  · {out_name:<24} {kind}+FPTM (portable bundle)")
            else:
                print(f"  · {out_name:<24} SKIP")

        # 4. Legacy FBZ1 for fuzzy_tm_infer's TMModel.from_fbz
        rules_json = src / "tm_rules.json"
        glade_json = src / "glade.json"
        if rules_json.exists() and glade_json.exists():
            rules = json.loads(rules_json.read_text())
            glade_payload = json.loads(glade_json.read_text())
            class_names_str = [str(c) for c in rules.get("classes", [])]
            out_fbz = TM_DEST / f"{stem}_model.fbz"
            n_bytes = legacy_fbz.write_fbz(
                str(out_fbz), rules, glade_payload,
                class_names=class_names_str,
            )
            shutil.copy2(out_fbz, dst / "model.fbz")
            shutil.copy2(out_fbz, dst / f"{stem}_model.fbz")
            print(f"  · {stem}_model.fbz  (legacy FBZ1, {n_bytes/1024:.2f} KB)")

        summary.append((human, stem, ok_count, shape[0]))

    print(f"\n{'='*60}\n  SYNC SUMMARY\n{'='*60}")
    for human, stem, ml, n_test in summary:
        print(f"  {human:<18} → {stem:<8}  "
              f"{ml} ML estimators, 3 TM bundles, {n_test} test samples")
    print(f"\nDestination: {ML_DEST}")


if __name__ == "__main__":
    main()
