#!/usr/bin/env python3
"""Fit and persist the three booleanizers (GLADE, KBins, Standard) for
each dataset.

Outputs (per dataset, in results/<dataset>/models/):
    glade.pkl       — pickled GLADEBooleanizer  (ready for inference)
    kbins.pkl       — pickled KBinsBooleanizer
    standard.pkl    — pickled StandardBinarizer

Loading at inference time:

    import pickle
    with open("glade.pkl", "rb") as f:
        bin_ = pickle.load(f)
    X_bool = bin_.transform(X_raw)

A small report at results/SUMMARY_binarizers.txt records bit count
and file size for each (dataset, binarizer).
"""

from __future__ import annotations

import pickle
from pathlib import Path

from .booleanizers.glade import GLADEBooleanizer
from .booleanizers.kbins import KBinsBooleanizer
from .booleanizers.standard import StandardBinarizer
from .config import DATASETS, GLADE_N_BINS, RESULTS_DIR
from .data_loader import load_and_preprocess


# Keep parameters comparable to the binariser-comparison results used in
# the paper (Table tab:binarizer_comparison): n_bins = GLADE_N_BINS = 15.
KBINS_PARAMS = {"n_bins": GLADE_N_BINS, "strategy": "quantile"}
STANDARD_PARAMS = {"max_bits_per_feature": GLADE_N_BINS}


def _fit_save(name: str, binarizer, X_train, out_path: Path) -> tuple[int, int]:
    """Fit `binarizer` on `X_train` and pickle it to `out_path`.
    Returns (n_bits, file_size_bytes)."""
    binarizer.fit(X_train)
    with out_path.open("wb") as f:
        pickle.dump(binarizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    n_bits = (
        binarizer.n_bits if hasattr(binarizer, "n_bits")
        else binarizer.number_of_features
    )
    return int(n_bits), out_path.stat().st_size


def main():
    rows = []
    for loader_name, sid, human in DATASETS:
        data = load_and_preprocess(loader_name)
        X_train = data["X_train"]
        models_dir = Path(RESULTS_DIR) / sid / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {human} (raw features = {X_train.shape[1]}) ===")
        # GLADE
        n_g, sz_g = _fit_save("GLADE",
                              GLADEBooleanizer(n_bins=GLADE_N_BINS),
                              X_train, models_dir / "glade.pkl")
        # KBins (quantile strategy)
        n_k, sz_k = _fit_save("KBins",
                              KBinsBooleanizer(**KBINS_PARAMS),
                              X_train, models_dir / "kbins.pkl")
        # StandardBinarizer (TMU-style)
        n_s, sz_s = _fit_save("Standard",
                              StandardBinarizer(**STANDARD_PARAMS),
                              X_train, models_dir / "standard.pkl")

        rows.append((human, n_g, sz_g, n_k, sz_k, n_s, sz_s))
        print(f"  GLADE   : {n_g:>4} bits  {sz_g/1024:>7.2f} KB")
        print(f"  KBins   : {n_k:>4} bits  {sz_k/1024:>7.2f} KB")
        print(f"  Standard: {n_s:>4} bits  {sz_s/1024:>7.2f} KB")

    out = Path(RESULTS_DIR) / "SUMMARY_binarizers.txt"
    lines = [f"{'Dataset':<18} "
             f"{'GLADE bits':>10} {'GLADE KB':>9}  "
             f"{'KBins bits':>10} {'KBins KB':>9}  "
             f"{'Std bits':>9} {'Std KB':>8}"]
    lines.append("-" * len(lines[0]))
    for human, n_g, sz_g, n_k, sz_k, n_s, sz_s in rows:
        lines.append(
            f"{human:<18} "
            f"{n_g:>10d} {sz_g/1024:>9.2f}  "
            f"{n_k:>10d} {sz_k/1024:>9.2f}  "
            f"{n_s:>9d} {sz_s/1024:>8.2f}"
        )
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
