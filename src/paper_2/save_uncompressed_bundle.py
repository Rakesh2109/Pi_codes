#!/usr/bin/env python3
"""Save a single uncompressed pickle bundle for each dataset that holds the
SAME content as the .fbz (GLADE thresholds + TM rules), so we can compare
plain-pickle vs zstd-1 sizes apples-to-apples.

The bundle dict layout:
    {
        "tm_rules": <tm_rules.json contents>,
        "glade":    <glade.json contents>,
    }
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from .config import DATASETS, RESULTS_DIR


def main():
    root = Path(RESULTS_DIR)
    print(f"{'Dataset':<10} {'pkl bundle (B)':>15} {'.fbz (B)':>10} "
          f"{'pkl KB':>9} {'fbz KB':>9}  {'fbz/pkl':>8}")
    for _loader, sid, _human in DATASETS:
        m = root / sid / "models"
        bundle = {
            "tm_rules": json.loads((m / "tm_rules.json").read_text()),
            "glade":    json.loads((m / "glade.json").read_text()),
        }
        out = m / "model.pkl"
        with out.open("wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        pkl_bytes = out.stat().st_size
        fbz_bytes = (m / "model.fbz").stat().st_size
        print(f"{sid:<10} {pkl_bytes:>15d} {fbz_bytes:>10d} "
              f"{pkl_bytes / 1024:>9.2f} {fbz_bytes / 1024:>9.2f}  "
              f"{fbz_bytes / pkl_bytes:>7.2f}x")


if __name__ == "__main__":
    main()
