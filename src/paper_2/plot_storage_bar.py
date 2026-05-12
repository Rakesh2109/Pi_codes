#!/usr/bin/env python3
"""Render a side-by-side bar chart comparing the GLADE+FPTM deployable
storage under two persistence formats:

  (B) model.pkl  — single uncompressed pickle bundle of GLADE + TM rules
  (C) model.fbz  — same content, zstd-1 compressed (FBZ container)

Numbers are taken directly from the saved files in
results/<dataset>/models/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import RESULTS_DIR

DATASETS = [
    ("wustl",   "WUSTL-EHMS-2020"),
    ("nslkdd",  "NSL-KDD"),
    ("ton_iot", "TON-IoT"),
    ("medsec",  "MedSec-25"),
]


def _kb(path: Path) -> float:
    return path.stat().st_size / 1024.0


def main():
    root = Path(RESULTS_DIR)
    sizes_b, sizes_c = [], []
    for sid, _ in DATASETS:
        m = root / sid / "models"
        sizes_b.append(_kb(m / "model.pkl"))
        sizes_c.append(_kb(m / "model.fbz"))

    labels = [d[1] for d in DATASETS]
    x = np.arange(len(labels))
    width = 0.36

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "axes.axisbelow": True,
    })

    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    bars_b = ax.bar(x - width / 2, sizes_b, width,
                    label="Uncompressed",
                    color="#5B8DEF", edgecolor="black", linewidth=0.7)
    bars_c = ax.bar(x + width / 2, sizes_c, width,
                    label="Compressed (zstd-1)",
                    color="#E89B5C", edgecolor="black", linewidth=0.7)

    for bar, val in zip(list(bars_b) + list(bars_c), sizes_b + sizes_c):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Model size (KB)")
    ax.set_ylim(0, max(sizes_b) * 1.20)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, handlelength=1.6,
              borderaxespad=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_pdf = Path(__file__).parent / "figures" / "fig_storage_bar.pdf"
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    print()
    print(f"{'Dataset':<18} {'B (KB)':>10} {'C (KB)':>10} {'C/B':>7}")
    for (sid, name), b, c in zip(DATASETS, sizes_b, sizes_c):
        print(f"{name:<18} {b:>10.2f} {c:>10.2f} {c / b:>6.2f}x")


if __name__ == "__main__":
    main()
