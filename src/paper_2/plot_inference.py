#!/usr/bin/env python3
"""Per-sample inference-latency comparison across all baselines and
GLADE+FPTM, with F1 sourced from the trained-paper reports.

Latency comes from SUMMARY_inference_all.json (host-PC, fuzzy_tm_infer
algorithms) and F1 comes from results/<dataset>/reports/<model>.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import RESULTS_DIR

# Display order, left to right (slow -> fast)
MODELS = [
    "XGBoost", "MLP_2C", "MLP_med", "MLP_small", "MLP_tiny",
    "GLADE+FPTM", "LogisticRegression", "LinearSVM",
    "DecisionTree", "GaussianNB",
]

# Map plotting name -> (folder suffix, report file stem)
F1_SOURCE = {
    "XGBoost":             ("",          "XGBoost"),
    "MLP_2C":              ("_tmmatched","MLP_2C"),
    "MLP_med":             ("",          "MLP_med"),
    "MLP_small":           ("",          "MLP_small"),
    "MLP_tiny":            ("",          "MLP_tiny"),
    "GLADE+FPTM":          ("",          "GLADE_FPTM"),
    "LogisticRegression":  ("",          "LogisticRegression"),
    "LinearSVM":           ("",          "LinearSVM"),
    "DecisionTree":        ("",          "DecisionTree"),
    "GaussianNB":          ("",          "GaussianNB"),
}

DATASETS = [
    ("wustl",   "WUSTL-EHMS-2020"),
    ("nslkdd",  "NSL-KDD"),
    ("toniot",  "TON-IoT"),  # bench uses 'toniot'
    ("medsec",  "MedSec-25"),
]
# Map for paper_2 reports dir (uses ton_iot)
DATASET_REPORT_DIR = {"wustl": "wustl", "nslkdd": "nslkdd",
                      "toniot": "ton_iot", "medsec": "medsec"}


def _load_latency():
    rows = json.loads(Path(RESULTS_DIR, "SUMMARY_inference_all.json").read_text())
    out = {}
    for r in rows:
        out[(r["dataset"], r["model"])] = r["us_per_sample"]
    return out


def _load_paper_f1(stem: str, model: str) -> float | None:
    suffix, fstem = F1_SOURCE[model]
    sid = DATASET_REPORT_DIR.get(stem, stem)
    p = Path(RESULTS_DIR) / f"{sid}{suffix}" / "reports" / f"{fstem}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return float(d["metrics"]["macro_f1"])


def main():
    lat = _load_latency()

    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.labelsize": 11, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 10,
        "axes.axisbelow": True,
    })
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharey=False)

    for ax, (stem, human) in zip(axes.ravel(), DATASETS):
        # Filter to models that have latency data and F1 from paper reports
        names, lats, f1s = [], [], []
        for m in MODELS:
            v = lat.get((stem, m))
            if v is None:
                continue
            f1 = _load_paper_f1(stem, m)
            if f1 is None:
                continue
            names.append(m); lats.append(v); f1s.append(f1)

        # Sort by latency ascending
        order = np.argsort(lats)
        names = [names[i] for i in order]
        lats = [lats[i] for i in order]
        f1s = [f1s[i] for i in order]

        x = np.arange(len(names))
        # Highlight GLADE+FPTM
        colours = ["#E89B5C" if n == "GLADE+FPTM" else "#5B8DEF" for n in names]
        bars = ax.bar(x, lats, color=colours, edgecolor="black",
                      linewidth=0.5)
        # Label each bar with F1
        for i, (bar, f1) in enumerate(zip(bars, f1s)):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.05,
                    f"F1={f1:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9.5)
        ax.set_yscale("log")
        ax.set_ylabel(r"$\mu$s / sample (log)")
        ax.set_title(human, fontsize=11.5, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_pdf = Path(__file__).parent / "figures" / "fig_inference.pdf"
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
