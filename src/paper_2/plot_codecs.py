#!/usr/bin/env python3
"""Codec comparison plotted as three separate IEEE TII-style line graphs.

Each metric (compressed size, compression time, decompression time) is
written to its own PDF/PNG so the paper can place them individually
where the discussion calls for them.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np

try:
    from .config import RESULTS_DIR
except ImportError:  # pragma: no cover - allow direct script execution
    from config import RESULTS_DIR

# Codecs ordered low → high compression effort
CODECS = ["zstd-1", "zstd-5", "gzip-9", "zstd-22", "lzma-9", "brotli-11"]

DATASET_HUMAN = {
    "wustl":   "WUSTL-EHMS-2020",
    "nslkdd":  "NSL-KDD",
    "ton_iot": "TON-IoT",
    "medsec":  "MedSec-25",
}
DATASET_ORDER = ["wustl", "nslkdd", "ton_iot", "medsec"]

# Saturated palette + distinct marker shapes (seaborn-darkgrid reference
# style: blue, green, red, gold — each dataset gets its own glyph so the
# series remain distinguishable in greyscale prints).
COLORS = ["#2E78B7", "#5DA64C", "#D04437", "#E2A52A"]
MARKERS = ["o", "s", "^", "D"]

PANELS = [
    ("ratio",     r"Compression ratio (raw / compressed)",
                                                   1.0,        "linear",
     "fig_codec_size"),
    ("comp_us",   r"Compression time ($\mu$s)",    1.0,        "log",
     "fig_codec_comp"),
    ("decomp_us", r"Decompression time ($\mu$s)",  1.0,        "log",
     "fig_codec_decomp"),
]


def _style():
    """Seaborn-darkgrid look: light grey axes panel, white grid,
    sans-serif typography. Implemented in pure matplotlib so we keep
    the dependency surface small."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Lato", "DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.labelsize": 11.5,
        "axes.labelweight": "normal",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        # Panel + grid (seaborn darkgrid replica)
        "axes.facecolor": "#EAEAF2",
        "axes.edgecolor": "white",
        "axes.linewidth": 0.0,
        "axes.grid": True,
        "grid.color": "white",
        "grid.linestyle": "-",
        "grid.linewidth": 1.0,
        "axes.axisbelow": True,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "xtick.bottom": False,
        "ytick.left": False,
        "axes.labelcolor": "#000000",
        "lines.linewidth": 1.6,
        "lines.markersize": 7,
        "lines.markeredgewidth": 0.0,
    })


def _plot_panel_matplotlib(idx: dict, key: str, ylabel: str, scale: float,
                           yscale: str, out_stem: str) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    x = np.arange(len(CODECS))
    for i, ds in enumerate(DATASET_ORDER):
        y = [idx[(ds, c)][key] * scale for c in CODECS]
        ax.plot(x, y, marker=MARKERS[i], color=COLORS[i],
                label=DATASET_HUMAN[ds],
                markeredgecolor=COLORS[i], markerfacecolor=COLORS[i])

    ax.set_xticks(x)
    ax.set_xticklabels(CODECS, rotation=0, ha="center")
    ax.set_xlabel("Compression method")
    ax.set_ylabel(ylabel + (" (log scale)" if yscale == "log" else ""))
    ax.set_yscale(yscale)

    # IEEE-tight y-axis. Reserve a bit of headroom for the inside-upper-
    # left legend without distorting the visual: log axes snap to decades,
    # linear axes get ~25 % top headroom (legend zone) and 8 % bottom.
    ymin = min(idx[(d, c)][key] * scale for d in DATASET_ORDER
               for c in CODECS)
    ymax = max(idx[(d, c)][key] * scale for d in DATASET_ORDER
               for c in CODECS)
    if yscale == "log":
        lo = 10 ** np.floor(np.log10(ymin))
        hi = 10 ** np.ceil(np.log10(ymax))
        # Half-decade headroom only when the data already touches the
        # next-decade boundary closely (so the inside-upper-left legend
        # stays clear of the highest curve).
        if np.log10(ymax) > np.log10(hi) - 0.25:
            hi *= 3.0
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    else:
        span = ymax - ymin
        ax.set_ylim(ymin - 0.08 * span, ymax + 0.25 * span)
    ax.set_xlim(-0.4, len(CODECS) - 0.6)

    # Hide the rectangular frame entirely (seaborn-darkgrid look)
    for s in ax.spines.values():
        s.set_visible(False)

    # Major ticks only, no tick marks (rcParams already turns them off)
    ax.minorticks_off()

    # Legend back inside the plot (upper-left), framed white panel —
    # matches the seaborn-darkgrid reference look the user approved.
    leg = ax.legend(loc="upper left", frameon=True, framealpha=0.95,
                    edgecolor="0.8", facecolor="white", fancybox=False,
                    handlelength=1.6, borderpad=0.5, borderaxespad=0.6)
    leg.get_frame().set_linewidth(0.6)
    fig.tight_layout()
    figs_dir = Path(__file__).parent / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs_dir / f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(figs_dir / f"{out_stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_panel_plotly(idx: dict, key: str, ylabel: str, scale: float,
                       yscale: str, out_stem: str) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("plotly is not installed") from exc

    x = list(range(len(CODECS)))
    fig = go.Figure()
    for i, ds in enumerate(DATASET_ORDER):
        y = [idx[(ds, c)][key] * scale for c in CODECS]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=DATASET_HUMAN[ds],
                line=dict(color=COLORS[i], width=1.8),
                marker=dict(size=7, color=COLORS[i]),
            )
        )

    ymin = min(idx[(d, c)][key] * scale for d in DATASET_ORDER for c in CODECS)
    ymax = max(idx[(d, c)][key] * scale for d in DATASET_ORDER for c in CODECS)
    yaxis = dict(
        title=ylabel,
        showline=True,
        mirror=True,
        linewidth=0.9,
        linecolor="black",
        ticks="outside",
        ticklen=4,
        tickwidth=0.9,
        tickcolor="black",
        showgrid=False,
    )
    if yscale == "log":
        lo = 10 ** math.floor(math.log10(ymin))
        hi = 10 ** math.ceil(math.log10(ymax))
        yaxis.update(type="log", range=[math.log10(lo), math.log10(hi)])

    fig.update_layout(
        width=660,
        height=340,
        font=dict(family="Times New Roman, Times, Nimbus Roman, DejaVu Serif", size=11),
        margin=dict(l=55, r=160, t=35, b=55),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=10),
        ),
        xaxis=dict(
            title="Compression method",
            tickmode="array",
            tickvals=x,
            ticktext=CODECS,
            tickangle=0,
            showline=True,
            mirror=True,
            linewidth=0.9,
            linecolor="black",
            ticks="outside",
            ticklen=4,
            tickwidth=0.9,
            tickcolor="black",
            showgrid=False,
            range=[-0.3, len(CODECS) - 0.7],
        ),
        yaxis=yaxis,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    figs_dir = Path(__file__).parent / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    fig.write_image(figs_dir / f"{out_stem}.pdf")
    fig.write_image(figs_dir / f"{out_stem}.png", scale=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot codec comparisons.")
    parser.add_argument(
        "--backend",
        choices=["matplotlib", "plotly", "both"],
        default="matplotlib",
        help="Rendering backend to use.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.backend in {"matplotlib", "both"}:
        _style()

    data = json.loads(Path(RESULTS_DIR, "SUMMARY_codecs.json").read_text())
    idx = {(r["dataset"], r["codec"]): r for r in data}
    for key, ylabel, scale, yscale, stem in PANELS:
        if args.backend in {"matplotlib", "both"}:
            _plot_panel_matplotlib(idx, key, ylabel, scale, yscale, stem)
            print(f"wrote figures/{stem}.pdf and .png")
        if args.backend in {"plotly", "both"}:
            plotly_stem = f"{stem}_plotly"
            try:
                _plot_panel_plotly(idx, key, ylabel, scale, yscale, plotly_stem)
            except RuntimeError as exc:
                print(f"plotly backend skipped: {exc}")
            else:
                print(f"wrote figures/{plotly_stem}.pdf and .png")


if __name__ == "__main__":
    main()
