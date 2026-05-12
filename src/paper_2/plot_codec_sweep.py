#!/usr/bin/env python3
"""Codec sweep → three YOLO-style line graphs for one representative dataset.

Reads ``results/SUMMARY_codecs_sweep.json`` (from ``bench_codecs_sweep.py``)
and renders, for **TON-IoT** (the largest model payload — a representative
mid-range case), three figures with **linear** axes:

    fig_codec_size_sweep.{pdf,png}      y = compressed model size (KB)
    fig_codec_comp_sweep.{pdf,png}      y = compression   time   (ms)
    fig_codec_decomp_sweep.{pdf,png}    y = decompression time   (ms)

x in all three = the codec's compression-level dial.  Each codec family
is one curve connecting its levels — the way the Ultralytics YOLO release
chart connects a model family's size variants (n/s/m/l/x).  Zstandard is
drawn bold (the codec the FBZ container ships with) and its default level,
zstd-1, is flagged.  Zstandard is shown for levels 1–10 (levels 11–22 only
plateau and would stretch the x-axis past every other family); the other
high-effort settings (brotli 9-11, lzma 7-9, gzip 8-9, all of bzip2) are
likewise excluded so the linear time axes stay readable.

All timings are in-memory (the codec ``compress``/``decompress`` calls run
on ``bytes`` in RAM, no disk I/O) so they reflect the codec, not the
filesystem.  The size values are the compressed byte counts — i.e. exactly
the size of the ``.fbz`` file on disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from .config import RESULTS_DIR
except ImportError:  # pragma: no cover - direct execution
    from config import RESULTS_DIR

# The one dataset shown (short id, human label).
DATASET = ("ton_iot", "TON-IoT")

# Family draw order, display name, colour, marker, highlight flag, and the
# kept level range (high-effort settings dropped — see module docstring).
FAMILIES = [
    # key       label              colour      marker  hi     levels
    ("zstd",   "Zstandard",       "#1F4E9C",  "o",    True,  range(1, 11)),
    ("brotli", "Brotli",          "#8E5BB5",  "D",    False, range(0, 9)),
    ("gzip",   "gzip (DEFLATE)",  "#5DA64C",  "s",    False, range(1, 8)),
    ("lzma",   "LZMA2",           "#D04437",  "^",    False, range(0, 7)),
]

# Default codec the paper actually ships with (gets a star + caption).
DEFAULT_FAMILY, DEFAULT_LEVEL = "zstd", 1

# Per-figure: metric key in the JSON, scale factor, y-axis label, file stem.
FIGURES = [
    ("size",      1 / 1024.0, "Compressed model size (kilobytes)",
     "fig_codec_size_sweep"),
    ("comp_us",   1 / 1000.0, "Compression time per model (milliseconds)",
     "fig_codec_comp_sweep"),
    ("decomp_us", 1 / 1000.0, "Decompression time per model (milliseconds)",
     "fig_codec_decomp_sweep"),
]


def _style() -> None:
    """Seaborn-darkgrid replica — matches the other Paper-2 figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Lato", "DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.labelsize": 11.5,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
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
        "lines.markeredgewidth": 0.0,
    })


def _index(sweep_path: Path) -> dict[tuple[str, str, int], dict]:
    rows = json.loads(sweep_path.read_text())
    return {(r["dataset"], r["family"], r["level"]): r for r in rows}


def _plot_one(idx, sid: str, human: str, metric: str, scale: float,
              ylabel: str, out_stem: str, figs_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    ys_all: list[float] = []
    handles: list = []

    for fkey, flabel, color, marker, hi, lvl_range in FAMILIES:
        levels = [l for l in lvl_range if (sid, fkey, l) in idx]
        if not levels:
            continue
        ys = [idx[(sid, fkey, l)][metric] * scale for l in levels]
        ys_all += ys
        lw = 2.8 if hi else 1.6
        ms = 8.5 if hi else 6.2
        z = 6 if hi else 4
        ax.plot(levels, ys, "-", color=color, linewidth=lw, zorder=z,
                solid_capstyle="round")
        h = ax.plot(levels, ys, marker, color=color, markersize=ms,
                    markeredgecolor="white",
                    markeredgewidth=0.9 if hi else 0.6,
                    zorder=z + 1, label=flabel)[0]
        handles.append(h)

    # Flag the codec setting the FBZ container actually ships with — a gold
    # star on the data point, explained in the legend (kept text-free so it
    # never collides with a nearby curve).
    if (sid, DEFAULT_FAMILY, DEFAULT_LEVEL) in idx:
        dy = idx[(sid, DEFAULT_FAMILY, DEFAULT_LEVEL)][metric] * scale
        ax.scatter([DEFAULT_LEVEL], [dy], s=300, marker="*",
                   color="#FFD23F", edgecolor="#5e4600", linewidth=1.0,
                   zorder=25)
        handles.append(Line2D([], [], marker="*", color="#FFD23F",
                              markeredgecolor="#5e4600", markeredgewidth=1.0,
                              linestyle="None", markersize=14,
                              label=f"FBZ default ({DEFAULT_FAMILY}-{DEFAULT_LEVEL})"))

    ax.set_xlabel("Compression level")
    ax.set_ylabel(ylabel)
    xs_seen = [l for _f, _l, _c, _m, _h, rng in FAMILIES for l in rng
               if (sid, _f, l) in idx]
    xhi = max(xs_seen)
    ax.set_xticks(range(0, xhi + 1, 1 if xhi <= 12 else 2))
    ax.set_xlim(min(xs_seen) - 0.5, xhi + 0.5)
    ymin, ymax = min(ys_all), max(ys_all)
    yspan = ymax - ymin or 1.0
    ax.set_ylim(max(0.0, ymin - 0.10 * yspan), ymax + 0.14 * yspan)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.minorticks_off()

    leg = ax.legend(handles=handles, loc="best", frameon=True,
                    framealpha=0.95, edgecolor="0.8", facecolor="white",
                    fancybox=False, handlelength=1.6, borderpad=0.5,
                    borderaxespad=0.6)
    leg.get_frame().set_linewidth(0.6)

    fig.tight_layout()
    figs_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs_dir / f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(figs_dir / f"{out_stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Codec sweep — 3 line graphs.")
    p.add_argument("--sweep", type=Path,
                   default=Path(RESULTS_DIR) / "SUMMARY_codecs_sweep.json")
    p.add_argument("--dataset", default=DATASET[0],
                   help="dataset short id to plot (default: ton_iot)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _style()
    idx = _index(args.sweep)
    sid = args.dataset
    human = {d[0]: d[1] for d in
             [("nslkdd", "NSL-KDD"), ("ton_iot", "TON-IoT"),
              ("medsec", "MedSec-25"), ("wustl", "WUSTL-EHMS-2020")]}.get(sid, sid)
    figs_dir = Path(__file__).parent / "figures"
    for metric, scale, ylabel, stem in FIGURES:
        _plot_one(idx, sid, human, metric, scale, ylabel, stem, figs_dir)
        print(f"wrote figures/{stem}.pdf and .png")


if __name__ == "__main__":
    main()
