#!/usr/bin/env python3
"""Full-sweep compression benchmark for the GLADE+FPTM model bundle.

Unlike ``bench_codecs.py`` (which samples three zstd levels plus one
setting each of gzip/lzma/brotli), this script walks **every level** of
five lossless codec families so the trade-off curves can be drawn the
way the Ultralytics YOLO Pareto chart draws model families:

    family   levels
    ──────   ──────────────────────────────
    zstd     1 … 22         (zstandard)
    gzip     1 … 9          (DEFLATE)
    lzma     0 … 9          (LZMA2, stdlib)
    brotli   0 … 11         (Google brotli, quality)
    bz2      1 … 9          (bzip2, stdlib)

For each (dataset, family, level) we measure the compressed size, the
compression ratio (raw / compressed), and the compression and
decompression latencies in microseconds.  The same in-memory FBZ
payload (GLADE thresholds + TM clause bitmasks, codec=0) is round-
tripped through ``BytesIO`` so the timings reflect the codec only.

Outputs:
    results/SUMMARY_codecs_sweep.json   one row per (dataset, family, level)
"""

from __future__ import annotations

import bz2
import gzip
import json
import lzma
import time
from pathlib import Path
from typing import Callable

import brotli
import zstandard as zstd

from . import fbz as fbz_mod
from .config import DATASETS, RESULTS_DIR


# ── Codec families: name -> (levels, encoder factory, decoder) ─────
def _zstd_enc(level: int) -> Callable[[bytes], bytes]:
    c = zstd.ZstdCompressor(level=level)
    return lambda b: c.compress(b)


_ZSTD_DEC = zstd.ZstdDecompressor()

FAMILIES: dict[str, dict] = {
    "zstd": {
        "levels": list(range(1, 23)),               # 1 … 22
        "enc": _zstd_enc,
        "dec": lambda b: _ZSTD_DEC.decompress(b),
        "marker": "o",
    },
    "gzip": {
        "levels": list(range(1, 10)),               # 1 … 9
        "enc": lambda lvl: (lambda b: gzip.compress(b, compresslevel=lvl)),
        "dec": gzip.decompress,
        "marker": "s",
    },
    "lzma": {
        "levels": list(range(0, 10)),               # presets 0 … 9
        "enc": lambda lvl: (lambda b: lzma.compress(b, preset=lvl)),
        "dec": lzma.decompress,
        "marker": "^",
    },
    "brotli": {
        "levels": list(range(0, 12)),               # quality 0 … 11
        "enc": lambda lvl: (lambda b: brotli.compress(b, quality=lvl)),
        "dec": brotli.decompress,
        "marker": "D",
    },
    "bz2": {
        "levels": list(range(1, 10)),               # 1 … 9
        "enc": lambda lvl: (lambda b: bz2.compress(b, compresslevel=lvl)),
        "dec": bz2.decompress,
        "marker": "v",
    },
}

ITERS = 200
SLOW_ITERS = 20            # high-effort settings: fewer iterations
WARMUP = 5
# (family, level >= threshold) -> use SLOW_ITERS for the compress loop
SLOW_FROM = {"zstd": 19, "lzma": 6, "brotli": 6, "bz2": 1, "gzip": 9}


def _avg_us(fn: Callable[[], object], iters: int) -> float:
    for _ in range(WARMUP):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1e6 / iters


def _build_payload(short_id: str) -> bytes:
    """The FBZ uncompressed payload — GLADE + TM clauses, fixed layout,
    header stripped (the canonical bytes every codec sees)."""
    m = Path(RESULTS_DIR) / short_id / "models"
    rules = json.loads((m / "tm_rules.json").read_text())
    glade = json.loads((m / "glade.json").read_text())
    raw = fbz_mod.pack(rules, glade=glade, level=0)
    return raw[fbz_mod.HEADER_SIZE:]


def _measure(payload: bytes, family: str, level: int) -> dict:
    spec = FAMILIES[family]
    enc = spec["enc"](level)
    dec = spec["dec"]
    blob = enc(payload)
    if dec(blob) != payload:
        raise AssertionError(f"{family}-{level}: round-trip mismatch")
    comp_iters = SLOW_ITERS if level >= SLOW_FROM.get(family, 99) else ITERS
    comp_us = _avg_us(lambda: enc(payload), comp_iters)
    decomp_us = _avg_us(lambda: dec(blob), ITERS)
    return {
        "size": len(blob),
        "comp_us": comp_us,
        "decomp_us": decomp_us,
        "ratio": round(len(payload) / len(blob), 4),
    }


def main() -> None:
    rows: list[dict] = []
    hdr = (f"{'dataset':<10} {'family':<7} {'lvl':>3} "
           f"{'size B':>8} {'ratio':>7} {'comp us':>10} {'dec us':>9}")
    print(hdr)
    print("-" * len(hdr))
    for _loader, sid, human in DATASETS:
        payload = _build_payload(sid)
        raw_bytes = len(payload)
        for family, spec in FAMILIES.items():
            for level in spec["levels"]:
                r = _measure(payload, family, level)
                rows.append({
                    "dataset": sid, "human": human,
                    "family": family, "level": level,
                    "label": f"{family}-{level}",
                    "raw_bytes": raw_bytes, **r,
                })
                print(f"{sid:<10} {family:<7} {level:>3d} "
                      f"{r['size']:>8d} {r['ratio']:>6.2f}x "
                      f"{r['comp_us']:>10.1f} {r['decomp_us']:>9.1f}")

    out = Path(RESULTS_DIR) / "SUMMARY_codecs_sweep.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
