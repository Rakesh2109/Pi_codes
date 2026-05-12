#!/usr/bin/env python3
"""Compression-method benchmark for the GLADE+FPTM model bundle.

Same payload (the FBZ uncompressed body — GLADE thresholds + TM clauses)
is round-tripped through six compression methods using ``BytesIO`` so
the timings reflect the codec only, not disk I/O.

Methods (configured once in ``CODECS``; the benchmark loop is method-
agnostic):
    - zstd-1, zstd-5, zstd-22  : the spectrum of zstandard levels
    - gzip-9                   : DEFLATE, ubiquitous baseline
    - lzma-9                   : LZMA, best-ratio stdlib codec
    - brotli-11                : modern web codec, max quality

Outputs:
    results/SUMMARY_codecs.json    machine-readable, per-(dataset, codec)
    results/SUMMARY_codecs.txt     human-readable table
"""

from __future__ import annotations

import bz2
import gzip
import io
import json
import lzma
import time
from pathlib import Path
from typing import Callable

import brotli
import zstandard as zstd

from . import fbz as fbz_mod
from .config import DATASETS, RESULTS_DIR


# ── Single source of truth for every codec ─────────────────────────
def _zstd(level: int):
    cctx = zstd.ZstdCompressor(level=level)
    dctx = zstd.ZstdDecompressor()
    return (lambda b, c=cctx: c.compress(b),
            lambda b, d=dctx: d.decompress(b))


CODECS: dict[str, tuple[Callable, Callable]] = {
    "zstd-1":    _zstd(1),
    "zstd-5":    _zstd(5),
    "zstd-22":   _zstd(22),
    "gzip-9":    (lambda b: gzip.compress(b, compresslevel=9),  gzip.decompress),
    "lzma-9":    (lambda b: lzma.compress(b, preset=9),         lzma.decompress),
    "brotli-11": (lambda b: brotli.compress(b, quality=11),     brotli.decompress),
}

ITERS = 200
WARMUP = 5


def _avg_us(fn, iters: int = ITERS) -> float:
    for _ in range(WARMUP):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1e6 / iters


def _measure(payload: bytes, codec_name: str) -> dict:
    """Return {'size','comp_us','decomp_us'} for one (payload, codec)."""
    enc, dec = CODECS[codec_name]
    blob = enc(payload)
    iters = ITERS // 10 if codec_name in ("zstd-22", "lzma-9", "brotli-11") else ITERS
    comp_us = _avg_us(lambda: enc(payload), iters)
    decomp_us = _avg_us(lambda: dec(blob), ITERS)
    out = dec(blob)
    if out != payload:
        raise AssertionError(f"{codec_name}: round-trip mismatch")
    return {"size": len(blob), "comp_us": comp_us, "decomp_us": decomp_us}


def _build_payload(short_id: str) -> bytes:
    """The FBZ uncompressed payload — GLADE + TM clauses, in fixed layout."""
    m = Path(RESULTS_DIR) / short_id / "models"
    rules = json.loads((m / "tm_rules.json").read_text())
    glade = json.loads((m / "glade.json").read_text())
    # Re-pack with codec=0 (no compression) and strip the 12-byte header,
    # leaving only the canonical bytes that all codecs see.
    raw = fbz_mod.pack(rules, glade=glade, level=0)
    return raw[fbz_mod.HEADER_SIZE:]


def main():
    rows = []
    print(f"{'Dataset':<10} {'Codec':<10} "
          f"{'size B':>8} {'comp us':>9} {'dec us':>8} {'ratio':>6}")
    print("-" * 56)
    for _loader, sid, human in DATASETS:
        payload = _build_payload(sid)
        for codec in CODECS:
            r = _measure(payload, codec)
            ratio = len(payload) / r["size"]
            rows.append({"dataset": sid, "human": human, "codec": codec,
                         "raw_bytes": len(payload), **r,
                         "ratio": round(ratio, 3)})
            print(f"{sid:<10} {codec:<10} "
                  f"{r['size']:>8d} {r['comp_us']:>9.0f} "
                  f"{r['decomp_us']:>8.0f} {ratio:>5.2f}x")

    out = Path(RESULTS_DIR) / "SUMMARY_codecs.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
