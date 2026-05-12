#!/usr/bin/env python3
"""Benchmark raw vs zstd-1 compression for TM rule artifacts.

Outputs a summary table for:
  1) Pickle bytes (tm_rules.json -> pickle.dumps)
  2) Compact bitmask block (lossless clause masks, FBZ-like block)
"""

import json
import os
import pickle
import time
from pathlib import Path

import zstandard as zstd

from .config import DATASETS, RESULTS_DIR, TM_PARAMS_PER_DATASET

ZSTD_LEVEL = 1
ITERS = 200


def _avg_us(fn, iters):
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e6 / iters


def _benchmark_zstd(data: bytes, level: int, iters: int):
    cctx = zstd.ZstdCompressor(level=level)
    dctx = zstd.ZstdDecompressor()

    comp = cctx.compress(data)

    comp_us = _avg_us(lambda: cctx.compress(data), iters)
    decomp_us = _avg_us(lambda: dctx.decompress(comp), iters)

    return comp, comp_us, decomp_us


def _build_bitmask_block(tm_rules: dict) -> bytes:
    """Build the raw clause bitmask block (FBZ-like), lossless.

    Layout per class, per polarity:
      u16 n_clauses
      for each clause:
        u8 clamp
        pos_mask [chunk_bytes]
        neg_mask [chunk_bytes]
    """
    n_bits = int(tm_rules["n_bits"])
    chunk_bytes = (n_bits + 7) // 8

    if isinstance(tm_rules.get("classes"), list):
        class_order = list(tm_rules["classes"])
        cls_table = tm_rules["class_rules"]
        pos_key, neg_key = "positive_clauses", "negative_clauses"
        inc_key, exc_key = "include", "exclude"
    else:
        cls_table = tm_rules["classes"]
        class_order = list(cls_table.keys())
        pos_key, neg_key = "positive", "negative"
        inc_key, exc_key = "include", "include_inverted"

    clamp_max = int((tm_rules.get("config") or {}).get("LF")
                    or tm_rules.get("LF") or 15)

    out = bytearray()
    for cls in class_order:
        spec = cls_table.get(cls) or cls_table.get(str(cls))
        for pol_key in (pos_key, neg_key):
            clauses = spec.get(pol_key, [])
            out += int(len(clauses)).to_bytes(2, "little", signed=False)
            for cl in clauses:
                clamp = min(int(cl.get("clamp") or clamp_max), 255)
                out.append(clamp)
                pos_mask = bytearray(chunk_bytes)
                neg_mask = bytearray(chunk_bytes)
                for i in (cl.get(inc_key) or []):
                    pos_mask[i >> 3] |= 1 << (i & 7)
                for i in (cl.get(exc_key) or []):
                    neg_mask[i >> 3] |= 1 << (i & 7)
                out += pos_mask + neg_mask
    return bytes(out)


def _fmt_row(ds, raw_b, comp_b, comp_us, decomp_us):
    ratio = raw_b / comp_b if comp_b else 0
    return (
        f"{ds:<10} "
        f"{raw_b:>9} B  "
        f"{comp_b:>9} B  "
        f"{comp_us:>8.0f} us  "
        f"{decomp_us:>8.0f} us  "
        f"{ratio:>5.2f}x"
    )


def main():
    results_root = Path(RESULTS_DIR)

    lines = []
    def wr(s=""):
        lines.append(s)

    wr("RLE vs zstd-1 (raw vs compressed) — measured results")
    wr("Note: raw rows show size only; zstd rows show avg over "
       f"{ITERS} iterations.")
    wr("")

    # Section 1: Pickle bytes
    wr("TM rules pickle bytes")
    wr(f"{'Dataset':<10} {'Raw':>11}  {'zstd-1':>11}  "
       f"{'Compress':>9}  {'Decompress':>9}  {'Ratio':>6}")
    wr("-" * 68)
    for _loader, sid, _human in DATASETS:
        tm_rules = json.load(open(results_root / sid / "models" / "tm_rules.json"))
        raw = pickle.dumps(tm_rules, protocol=pickle.HIGHEST_PROTOCOL)
        comp, c_us, d_us = _benchmark_zstd(raw, ZSTD_LEVEL, ITERS)
        wr(_fmt_row(sid, len(raw), len(comp), c_us, d_us))
    wr("")

    # Section 2: Compact bitmask block
    wr("TM compact bitmask block (FBZ-like, no quantization)")
    wr(f"{'Dataset':<10} {'Raw':>11}  {'zstd-1':>11}  "
       f"{'Compress':>9}  {'Decompress':>9}  {'Ratio':>6}")
    wr("-" * 68)
    for _loader, sid, _human in DATASETS:
        tm_rules = json.load(open(results_root / sid / "models" / "tm_rules.json"))
        raw = _build_bitmask_block(tm_rules)
        comp, c_us, d_us = _benchmark_zstd(raw, ZSTD_LEVEL, ITERS)
        wr(_fmt_row(sid, len(raw), len(comp), c_us, d_us))
    wr("")

    out_path = results_root / "SUMMARY_compression.txt"
    out_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
