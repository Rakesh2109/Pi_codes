#!/usr/bin/env python3
"""FBZ — FPTM Bitmask Zstd container (v3, lean header).

A small, self-describing, cross-language binary format that stores
**both** the GLADE binariser thresholds and the FPTM clause bitmasks
in a single zstd-compressed file.  A reader only needs zstd plus
little-endian struct unpacking.

File layout (all multi-byte integers little-endian, unsigned)
=============================================================

Header (12 bytes):
  off  size  field
  0    4     magic        = b"FBZ1"
  4    1     version      = 3
  5    1     packed       =  bit 7..3  zstd level (0..31; 0 = uncompressed)
                             bit 2     reserved
                             bit 1     label_kind (0 = int32, 1 = utf-8)
                             bit 0     has_glade
  6    2     n_bits       (u16)
  8    2     n_classes    (u16)
  10   1     clamp_max    (u8)
  11   1     reserved     = 0

The compressed payload follows the header. Its size is derivable
(file_size - 12); its uncompressed size is read from the zstd frame
header by `ZSTD_getFrameContentSize` (the Python `zstandard` library
includes content size in the frame by default).

Compressed payload (zstd):
  if has_glade:
    u32           n_features_in
    i32[n_bits]   feat_idx       # which raw input column each bit reads
    f32[n_bits]   thresh         # `bit_j = X[feat_idx[j]] >= thresh[j]`

  # class labels in declaration order
  if label_kind == int32:
    i32[n_classes] labels
  else:
    for c in 0..n_classes-1:
      u16 len; bytes[len] utf8

  # clause bitmasks
  chunk_bytes = ceil(n_bits / 8)
  for c in 0..n_classes-1:
    for polarity in (positive, negative):
      u16 n_clauses
      for k in 0..n_clauses-1:
        u8                 clamp                 # per-clause LF cap
        bytes[chunk_bytes] include_mask          # literal x_j
        bytes[chunk_bytes] exclude_mask          # literal !x_j

Inference matches `vote()` in FuzzyPatternTM/src/Tsetlin.jl:
  for each clause k of class c:
      v = |{ i in include_k : x_i==0 }| + |{ i in exclude_k : x_i==1 }|
      out_k = max(clamp_k - v, 0)
  pos_vote_c = sum_k out_k (positive clauses)
  neg_vote_c = sum_k out_k (negative clauses)
  y_hat = argmax_c (pos_vote_c - neg_vote_c)
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import zstandard as zstd

MAGIC = b"FBZ1"
VERSION = 3
HEADER_SIZE = 12
HEADER_FMT = "<4sBB HH BB"  # 4+1+1+2+2+1+1 = 12
FLAG_HAS_GLADE = 0x01
FLAG_LABEL_UTF8 = 0x02
LEVEL_SHIFT = 3              # bits 7..3 of the packed byte hold the zstd level
LEVEL_MASK = 0x1F            # 5-bit level field (0..31)


def _normalise_rules(tm_rules: dict) -> dict:
    n_bits = int(tm_rules["n_bits"])
    clamp_max = int((tm_rules.get("config") or {}).get("LF")
                    or tm_rules.get("LF") or 15)

    if isinstance(tm_rules.get("classes"), list):
        labels = list(tm_rules["classes"])
        cls_table = tm_rules["class_rules"]
        pos_key, neg_key = "positive_clauses", "negative_clauses"
        inc_key, exc_key = "include", "exclude"
    else:
        cls_table = tm_rules["classes"]
        labels = list(cls_table.keys())
        pos_key, neg_key = "positive", "negative"
        inc_key, exc_key = "include", "include_inverted"

    rules: dict[Any, dict[str, list]] = {}
    for lbl in labels:
        spec = cls_table.get(lbl) or cls_table.get(str(lbl))
        rules[lbl] = {
            "positive": [
                {"clamp": int(c.get("clamp") or clamp_max),
                 "include": list(c.get(inc_key) or []),
                 "exclude": list(c.get(exc_key) or [])}
                for c in (spec.get(pos_key) or [])
            ],
            "negative": [
                {"clamp": int(c.get("clamp") or clamp_max),
                 "include": list(c.get(inc_key) or []),
                 "exclude": list(c.get(exc_key) or [])}
                for c in (spec.get(neg_key) or [])
            ],
        }
    return {"n_bits": n_bits, "clamp_max": clamp_max,
            "labels": labels, "rules": rules}


def _normalise_glade(glade_payload: dict) -> dict:
    """Pull (n_features_in, feat_idx, thresh) out of a glade.json dict."""
    if glade_payload.get("quantised", False):
        q = np.asarray(glade_payload["thresh_q"], dtype=np.float64)
        scale = float(glade_payload["thresh_scale"])
        zp = float(glade_payload["thresh_zp"])
        thresh = (q * scale + zp).astype(np.float32)
    else:
        thresh = np.asarray(glade_payload["thresh"], dtype=np.float32)
    return {
        "n_features_in": int(glade_payload["n_features_in"]),
        "feat_idx": np.asarray(glade_payload["feat_idx"], dtype=np.int32),
        "thresh": thresh,
        "n_bits_param": int(glade_payload["n_bits"]),
        "n_bins_param": int(glade_payload.get("n_bins_param", 15)),
    }


def _encode_payload(view: dict, glade: dict | None) -> tuple[bytes, int]:
    n_bits = view["n_bits"]
    clamp_max = view["clamp_max"]
    labels = view["labels"]
    rules = view["rules"]
    chunk_bytes = (n_bits + 7) // 8
    flags = 0
    if not all(isinstance(l, int) for l in labels):
        flags |= FLAG_LABEL_UTF8

    buf = bytearray()

    # GLADE section
    if glade is not None:
        flags |= FLAG_HAS_GLADE
        if glade["feat_idx"].size != n_bits or glade["thresh"].size != n_bits:
            raise ValueError("GLADE feat_idx/thresh length mismatch with n_bits")
        buf += struct.pack("<I", glade["n_features_in"])
        buf += glade["feat_idx"].astype("<i4", copy=False).tobytes(order="C")
        buf += glade["thresh"].astype("<f4", copy=False).tobytes(order="C")

    # Class labels
    if flags & FLAG_LABEL_UTF8:
        for lbl in labels:
            s = str(lbl).encode("utf-8")
            buf += struct.pack("<H", len(s)) + s
    else:
        for lbl in labels:
            buf += struct.pack("<i", int(lbl))

    # Clauses
    for lbl in labels:
        spec = rules[lbl]
        for pol in ("positive", "negative"):
            clauses = spec[pol]
            buf += struct.pack("<H", len(clauses))
            for cl in clauses:
                clamp = min(int(cl["clamp"]) if cl["clamp"] is not None else clamp_max, 255)
                buf.append(clamp)
                pos_mask = bytearray(chunk_bytes)
                neg_mask = bytearray(chunk_bytes)
                for i in cl["include"]:
                    pos_mask[i >> 3] |= 1 << (i & 7)
                for i in cl["exclude"]:
                    neg_mask[i >> 3] |= 1 << (i & 7)
                buf += pos_mask + neg_mask

    return bytes(buf), flags


def pack(tm_rules: dict, glade: dict | None = None, level: int = 1) -> bytes:
    """Serialise FPTM model (and optionally GLADE) to FBZ bytes.

    `tm_rules` : dict from tm_rules.json (Julia output)
    `glade`    : dict from glade.json    (GLADE.to_dict())
    `level`    : zstd level (0 = uncompressed)
    """
    view = _normalise_rules(tm_rules)
    g = _normalise_glade(glade) if glade is not None else None
    if g is not None and g["n_bits_param"] != view["n_bits"]:
        raise ValueError(
            f"GLADE n_bits ({g['n_bits_param']}) != TM n_bits ({view['n_bits']})"
        )

    payload, flags = _encode_payload(view, g)

    if level < 0 or level > LEVEL_MASK:
        raise ValueError(f"zstd level out of range (0..{LEVEL_MASK})")
    if level > 0:
        comp = zstd.ZstdCompressor(level=level).compress(payload)
    else:
        comp = payload

    n_classes = len(view["labels"])
    if n_classes > 0xFFFF or view["n_bits"] > 0xFFFF:
        raise ValueError("n_bits or n_classes exceed u16 range")

    packed = ((level & LEVEL_MASK) << LEVEL_SHIFT) | flags
    header = struct.pack(
        HEADER_FMT,
        MAGIC, VERSION, packed,
        view["n_bits"], n_classes,
        view["clamp_max"], 0,
    )
    return header + comp


def unpack(blob: bytes) -> dict:
    if len(blob) < HEADER_SIZE or blob[:4] != MAGIC:
        raise ValueError("not an FBZ file (bad magic)")

    (magic, version, packed,
     n_bits, n_classes,
     clamp_max, _r) = struct.unpack(HEADER_FMT, blob[:HEADER_SIZE])

    if version != VERSION:
        raise ValueError(f"unsupported FBZ version {version}")

    level = (packed >> LEVEL_SHIFT) & LEVEL_MASK
    flags = packed & 0x07

    body = blob[HEADER_SIZE:]
    if level > 0:
        payload = zstd.ZstdDecompressor().decompress(body)
    else:
        payload = body

    p = 0
    out: dict[str, Any] = {
        "version": version,
        "codec": "zstd" if level > 0 else "none",
        "codec_level": level,
        "n_bits": n_bits, "clamp_max": clamp_max,
    }

    # GLADE
    if flags & FLAG_HAS_GLADE:
        (n_features_in,) = struct.unpack_from("<I", payload, p); p += 4
        feat_idx = np.frombuffer(payload, dtype="<i4", count=n_bits, offset=p)
        p += 4 * n_bits
        thresh = np.frombuffer(payload, dtype="<f4", count=n_bits, offset=p)
        p += 4 * n_bits
        out["glade"] = {
            "n_features_in": n_features_in,
            "feat_idx": feat_idx.copy(),
            "thresh": thresh.copy(),
        }

    # Class labels
    label_utf8 = bool(flags & FLAG_LABEL_UTF8)
    labels: list = []
    if label_utf8:
        for _ in range(n_classes):
            (ln,) = struct.unpack_from("<H", payload, p); p += 2
            labels.append(payload[p:p + ln].decode("utf-8")); p += ln
    else:
        for _ in range(n_classes):
            (lbl,) = struct.unpack_from("<i", payload, p); p += 4
            labels.append(lbl)
    out["labels"] = labels

    # Clauses
    chunk_bytes = (n_bits + 7) // 8
    rules: dict = {}
    for lbl in labels:
        spec = {"positive": [], "negative": []}
        for pol in ("positive", "negative"):
            (n_cl,) = struct.unpack_from("<H", payload, p); p += 2
            for _ in range(n_cl):
                clamp = payload[p]; p += 1
                pos_mask = payload[p:p + chunk_bytes]; p += chunk_bytes
                neg_mask = payload[p:p + chunk_bytes]; p += chunk_bytes
                spec[pol].append({
                    "clamp": clamp,
                    "include": _bits(pos_mask, n_bits),
                    "exclude": _bits(neg_mask, n_bits),
                })
        rules[lbl] = spec
    out["rules"] = rules
    return out


def _bits(mask: bytes, n_bits: int) -> list[int]:
    out = []
    for i in range(n_bits):
        if mask[i >> 3] & (1 << (i & 7)):
            out.append(i)
    return out


def write(path: Path | str, tm_rules: dict,
          glade: dict | None = None, level: int = 1) -> int:
    blob = pack(tm_rules, glade=glade, level=level)
    Path(path).write_bytes(blob)
    return len(blob)


def read(path: Path | str) -> dict:
    return unpack(Path(path).read_bytes())


def transform(model: dict, X: np.ndarray) -> np.ndarray:
    """Apply the bundled GLADE thresholds: returns boolean (uint8) matrix."""
    if "glade" not in model:
        raise ValueError("FBZ has no GLADE section — cannot binarise raw input")
    g = model["glade"]
    X = np.asarray(X, dtype=np.float64)
    if X.shape[1] != g["n_features_in"]:
        raise ValueError(
            f"expected {g['n_features_in']} features, got {X.shape[1]}")
    return (X[:, g["feat_idx"]] >= g["thresh"][None, :]).astype(np.uint8)


def _cli():
    ap = argparse.ArgumentParser(
        description="Pack/unpack FPTM models to .fbz (zstd-1 by default)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_pack = sub.add_parser("pack", help="JSON tm_rules (+ glade) -> .fbz")
    p_pack.add_argument("input", type=Path, help="path to tm_rules.json")
    p_pack.add_argument("-o", "--output", type=Path, required=True)
    p_pack.add_argument("-g", "--glade", type=Path, default=None,
                        help="path to glade.json (bundles binariser thresholds)")
    p_pack.add_argument("-l", "--level", type=int, default=1,
                        help="zstd level (0 = uncompressed; default: 1)")

    p_unpack = sub.add_parser("unpack", help=".fbz -> JSON")
    p_unpack.add_argument("input", type=Path)
    p_unpack.add_argument("-o", "--output", type=Path, required=True)

    p_info = sub.add_parser("info", help="print FBZ header + sections")
    p_info.add_argument("input", type=Path)

    args = ap.parse_args()

    if args.cmd == "pack":
        rules = json.loads(args.input.read_text())
        glade_payload = json.loads(args.glade.read_text()) if args.glade else None
        n = write(args.output, rules, glade=glade_payload, level=args.level)
        print(f"wrote {args.output} ({n} bytes, zstd-{args.level}, "
              f"{'GLADE+TM' if glade_payload else 'TM only'})")

    elif args.cmd == "unpack":
        view = read(args.input)
        # Make GLADE arrays JSON-serialisable
        if "glade" in view:
            view["glade"]["feat_idx"] = view["glade"]["feat_idx"].tolist()
            view["glade"]["thresh"] = [float(x) for x in view["glade"]["thresh"]]
        args.output.write_text(json.dumps(view, indent=2))
        print(f"wrote {args.output}")

    elif args.cmd == "info":
        blob = Path(args.input).read_bytes()
        (magic, version, packed,
         n_bits, n_classes,
         clamp_max, _r) = struct.unpack(HEADER_FMT, blob[:HEADER_SIZE])
        level = (packed >> LEVEL_SHIFT) & LEVEL_MASK
        flags = packed & 0x07
        has_glade = bool(flags & FLAG_HAS_GLADE)
        label_kind = "utf8" if (flags & FLAG_LABEL_UTF8) else "int32"
        comp_sz = len(blob) - HEADER_SIZE
        view = unpack(blob)
        print(f"file        : {args.input}")
        print(f"magic       : {magic.decode('ascii', 'replace')}  v{version}")
        print(f"codec       : {'zstd' if level > 0 else 'none'} (level {level})")
        print(f"sections    : {'GLADE + TM' if has_glade else 'TM only'}")
        print(f"label_kind  : {label_kind}")
        print(f"n_bits      : {n_bits}")
        print(f"n_classes   : {n_classes}")
        print(f"clamp_max   : {clamp_max}")
        print(f"header      : {HEADER_SIZE} B")
        print(f"compressed  : {comp_sz} B  (file = {len(blob)} B)")
        if has_glade:
            g = view["glade"]
            print(f"  GLADE     : {g['n_features_in']} raw features → "
                  f"{n_bits} boolean bits ({4 * n_bits} B feat_idx "
                  f"+ {4 * n_bits} B thresh)")


if __name__ == "__main__":
    _cli()
