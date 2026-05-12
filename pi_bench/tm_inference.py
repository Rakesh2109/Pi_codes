#!/usr/bin/env python3
"""TM inference using the same numba-JIT kernel that fuzzy_tm_infer
ships in /IoT/GLADE_Enhanced_TM/src/fuzzy_tm_infer/algorithms/py/
fuzzy_tm_numba.py — copied here verbatim so Paper_2 inference
benchmarks share the exact same predict path.

Construction differs from fuzzy_tm_infer's `read_fbz` only in input
shape: Paper_2 keeps the GLADE thresholds and TM clauses in plain
Python dicts (tm_rules.json / glade.json or the bundled ``model.pkl``),
not in the legacy FBZ1 binary container. Helper :func:`build_fbz_from_bundle`
adapts the dict layout into the dataclass the kernel expects.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})

import numpy as np
from numba import int32, njit, uint64
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]
Int32Array = NDArray[np.int32]
Uint64Array = NDArray[np.uint64]
Uint8Array = NDArray[np.uint8]


# ── Layout dataclasses (mirror fuzzy_tm_infer) ──────────────────────
@dataclass(frozen=True, slots=True)
class FBZModel:
    n_literals: int
    n_classes: int
    h_words: int
    feat_idx: Int32Array
    thresh: Float32Array
    lits: Uint64Array
    inv: Uint64Array
    clamp: Int32Array
    sign: Int32Array
    cls: Int32Array


@dataclass(frozen=True, slots=True)
class TMLayout:
    n_literals: int
    n_classes: int
    h_words: int
    feat_idx: Int32Array
    thresh: Float32Array
    inter: Uint64Array
    clamp: Int32Array
    pos_start: Int32Array
    pos_end: Int32Array
    neg_start: Int32Array
    neg_end: Int32Array


# ── Numba-JIT predict kernel — IDENTICAL to fuzzy_tm_infer ──────────
@njit(cache=True, nogil=True, inline="always", boundscheck=False)
def popcount64(v):
    v = v - ((v >> uint64(1)) & uint64(0x5555555555555555))
    v = (v & uint64(0x3333333333333333)) + (
        (v >> uint64(2)) & uint64(0x3333333333333333)
    )
    v = (v + (v >> uint64(4))) & uint64(0x0F0F0F0F0F0F0F0F)
    return int32((v * uint64(0x0101010101010101)) >> uint64(56))


@njit(cache=True, nogil=True, fastmath=True, boundscheck=False)
def _predict_with_scratch(
    row, feat_idx, thresh, inter, clamp, ps, pe, ns, ne,
    n_literals, n_classes, h_words, current, votes,
):
    for h in range(h_words):
        current[h] = uint64(0)
    for i in range(n_literals):
        if row[feat_idx[i]] >= thresh[i]:
            current[i >> 6] |= uint64(1) << uint64(i & 63)

    best_vote = int32(-2_000_000_000)
    best_class = int32(0)
    for k in range(n_classes):
        pos_sum = int32(0)
        c = ps[k]
        while c < pe[k]:
            mm = int32(0)
            for h in range(h_words):
                mm += popcount64(inter[c, h] ^ (inter[c, h + h_words] & current[h]))
            out = clamp[c] - mm
            pos_sum += out if out > 0 else int32(0)
            c += 1
        neg_sum = int32(0)
        c = ns[k]
        while c < ne[k]:
            mm = int32(0)
            for h in range(h_words):
                mm += popcount64(inter[c, h] ^ (inter[c, h + h_words] & current[h]))
            out = clamp[c] - mm
            neg_sum += out if out > 0 else int32(0)
            c += 1
        vote = pos_sum - neg_sum
        votes[k] = vote
        if vote > best_vote:
            best_vote = vote
            best_class = int32(k)
    return best_class


@njit(cache=True, nogil=True, fastmath=True, boundscheck=False)
def _predict_batch(
    rows, feat_idx, thresh, inter, clamp, ps, pe, ns, ne,
    n_literals, n_classes, h_words,
):
    out = np.empty(rows.shape[0], np.int32)
    current = np.empty(h_words, np.uint64)
    votes = np.zeros(n_classes, np.int32)
    for i in range(rows.shape[0]):
        out[i] = _predict_with_scratch(
            rows[i], feat_idx, thresh, inter, clamp, ps, pe, ns, ne,
            n_literals, n_classes, h_words, current, votes,
        )
    return out


# ── Bundle → FBZModel adapter (Paper_2-specific) ────────────────────
def build_fbz_from_dicts(tm_rules: dict, glade_payload: dict) -> FBZModel:
    """Construct the kernel-ready FBZModel from Paper_2's tm_rules and
    glade.json dicts (no on-disk FBZ container needed)."""
    n_literals = int(tm_rules["n_bits"])
    n_classes = int(tm_rules["n_classes"])
    h_words = (n_literals + 63) // 64
    chunk_bytes = (n_literals + 7) // 8

    # GLADE thresholds
    if glade_payload.get("quantised", False):
        q = np.asarray(glade_payload["thresh_q"], dtype=np.float64)
        thresh = (q * float(glade_payload["thresh_scale"])
                  + float(glade_payload["thresh_zp"])).astype(np.float32)
    else:
        thresh = np.asarray(glade_payload["thresh"], dtype=np.float32)
    feat_idx = np.asarray(glade_payload["feat_idx"], dtype=np.int32)

    # TM clauses → packed uint64 masks
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

    lits_l, inv_l, clamp_l, sign_l, cls_l = [], [], [], [], []
    for k, lbl in enumerate(labels):
        spec = cls_table.get(lbl) or cls_table.get(str(lbl))
        for pol_key, sign in ((pos_key, +1), (neg_key, -1)):
            for cl in (spec.get(pol_key) or []):
                pos_pad = np.zeros(h_words * 8, np.uint8)
                neg_pad = np.zeros(h_words * 8, np.uint8)
                for i in cl.get(inc_key) or []:
                    pos_pad[i >> 3] |= 1 << (i & 7)
                for i in cl.get(exc_key) or []:
                    neg_pad[i >> 3] |= 1 << (i & 7)
                lits_l.append(pos_pad.view(np.uint64).copy())
                inv_l.append(neg_pad.view(np.uint64).copy())
                clamp_l.append(int(cl.get("clamp")
                                   or (tm_rules.get("config") or {}).get("LF")
                                   or 15))
                sign_l.append(sign)
                cls_l.append(k)

    return FBZModel(
        n_literals=n_literals, n_classes=n_classes, h_words=h_words,
        feat_idx=feat_idx, thresh=thresh,
        lits=np.stack(lits_l) if lits_l
             else np.zeros((0, h_words), np.uint64),
        inv=np.stack(inv_l) if inv_l
             else np.zeros((0, h_words), np.uint64),
        clamp=np.asarray(clamp_l, np.int32),
        sign=np.asarray(sign_l, np.int32),
        cls=np.asarray(cls_l, np.int32),
    )


def build_layout(model: FBZModel) -> TMLayout:
    """Re-pack clauses into a contiguous, sign/class-sorted block so the
    kernel can iterate slices instead of indexing scattered rows."""
    lits = np.asarray(model.lits, dtype=np.uint64)
    inv = np.asarray(model.inv, dtype=np.uint64)
    clamp = np.asarray(model.clamp, dtype=np.int32)
    sign = np.asarray(model.sign, dtype=np.int32)
    cls = np.asarray(model.cls, dtype=np.int32)

    xor_pre = np.bitwise_xor(lits, inv)
    keys = cls.astype(np.int64) * 4 + (1 - (sign > 0).astype(np.int64))
    order = np.argsort(keys, kind="stable")
    lits, xor_pre = lits[order], xor_pre[order]
    clamp, sign, cls = clamp[order], sign[order], cls[order]

    pos_start = np.zeros(model.n_classes, np.int32)
    pos_end = np.zeros(model.n_classes, np.int32)
    neg_start = np.zeros(model.n_classes, np.int32)
    neg_end = np.zeros(model.n_classes, np.int32)
    for k in range(model.n_classes):
        pidx = np.where((cls == k) & (sign > 0))[0]
        nidx = np.where((cls == k) & (sign < 0))[0]
        if len(pidx):
            pos_start[k], pos_end[k] = int(pidx[0]), int(pidx[-1]) + 1
        if len(nidx):
            neg_start[k], neg_end[k] = int(nidx[0]), int(nidx[-1]) + 1

    inter = np.empty((lits.shape[0], 2 * model.h_words), np.uint64)
    inter[:, :model.h_words] = lits
    inter[:, model.h_words:2 * model.h_words] = xor_pre

    return TMLayout(
        n_literals=model.n_literals, n_classes=model.n_classes,
        h_words=model.h_words,
        feat_idx=np.asarray(model.feat_idx, dtype=np.int32),
        thresh=np.asarray(model.thresh, dtype=np.float32),
        inter=np.ascontiguousarray(inter),
        clamp=clamp, pos_start=pos_start, pos_end=pos_end,
        neg_start=neg_start, neg_end=neg_end,
    )


# ── Public class ────────────────────────────────────────────────────
class TMModel:
    __slots__ = ("layout", "votes")

    def __init__(self, model: FBZModel) -> None:
        self.layout = build_layout(model)
        self.votes = np.zeros(self.layout.n_classes, np.int32)

    @classmethod
    def from_pkl_bundle(cls, path: str | os.PathLike) -> "TMModel":
        """Build a TMModel from a Paper_2 model.pkl bundle of the form
        ``{'tm_rules': {...}, 'glade': {...}}``."""
        with open(path, "rb") as f:
            b = pickle.load(f)
        # Accept either {'tm_rules', 'glade'} or {'tm_rules', 'binarizer'}
        if "glade" in b:
            glade_payload = b["glade"]
        elif "binarizer" in b and hasattr(b["binarizer"], "to_dict"):
            glade_payload = b["binarizer"].to_dict()
        else:
            raise ValueError(
                f"{path}: bundle has neither 'glade' nor a binarizer "
                f"with .to_dict()")
        return cls(build_fbz_from_dicts(b["tm_rules"], glade_payload))

    @classmethod
    def from_dicts(cls, tm_rules: dict, glade_payload: dict) -> "TMModel":
        return cls(build_fbz_from_dicts(tm_rules, glade_payload))

    def predict(self, row: Float32Array) -> int:
        L = self.layout
        current = np.empty(L.h_words, np.uint64)
        return int(_predict_with_scratch(
            row, L.feat_idx, L.thresh, L.inter, L.clamp,
            L.pos_start, L.pos_end, L.neg_start, L.neg_end,
            L.n_literals, L.n_classes, L.h_words, current, self.votes,
        ))

    def predict_batch(self, rows: Float32Array) -> Int32Array:
        x = np.ascontiguousarray(rows, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"predict_batch expects 2D, got {x.shape}")
        L = self.layout
        return _predict_batch(
            x, L.feat_idx, L.thresh, L.inter, L.clamp,
            L.pos_start, L.pos_end, L.neg_start, L.neg_end,
            L.n_literals, L.n_classes, L.h_words,
        )
