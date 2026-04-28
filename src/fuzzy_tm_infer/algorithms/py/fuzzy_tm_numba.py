from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from os import PathLike

os.environ.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})

import numpy as np
import zstandard as zstd
from numba import int32, njit, uint64  # type: ignore[import-untyped]
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]
Int32Array = NDArray[np.int32]
Uint64Array = NDArray[np.uint64]
Uint8Array = NDArray[np.uint8]


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


def read_fbz(path: str | PathLike[str]) -> FBZModel:
    with open(path, "rb") as f:
        blob = f.read()

    hdr = "<4s B H H B I I I"
    magic, ver, n_literals, n_classes, _cmax, _total, comp_sz, _uncomp_sz = (
        struct.unpack_from(hdr, blob, 0)
    )
    if magic != b"FBZ1" or ver != 1:
        raise ValueError(f"{path} is not an FBZ1 model")

    off = struct.calcsize(hdr)
    feat_idx = np.frombuffer(blob, dtype=np.int32, count=n_literals, offset=off).copy()
    off += 4 * n_literals
    thresh = np.frombuffer(blob, dtype=np.float32, count=n_literals, offset=off).copy()
    off += 4 * n_literals

    for _ in range(2):
        (n_strings,) = struct.unpack_from("<H", blob, off)
        off += 2
        for _ in range(n_strings):
            (length,) = struct.unpack_from("<H", blob, off)
            off += 2 + length

    bitmasks: bytes = zstd.ZstdDecompressor().decompress(blob[off : off + comp_sz])
    h_words = (n_literals + 63) // 64
    chunk_bytes = (n_literals + 7) // 8

    lits_l: list[Uint64Array] = []
    inv_l: list[Uint64Array] = []
    clamp_l: list[int] = []
    sign_l: list[int] = []
    cls_l: list[int] = []
    boff = 0

    for k in range(n_classes):
        for polarity in range(2):
            sign = 1 if polarity == 0 else -1
            (n_clauses,) = struct.unpack_from("<H", bitmasks, boff)
            boff += 2
            for _ in range(n_clauses):
                clamp_l.append(bitmasks[boff])
                boff += 1
                pos_raw = bitmasks[boff : boff + chunk_bytes]
                boff += chunk_bytes
                neg_raw = bitmasks[boff : boff + chunk_bytes]
                boff += chunk_bytes

                pos_pad: Uint8Array = np.zeros(h_words * 8, np.uint8)
                neg_pad: Uint8Array = np.zeros(h_words * 8, np.uint8)
                pos_pad[: len(pos_raw)] = np.frombuffer(pos_raw, np.uint8)
                neg_pad[: len(neg_raw)] = np.frombuffer(neg_raw, np.uint8)
                lits_l.append(pos_pad.view(np.uint64).copy())
                inv_l.append(neg_pad.view(np.uint64).copy())
                sign_l.append(sign)
                cls_l.append(k)

    return FBZModel(
        n_literals=n_literals,
        n_classes=n_classes,
        h_words=h_words,
        feat_idx=feat_idx,
        thresh=thresh,
        lits=np.stack(lits_l) if lits_l else np.zeros((0, h_words), np.uint64),
        inv=np.stack(inv_l) if inv_l else np.zeros((0, h_words), np.uint64),
        clamp=np.asarray(clamp_l, np.int32),
        sign=np.asarray(sign_l, np.int32),
        cls=np.asarray(cls_l, np.int32),
    )


def build_layout(model: FBZModel) -> TMLayout:
    n_literals = model.n_literals
    n_classes = model.n_classes
    h_words = model.h_words
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

    pos_start = np.zeros(n_classes, np.int32)
    pos_end = np.zeros(n_classes, np.int32)
    neg_start = np.zeros(n_classes, np.int32)
    neg_end = np.zeros(n_classes, np.int32)

    for k in range(n_classes):
        pidx = np.where((cls == k) & (sign > 0))[0]
        nidx = np.where((cls == k) & (sign < 0))[0]
        if len(pidx):
            pos_start[k] = int(pidx[0])
            pos_end[k] = int(pidx[-1]) + 1
        if len(nidx):
            neg_start[k] = int(nidx[0])
            neg_end[k] = int(nidx[-1]) + 1

    inter = np.empty((lits.shape[0], 2 * h_words), np.uint64)
    inter[:, :h_words] = lits
    inter[:, h_words : 2 * h_words] = xor_pre

    return TMLayout(
        n_literals=n_literals,
        n_classes=n_classes,
        h_words=h_words,
        feat_idx=np.asarray(model.feat_idx, dtype=np.int32),
        thresh=np.asarray(model.thresh, dtype=np.float32),
        inter=np.ascontiguousarray(inter),
        clamp=clamp,
        pos_start=pos_start,
        pos_end=pos_end,
        neg_start=neg_start,
        neg_end=neg_end,
    )


@njit(cache=True, nogil=True, inline="always", boundscheck=False)
def popcount64(v):
    v = v - ((v >> uint64(1)) & uint64(0x5555555555555555))
    v = (v & uint64(0x3333333333333333)) + (
        (v >> uint64(2)) & uint64(0x3333333333333333)
    )
    v = (v + (v >> uint64(4))) & uint64(0x0F0F0F0F0F0F0F0F)
    return int32((v * uint64(0x0101010101010101)) >> uint64(56))


@njit(cache=True, nogil=True, fastmath=True, boundscheck=False)
def _predict_generic(
    row, feat_idx, thresh, inter, clamp, ps, pe, ns, ne, n_literals, n_classes, h_words, votes
):
    current = np.zeros(h_words, np.uint64)
    return _predict_with_scratch(
        row,
        feat_idx,
        thresh,
        inter,
        clamp,
        ps,
        pe,
        ns,
        ne,
        n_literals,
        n_classes,
        h_words,
        current,
        votes,
    )


@njit(cache=True, nogil=True, fastmath=True, boundscheck=False)
def _predict_with_scratch(
    row,
    feat_idx,
    thresh,
    inter,
    clamp,
    ps,
    pe,
    ns,
    ne,
    n_literals,
    n_classes,
    h_words,
    current,
    votes,
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
            mismatches = int32(0)
            for h in range(h_words):
                mismatches += popcount64(inter[c, h] ^ (inter[c, h + h_words] & current[h]))
            out = clamp[c] - mismatches
            pos_sum += out if out > 0 else int32(0)
            c += 1

        neg_sum = int32(0)
        c = ns[k]
        while c < ne[k]:
            mismatches = int32(0)
            for h in range(h_words):
                mismatches += popcount64(inter[c, h] ^ (inter[c, h + h_words] & current[h]))
            out = clamp[c] - mismatches
            neg_sum += out if out > 0 else int32(0)
            c += 1

        vote = pos_sum - neg_sum
        votes[k] = vote
        if vote > best_vote:
            best_vote = vote
            best_class = int32(k)
    return best_class


@njit(cache=True, nogil=True, fastmath=True, boundscheck=False)
def _predict_batch_generic(
    rows,
    feat_idx,
    thresh,
    inter,
    clamp,
    ps,
    pe,
    ns,
    ne,
    n_literals,
    n_classes,
    h_words,
):
    out = np.empty(rows.shape[0], np.int32)
    current = np.empty(h_words, np.uint64)
    votes = np.zeros(n_classes, np.int32)
    for i in range(rows.shape[0]):
        out[i] = _predict_with_scratch(
            rows[i],
            feat_idx,
            thresh,
            inter,
            clamp,
            ps,
            pe,
            ns,
            ne,
            n_literals,
            n_classes,
            h_words,
            current,
            votes,
        )
    return out


class TMModel:
    __slots__ = ("layout", "votes")
    layout: TMLayout
    votes: Int32Array

    def __init__(self, model: FBZModel) -> None:
        self.layout = build_layout(model)
        self.votes = np.zeros(self.layout.n_classes, np.int32)

    @classmethod
    def from_fbz(cls, path: str | PathLike[str]) -> TMModel:
        return cls(read_fbz(path))

    def predict(self, row: Float32Array) -> int:
        layout = self.layout
        return int(
            _predict_generic(
                row,
                layout.feat_idx,
                layout.thresh,
                layout.inter,
                layout.clamp,
                layout.pos_start,
                layout.pos_end,
                layout.neg_start,
                layout.neg_end,
                layout.n_literals,
                layout.n_classes,
                layout.h_words,
                self.votes,
            )
        )

    def predict_batch(self, rows: Float32Array) -> Int32Array:
        x = np.ascontiguousarray(rows, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"predict_batch() expects a 2D matrix, got shape {x.shape}")
        layout = self.layout
        return _predict_batch_generic(
            x,
            layout.feat_idx,
            layout.thresh,
            layout.inter,
            layout.clamp,
            layout.pos_start,
            layout.pos_end,
            layout.neg_start,
            layout.neg_end,
            layout.n_literals,
            layout.n_classes,
            layout.h_words,
        )
