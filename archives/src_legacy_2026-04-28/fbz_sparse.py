#!/usr/bin/env python3
"""
FBZ2 — sparse literal encoding of FBZ clause bitmasks.

Instead of storing full pos_mask + neg_mask bitmaps per clause, stores
only the indices of set bits (active literals) as uint16 arrays.

FBZ  format per clause: clamp(u8) + pos_mask(uint8[⌈N/8⌉]) + neg_mask(uint8[⌈N/8⌉])
FBZ2 format per clause: clamp(u8) + n_pos(u16) + pos_idx(u16[n_pos])
                                   + n_neg(u16) + neg_idx(u16[n_neg])

Saves space when avg_active_literals < ⌈N/8⌉ (breakeven).
Inference is identical: reader reconstructs uint64 bitmasks at load time.

Usage:
  python fbz_sparse.py
"""
import os, struct, time
import numpy as np
import zstandard as zstd

PROJ = "/home/reddy/pi_zero2w_deploy"

DATASETS = [
    ("nslkdd", "NSL-KDD"),
    ("toniot", "TON-IoT"),
    ("medsec", "MedSec-25"),
    ("wustl",  "WUSTL"),
]

MAGIC_FBZ  = b"FBZ1"
MAGIC_FBZ2 = b"FBZ2"
ZSTD_LEVEL = 22


# ── helpers ───────────────────────────────────────────────────────────────

def _bits_to_indices(mask_bytes: bytes, N: int):
    """Return list of set-bit positions in mask_bytes (up to bit N-1)."""
    out = []
    for byte_idx, byte in enumerate(mask_bytes):
        base = byte_idx * 8
        if base >= N:
            break
        b = byte
        while b:
            bit = (b & -b).bit_length() - 1
            idx = base + bit
            if idx < N:
                out.append(idx)
            b &= b - 1
    return out


def _indices_to_uint64(indices, chunks64: int) -> np.ndarray:
    """Reconstruct uint64 bitmask array from a list of bit indices."""
    arr = np.zeros(chunks64, dtype=np.uint64)
    for i in indices:
        chunk = i >> 6
        bit   = i & 63
        arr[chunk] |= np.uint64(1) << np.uint64(bit)
    return arr


# ── FBZ reader (original format) ─────────────────────────────────────────

def _read_strings(blob, off):
    (n,) = struct.unpack_from("<H", blob, off); off += 2
    out = []
    for _ in range(n):
        (ln,) = struct.unpack_from("<H", blob, off); off += 2
        out.append(blob[off:off+ln].decode("utf-8")); off += ln
    return out, off


def read_fbz_raw(path):
    """Read FBZ1 file, return header fields + raw per-clause index lists."""
    blob = open(path, "rb").read()
    hdr_fmt = "<4s B H H B I I I"
    hdr_size = struct.calcsize(hdr_fmt)
    magic, version, N, K, clamp_max, total_clauses, comp_size, uncomp_size = \
        struct.unpack_from(hdr_fmt, blob, 0)
    assert magic == MAGIC_FBZ, f"not FBZ1: {magic!r}"

    off = hdr_size
    feat_idx   = np.frombuffer(blob, dtype=np.int32,   count=N, offset=off).copy(); off += 4*N
    thresholds = np.frombuffer(blob, dtype=np.float32, count=N, offset=off).copy(); off += 4*N
    feat_names,  off = _read_strings(blob, off)
    class_names, off = _read_strings(blob, off)

    dctx = zstd.ZstdDecompressor()
    bm = dctx.decompress(blob[off:off+comp_size])
    chunk_bytes = (N + 7) // 8
    chunks64    = (N + 63) // 64

    # parse bitmask block → per-clause index lists
    clauses_data = []   # list of (clamp, pos_indices, neg_indices)
    boff = 0
    for _ in range(K * 2):
        (nc,) = struct.unpack_from("<H", bm, boff); boff += 2
        group = []
        for _ in range(nc):
            clamp    = bm[boff]; boff += 1
            pos_raw  = bm[boff:boff+chunk_bytes]; boff += chunk_bytes
            neg_raw  = bm[boff:boff+chunk_bytes]; boff += chunk_bytes
            pos_idx  = _bits_to_indices(pos_raw, N)
            neg_idx  = _bits_to_indices(neg_raw, N)
            group.append((clamp, pos_idx, neg_idx))
        clauses_data.append(group)

    return dict(N=N, K=K, clamp_max=clamp_max, total_clauses=total_clauses,
                feat_idx=feat_idx, thresholds=thresholds,
                feat_names=feat_names, class_names=class_names,
                clauses_data=clauses_data, chunks64=chunks64)


# ── FBZ2 writer ──────────────────────────────────────────────────────────

def write_fbz2(path, data):
    """Write FBZ2 (sparse) file from parsed clause data. Returns bytes written."""
    N, K = data["N"], data["K"]
    clamp_max = data["clamp_max"]

    def _pack_strings(strs):
        out = bytearray()
        out += struct.pack("<H", len(strs))
        for s in strs:
            b = s.encode("utf-8")
            out += struct.pack("<H", len(b)) + b
        return bytes(out)

    str_block = _pack_strings(data["feat_names"]) + _pack_strings(data["class_names"])

    # build sparse bitmask block
    bm_buf = bytearray()
    total_clauses = 0
    for group in data["clauses_data"]:
        bm_buf += struct.pack("<H", len(group))
        for clamp, pos_idx, neg_idx in group:
            total_clauses += 1
            bm_buf.append(clamp)
            bm_buf += struct.pack("<H", len(pos_idx))
            for i in pos_idx:
                bm_buf += struct.pack("<H", i)
            bm_buf += struct.pack("<H", len(neg_idx))
            for i in neg_idx:
                bm_buf += struct.pack("<H", i)

    cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
    compressed = cctx.compress(bytes(bm_buf))

    header = struct.pack(
        "<4s B H H B I I I",
        MAGIC_FBZ2, 1, N, K, clamp_max,
        total_clauses, len(compressed), len(bm_buf)
    )

    feat_idx   = data["feat_idx"].astype(np.int32)
    thresholds = data["thresholds"].astype(np.float32)
    blob = header + feat_idx.tobytes() + thresholds.tobytes() + str_block + compressed

    with open(path, "wb") as f:
        f.write(blob)
    return len(blob)


# ── FBZ2 reader (returns same FBZModel as original) ──────────────────────

class FBZ2Model:
    def __init__(self, N, K, clamp_max, feat_idx, thresholds,
                 feat_names, class_names, class_pos_masks, class_neg_masks, class_clamps):
        self.N = N; self.K = K; self.clamp_max = clamp_max
        self.feat_idx = feat_idx; self.thresholds = thresholds
        self.feat_names = feat_names; self.class_names = class_names
        self.chunks = (N + 63) // 64
        self.class_pos_masks = class_pos_masks
        self.class_neg_masks = class_neg_masks
        self.class_clamps    = class_clamps

    def binarise(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        bits = (X[:, self.feat_idx] >= self.thresholds).astype(np.uint8)
        S = bits.shape[0]; padded = self.chunks * 64
        if bits.shape[1] < padded:
            bits = np.concatenate(
                [bits, np.zeros((S, padded - bits.shape[1]), dtype=np.uint8)], axis=1)
        bits = bits.reshape(S, self.chunks, 64)
        weights = (1 << np.arange(64, dtype=np.uint64))
        return (bits.astype(np.uint64) * weights).sum(axis=2).astype(np.uint64)

    def _popcount(self, a):
        if hasattr(np, "bitwise_count"):
            return np.bitwise_count(a).astype(np.int32)
        tbl = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)
        return tbl[a.view(np.uint8).reshape(*a.shape, 8)].sum(axis=-1)

    def predict(self, X):
        x = self.binarise(X)
        S = x.shape[0]; not_x = np.bitwise_not(x)
        votes = np.zeros((S, self.K), dtype=np.int32)
        idx = 0
        for k in range(self.K):
            for sign in (+1, -1):
                pm = self.class_pos_masks[idx]; nm = self.class_neg_masks[idx]
                cl = self.class_clamps[idx]
                if pm.shape[0] == 0:
                    idx += 1; continue
                m = (self._popcount(pm[None,:,:] & not_x[:,None,:]).sum(axis=2)
                   + self._popcount(nm[None,:,:] &     x[:,None,:]).sum(axis=2))
                votes[:, k] += sign * np.maximum(cl[None,:] - m, 0).sum(axis=1)
                idx += 1
        return np.argmax(votes, axis=1)


def read_fbz2(path):
    blob = open(path, "rb").read()
    hdr_fmt = "<4s B H H B I I I"
    hdr_size = struct.calcsize(hdr_fmt)
    magic, version, N, K, clamp_max, total_clauses, comp_size, uncomp_size = \
        struct.unpack_from(hdr_fmt, blob, 0)
    assert magic == MAGIC_FBZ2, f"not FBZ2: {magic!r}"

    off = hdr_size
    feat_idx   = np.frombuffer(blob, dtype=np.int32,   count=N, offset=off).copy(); off += 4*N
    thresholds = np.frombuffer(blob, dtype=np.float32, count=N, offset=off).copy(); off += 4*N
    feat_names,  off = _read_strings(blob, off)
    class_names, off = _read_strings(blob, off)

    dctx = zstd.ZstdDecompressor()
    bm = dctx.decompress(blob[off:off+comp_size])
    chunks64 = (N + 63) // 64

    boff = 0
    class_pos_masks = []; class_neg_masks = []; class_clamps_list = []
    for _ in range(K * 2):
        (nc,) = struct.unpack_from("<H", bm, boff); boff += 2
        pm = np.zeros((nc, chunks64), dtype=np.uint64)
        nm = np.zeros((nc, chunks64), dtype=np.uint64)
        cl = np.zeros(nc, dtype=np.int32)
        for c in range(nc):
            cl[c] = bm[boff]; boff += 1
            (n_pos,) = struct.unpack_from("<H", bm, boff); boff += 2
            pos_idx = struct.unpack_from(f"<{n_pos}H", bm, boff); boff += 2*n_pos
            (n_neg,) = struct.unpack_from("<H", bm, boff); boff += 2
            neg_idx = struct.unpack_from(f"<{n_neg}H", bm, boff); boff += 2*n_neg
            pm[c] = _indices_to_uint64(pos_idx, chunks64)
            nm[c] = _indices_to_uint64(neg_idx, chunks64)
        class_pos_masks.append(pm)
        class_neg_masks.append(nm)
        class_clamps_list.append(cl)

    return FBZ2Model(N=N, K=K, clamp_max=clamp_max,
                     feat_idx=feat_idx, thresholds=thresholds,
                     feat_names=feat_names, class_names=class_names,
                     class_pos_masks=class_pos_masks,
                     class_neg_masks=class_neg_masks,
                     class_clamps=class_clamps_list)


# ── macro F1 (no sklearn) ─────────────────────────────────────────────────

def macro_f1(y_true, y_pred, K):
    f1s = []
    for k in range(K):
        tp = int(((y_pred == k) & (y_true == k)).sum())
        fp = int(((y_pred == k) & (y_true != k)).sum())
        fn = int(((y_pred != k) & (y_true == k)).sum())
        p  = tp / (tp + fp) if tp + fp else 0.0
        r  = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2*p*r / (p+r) if p+r else 0.0)
    return float(np.mean(f1s))


# ── main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 75)
    print("  FBZ → FBZ2 (sparse literal encoding)  — size + F1 comparison")
    print("=" * 75)
    print(f"  {'Dataset':<12}  {'N':>4}  {'Clauses':>7}  "
          f"{'Avg lit':>7}  {'FBZ KB':>7}  {'FBZ2 KB':>7}  {'Saving':>7}  "
          f"{'F1 FBZ':>8}  {'F1 FBZ2':>8}  {'Match':>6}")
    print("  " + "-" * 73)

    for stem, ds_name in DATASETS:
        fbz_path  = os.path.join(PROJ, "ml_models", stem, f"{stem}_model.fbz")
        fbz2_path = fbz_path.replace(".fbz", ".fbz2")
        npz_path  = os.path.join(PROJ, "ml_models", stem, "npz_models", "testset.npz")

        if not os.path.exists(fbz_path) or not os.path.exists(npz_path):
            print(f"  {ds_name}: files not found — skip"); continue

        # --- read original FBZ
        data = read_fbz_raw(fbz_path)
        N = data["N"]; K = data["K"]
        total_clauses = sum(len(g) for g in data["clauses_data"])

        # average active literals per clause
        total_lits = sum(
            len(p) + len(n)
            for group in data["clauses_data"]
            for _, p, n in group
        )
        avg_lit = total_lits / total_clauses if total_clauses else 0

        # --- write FBZ2
        fbz2_bytes = write_fbz2(fbz2_path, data)
        fbz_bytes  = os.path.getsize(fbz_path)

        # --- load test data
        d = np.load(npz_path)
        X = d["X_te"].astype(np.float32); Y = d["Y_te"].astype(np.int32)

        # --- F1 from original FBZ (re-read via original reader)
        from fcm_bitmask_zstd import read_fbz as read_fbz_orig
        m1 = read_fbz_orig(fbz_path)
        y1 = m1.predict(X)
        f1_orig = macro_f1(Y, y1, K)

        # --- F1 from FBZ2
        m2 = read_fbz2(fbz2_path)
        y2 = m2.predict(X)
        f1_new  = macro_f1(Y, y2, K)

        match = "YES" if np.array_equal(y1, y2) else "NO "
        saving = (1 - fbz2_bytes / fbz_bytes) * 100

        print(f"  {ds_name:<12}  {N:>4}  {total_clauses:>7}  "
              f"{avg_lit:>7.1f}  {fbz_bytes/1024:>7.1f}  {fbz2_bytes/1024:>7.1f}  "
              f"{saving:>6.1f}%  {f1_orig:>8.4f}  {f1_new:>8.4f}  {match:>6}")

    print("=" * 75)
    print("  FBZ2 files written alongside originals (*_model.fbz2)")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    main()
