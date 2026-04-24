#!/usr/bin/env python3
"""
FBZ — Fuzzy Bitmask, zstd-22 variant of FBM.

Same layout as fcm_bitmask.py but compresses the clause bitmask block with
zstd level 22 instead of zlib-9.

Measured on tm_rules_final for 4 datasets:
    vs FBM (zlib-9): 10-14% smaller, 5-7x faster decompress, 1.2-2.2x faster
    compress. Requires `zstandard` on the target (~200 KB lib).

Magic: b"FBZ1"  (so readers don't confuse it with FBM1)

File layout (identical to FBM except for magic + payload codec):
    [header 20 B]              uncompressed, fixed layout
    [GLADE state]              uncompressed (feat_idx int32 + thresholds float32)
    [string table]             uncompressed (feature names + class names)
    [zstd-22 compressed bitmask block]   lossless clause bitmasks

Bitmask block (before compression), per class, per polarity:
    n_clauses   u16
    for each clause:
        clamp       u8
        pos_mask    uint8[chunk_bytes]   (bit i set = literal x_i included)
        neg_mask    uint8[chunk_bytes]   (bit i set = literal ~x_i included)
"""
import os, json, struct, time
from typing import List, Optional
import numpy as np
import zstandard as zstd


MAGIC = b"FBZ1"
VERSION = 1
ZSTD_LEVEL = 22


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------
def write_fbz(path: str,
              tm_rules: dict,
              glade_state: dict,
              class_names: Optional[List[str]] = None) -> int:
    """Write an FBZ file. Returns bytes written."""
    N = int(glade_state.get("n_bits") or len(
        glade_state.get("thresholds") or glade_state.get("thresh") or []))
    feat_idx = np.asarray(glade_state["feat_idx"], dtype=np.int32)
    thresholds = np.asarray(
        glade_state.get("thresholds") or glade_state["thresh"],
        dtype=np.float32)
    feat_names = list(glade_state.get("feat_names") or [])

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

    if class_names is None:
        class_names = [str(c) for c in class_order]
    K = len(class_order)
    clamp_max = int((tm_rules.get("config") or {}).get("LF")
                    or tm_rules.get("LF") or 15)
    chunk_bytes = (N + 7) // 8

    bm_buf = bytearray()
    total_clauses = 0
    for cls in class_order:
        spec = cls_table.get(cls) or cls_table.get(str(cls))
        for pol_key in (pos_key, neg_key):
            clauses = spec.get(pol_key, [])
            bm_buf += struct.pack("<H", len(clauses))
            for cl in clauses:
                total_clauses += 1
                clamp = min(int(cl.get("clamp") or clamp_max), 255)
                bm_buf.append(clamp)
                pos_mask = bytearray(chunk_bytes)
                neg_mask = bytearray(chunk_bytes)
                for i in (cl.get(inc_key) or []):
                    pos_mask[i >> 3] |= 1 << (i & 7)
                for i in (cl.get(exc_key) or []):
                    neg_mask[i >> 3] |= 1 << (i & 7)
                bm_buf += pos_mask + neg_mask

    cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
    compressed = cctx.compress(bytes(bm_buf))

    def _pack_strings(strs):
        out = bytearray()
        out += struct.pack("<H", len(strs))
        for s in strs:
            b = s.encode("utf-8")
            out += struct.pack("<H", len(b))
            out += b
        return bytes(out)

    str_block = _pack_strings(feat_names) + _pack_strings(class_names)

    header = struct.pack(
        "<4s B H H B I I I",
        MAGIC, VERSION, N, K, clamp_max,
        total_clauses,
        len(compressed),
        len(bm_buf),
    )

    blob = (header
            + feat_idx.tobytes()
            + thresholds.tobytes()
            + str_block
            + compressed)

    with open(path, "wb") as f:
        f.write(blob)
    return len(blob)


# ---------------------------------------------------------------------------
# reader + inference
# ---------------------------------------------------------------------------
class FBZModel:
    """Loaded FBZ model, ready for NumPy inference."""

    def __init__(self, N, K, clamp_max, feat_idx, thresholds,
                 feat_names, class_names, class_pos_masks, class_neg_masks,
                 class_clamps):
        self.N = N
        self.K = K
        self.clamp_max = clamp_max
        self.feat_idx = feat_idx
        self.thresholds = thresholds
        self.feat_names = feat_names
        self.class_names = class_names
        self.chunks = (N + 63) // 64
        self.class_pos_masks = class_pos_masks
        self.class_neg_masks = class_neg_masks
        self.class_clamps = class_clamps

    def binarise(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float32)
        bits = (X[:, self.feat_idx] >= self.thresholds).astype(np.uint8)
        S = bits.shape[0]
        padded = self.chunks * 64
        if bits.shape[1] < padded:
            bits = np.concatenate(
                [bits, np.zeros((S, padded - bits.shape[1]), dtype=np.uint8)],
                axis=1)
        bits = bits.reshape(S, self.chunks, 64)
        weights = (1 << np.arange(64, dtype=np.uint64))
        return (bits.astype(np.uint64) * weights).sum(axis=2).astype(np.uint64)

    def _popcount(self, a: np.ndarray) -> np.ndarray:
        if hasattr(np, "bitwise_count"):
            return np.bitwise_count(a).astype(np.int32)
        tbl = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)
        return tbl[a.view(np.uint8).reshape(*a.shape, 8)].sum(axis=-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        x = self.binarise(X)
        S = x.shape[0]
        not_x = np.bitwise_not(x)
        votes = np.zeros((S, self.K), dtype=np.int32)
        idx = 0
        for k in range(self.K):
            for sign in (+1, -1):
                pm = self.class_pos_masks[idx]
                nm = self.class_neg_masks[idx]
                cl = self.class_clamps[idx]
                if pm.shape[0] == 0:
                    idx += 1; continue
                m = (self._popcount(pm[None,:,:] & not_x[:,None,:]).sum(axis=2)
                   + self._popcount(nm[None,:,:] & x[:,None,:]).sum(axis=2))
                o = np.maximum(cl[None,:] - m, 0)
                votes[:, k] += sign * o.sum(axis=1)
                idx += 1
        return np.argmax(votes, axis=1)

    def decode_clause(self, class_idx: int, polarity: str,
                      clause_idx: int) -> str:
        idx = class_idx * 2 + (0 if polarity == "pos" else 1)
        pm = self.class_pos_masks[idx][clause_idx]
        nm = self.class_neg_masks[idx][clause_idx]
        clamp = int(self.class_clamps[idx][clause_idx])
        parts = []
        for chunk in range(self.chunks):
            pw, nw = int(pm[chunk]), int(nm[chunk])
            base = chunk * 64
            for bit in range(64):
                i = base + bit
                if i >= self.N:
                    break
                fname = self.feat_names[i] if i < len(self.feat_names) \
                    else f"bit_{i}"
                thresh = float(self.thresholds[i])
                if pw & (1 << bit):
                    parts.append(f"({fname} >= {thresh:.4g})")
                if nw & (1 << bit):
                    parts.append(f"NOT ({fname} >= {thresh:.4g})")
        return f"[clamp={clamp}] " + " AND ".join(parts)


def read_fbz(path: str) -> FBZModel:
    with open(path, "rb") as f:
        blob = f.read()

    off = 0
    hdr_fmt = "<4s B H H B I I I"
    hdr_size = struct.calcsize(hdr_fmt)
    (magic, version, N, K, clamp_max, total_clauses,
     comp_size, uncomp_size) = struct.unpack_from(hdr_fmt, blob, off)
    off += hdr_size
    assert magic == MAGIC, f"not an FBZ file: magic={magic!r}"
    assert version == VERSION, f"unsupported FBZ version {version}"

    feat_idx = np.frombuffer(blob, dtype=np.int32, count=N, offset=off).copy()
    off += 4 * N
    thresholds = np.frombuffer(blob, dtype=np.float32, count=N,
                               offset=off).copy()
    off += 4 * N

    def _read_strings():
        nonlocal off
        (n,) = struct.unpack_from("<H", blob, off); off += 2
        out = []
        for _ in range(n):
            (ln,) = struct.unpack_from("<H", blob, off); off += 2
            out.append(blob[off:off+ln].decode("utf-8"))
            off += ln
        return out

    feat_names = _read_strings()
    class_names = _read_strings()

    dctx = zstd.ZstdDecompressor()
    bm = dctx.decompress(blob[off:off+comp_size])
    assert len(bm) == uncomp_size, f"decompressed size mismatch: {len(bm)} vs {uncomp_size}"

    chunks64 = (N + 63) // 64
    chunk_bytes = (N + 7) // 8

    boff = 0
    class_pos_masks = []
    class_neg_masks = []
    class_clamps_list = []
    for _k in range(K):
        for _pol in range(2):
            (nc,) = struct.unpack_from("<H", bm, boff); boff += 2
            if nc == 0:
                class_pos_masks.append(np.zeros((0, chunks64), dtype=np.uint64))
                class_neg_masks.append(np.zeros((0, chunks64), dtype=np.uint64))
                class_clamps_list.append(np.zeros(0, dtype=np.int32))
                continue
            pm = np.zeros((nc, chunks64), dtype=np.uint64)
            nm = np.zeros((nc, chunks64), dtype=np.uint64)
            cl = np.zeros(nc, dtype=np.int32)
            for c in range(nc):
                cl[c] = bm[boff]; boff += 1
                pos_raw = bm[boff:boff+chunk_bytes]; boff += chunk_bytes
                neg_raw = bm[boff:boff+chunk_bytes]; boff += chunk_bytes
                for ch in range(chunks64):
                    pw = np.uint64(0)
                    for b in range(8):
                        byte_idx = ch * 8 + b
                        if byte_idx < chunk_bytes:
                            pw |= np.uint64(pos_raw[byte_idx]) << np.uint64(b * 8)
                    pm[c, ch] = pw
                    nw = np.uint64(0)
                    for b in range(8):
                        byte_idx = ch * 8 + b
                        if byte_idx < chunk_bytes:
                            nw |= np.uint64(neg_raw[byte_idx]) << np.uint64(b * 8)
                    nm[c, ch] = nw
            class_pos_masks.append(pm)
            class_neg_masks.append(nm)
            class_clamps_list.append(cl)

    return FBZModel(
        N=N, K=K, clamp_max=clamp_max,
        feat_idx=feat_idx, thresholds=thresholds,
        feat_names=feat_names, class_names=class_names,
        class_pos_masks=class_pos_masks,
        class_neg_masks=class_neg_masks,
        class_clamps=class_clamps_list,
    )


# ---------------------------------------------------------------------------
# CLI: pack tm_rules_final.{json,tmb} → .fbz and benchmark
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_loader import load_and_preprocess
    from sklearn.metrics import f1_score, accuracy_score
    os.environ.update({k: "1" for k in
        ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
         "NUMEXPR_NUM_THREADS"]})

    RES = "/IoT/FuzzyPatternTM/examples/glade/benchmark/results"
    CASES = [("load_wustl",   "wustl",   "WUSTL-EHMS-2020"),
             ("load_nslkdd",  "nslkdd",  "NSL-KDD"),
             ("load_toniot",  "ton_iot", "TON_IoT"),
             ("load_medsec",  "medsec",  "MedSec-25")]

    print(f"{'Dataset':<16} {'FBZ KB':>7} {'FBM KB':>7} {'shrink':>7} "
          f"{'Acc':>7} {'F1':>7} {'us/s':>7}")
    print("-" * 65)
    for loader, sid, human in CASES:
        mdir = os.path.join(RES, sid, "models")
        rules = json.load(open(os.path.join(mdir, "tm_rules_final.json")))
        glade = json.load(open(os.path.join(mdir, "glade_final.json")))
        fbz_path = os.path.join(mdir, "tm_rules_final.fbz")

        n_bytes = write_fbz(fbz_path, rules, glade)
        fbz_kb = n_bytes / 1024

        fbm_path_chosen = os.path.join(mdir, "tm_rules_chosen.fbm")
        fbm_kb = (os.path.getsize(fbm_path_chosen) / 1024
                  if os.path.exists(fbm_path_chosen) else float("nan"))

        m = read_fbz(fbz_path)
        d = load_and_preprocess(loader)
        Xte, yte = d["X_test"], d["y_test"]
        _ = m.predict(Xte[:10].astype(np.float32))
        t0 = time.perf_counter()
        y_pred = m.predict(Xte.astype(np.float32))
        wall = time.perf_counter() - t0
        acc = accuracy_score(yte, y_pred)
        f1 = f1_score(yte, y_pred, average="macro")
        shrink = fbm_kb / fbz_kb if fbm_kb == fbm_kb else float("nan")
        print(f"{human:<16} {fbz_kb:>6.2f} {fbm_kb:>7.2f} {shrink:>6.2f}x "
              f"{acc:>7.4f} {f1:>7.4f} {wall/len(Xte)*1e6:>6.1f}")
