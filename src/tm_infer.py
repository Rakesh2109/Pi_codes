#!/usr/bin/env python3
"""
TM inference benchmark — Numba V_BDC kernels, single-threaded.

Loads .fbz models and pre-binarised .bin test sets from pi_zero2w_deploy,
runs per-sample latency + macro-F1 benchmark, and prints a summary table
matching the format of ml_numpy_infer.py.

Deps: numpy, numba, zstandard.
"""
import os, struct, time
os.environ.update({"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1"})

import numpy as np
import zstandard as zstd
from numba import njit, uint64, int32

PROJ = "/home/reddy/pi_zero2w_deploy"

DATASETS = [
    ("wustl",  "WUSTL"),
    ("nslkdd", "NSLKDD"),
    ("toniot", "TonIoT"),
    ("medsec", "MedSec"),
]


# ── FBZ reader ────────────────────────────────────────────────────────────
def read_fbz(path):
    with open(path, "rb") as f:
        blob = f.read()
    hdr = "<4s B H H B I I I"
    magic, ver, N, K, _cmax, total, comp_sz, uncomp_sz = struct.unpack_from(hdr, blob, 0)
    assert magic == b"FBZ1" and ver == 1
    off = struct.calcsize(hdr)
    feat_idx = np.frombuffer(blob, dtype=np.int32,   count=N, offset=off).copy(); off += 4*N
    thresh   = np.frombuffer(blob, dtype=np.float32, count=N, offset=off).copy(); off += 4*N

    for _ in range(2):                               # skip two string tables
        (n,) = struct.unpack_from("<H", blob, off); off += 2
        for _ in range(n):
            (ln,) = struct.unpack_from("<H", blob, off); off += 2 + ln

    bm = zstd.ZstdDecompressor().decompress(blob[off:off+comp_sz])
    H  = (N + 63) // 64
    chunk_bytes = (N + 7) // 8

    lits_l, inv_l, clamp_l, sign_l, cls_l = [], [], [], [], []
    boff = 0
    for k in range(K):
        for pol in range(2):
            s = +1 if pol == 0 else -1
            (nc,) = struct.unpack_from("<H", bm, boff); boff += 2
            for _ in range(nc):
                clamp_l.append(bm[boff]); boff += 1
                pos_raw = bm[boff:boff+chunk_bytes]; boff += chunk_bytes
                neg_raw = bm[boff:boff+chunk_bytes]; boff += chunk_bytes
                pad_p = np.zeros(H*8, np.uint8); pad_p[:len(pos_raw)] = np.frombuffer(pos_raw, np.uint8)
                pad_n = np.zeros(H*8, np.uint8); pad_n[:len(neg_raw)] = np.frombuffer(neg_raw, np.uint8)
                lits_l.append(pad_p.view(np.uint64).copy())
                inv_l .append(pad_n.view(np.uint64).copy())
                sign_l.append(s); cls_l.append(k)

    lits  = np.stack(lits_l)  if lits_l  else np.zeros((0, H), np.uint64)
    inv   = np.stack(inv_l)   if inv_l   else np.zeros((0, H), np.uint64)
    return dict(N=N, K=K, H=H, feat_idx=feat_idx, thresh=thresh,
                lits=lits, inv=inv,
                clamp=np.asarray(clamp_l, np.int32),
                sign =np.asarray(sign_l,  np.int32),
                cls  =np.asarray(cls_l,   np.int32))


# ── layout: sort by (class, polarity) + interleaved [lits, xor_pre] ──────
def build_layout(m):
    N, K, H = m["N"], m["K"], m["H"]
    lits, inv  = m["lits"], m["inv"]
    clamp, sign, cls = m["clamp"], m["sign"], m["cls"]
    xor_pre = np.bitwise_xor(lits, inv)

    keys  = cls.astype(np.int64) * 4 + (1 - (sign > 0).astype(np.int64))
    order = np.argsort(keys, kind="stable")
    lits, xor_pre = lits[order], xor_pre[order]
    clamp, sign, cls = clamp[order], sign[order], cls[order]

    pos_start = np.zeros(K, np.int32); pos_end = np.zeros(K, np.int32)
    neg_start = np.zeros(K, np.int32); neg_end = np.zeros(K, np.int32)
    for k in range(K):
        pidx = np.where((cls == k) & (sign > 0))[0]
        nidx = np.where((cls == k) & (sign < 0))[0]
        if len(pidx): pos_start[k]=int(pidx[0]);  pos_end[k]=int(pidx[-1])+1
        if len(nidx): neg_start[k]=int(nidx[0]);  neg_end[k]=int(nidx[-1])+1

    inter = np.empty((lits.shape[0], 2*H), np.uint64)
    inter[:, :H]    = lits
    inter[:, H:2*H] = xor_pre
    inter = np.ascontiguousarray(inter)

    return dict(N=N, K=K, H=H,
                feat_idx=m["feat_idx"], thresh=m["thresh"],
                inter=inter, clamp=clamp,
                pos_start=pos_start, pos_end=pos_end,
                neg_start=neg_start, neg_end=neg_end)


# ── SWAR popcount ─────────────────────────────────────────────────────────
@njit(cache=True, nogil=True, inline="always", boundscheck=False)
def pc(v):
    v = v - ((v >> uint64(1)) & uint64(0x5555555555555555))
    v = (v & uint64(0x3333333333333333)) + ((v >> uint64(2)) & uint64(0x3333333333333333))
    v = (v + (v >> uint64(4))) & uint64(0x0F0F0F0F0F0F0F0F)
    return int32((v * uint64(0x0101010101010101)) >> uint64(56))


# ── H=2 kernel ────────────────────────────────────────────────────────────
@njit(cache=True, nogil=True, fastmath=True, boundscheck=False)
def _predict_h2(row, feat_idx, thresh, inter, clamp,
                ps, pe, ns, ne, N, K, votes):
    c0=uint64(0); c1=uint64(0)
    for i in range(N):
        if row[feat_idx[i]] >= thresh[i]:
            b = uint64(1) << uint64(i & 63)
            if (i>>6)==0: c0|=b
            else:         c1|=b
    bv=int32(-2_000_000_000); bk=int32(0)
    for k in range(K):
        ap=int32(0); c=ps[k]
        while c+4<=pe[k]:
            for off in range(4):
                ma=pc(inter[c+off,0]^(inter[c+off,2]&c0))+pc(inter[c+off,1]^(inter[c+off,3]&c1))
                o=clamp[c+off]-ma; ap+=(o if o>0 else int32(0))
            c+=4
        while c<pe[k]:
            m=pc(inter[c,0]^(inter[c,2]&c0))+pc(inter[c,1]^(inter[c,3]&c1))
            o=clamp[c]-m; ap+=(o if o>0 else int32(0)); c+=1
        an=int32(0); c=ns[k]
        while c+4<=ne[k]:
            for off in range(4):
                ma=pc(inter[c+off,0]^(inter[c+off,2]&c0))+pc(inter[c+off,1]^(inter[c+off,3]&c1))
                o=clamp[c+off]-ma; an+=(o if o>0 else int32(0))
            c+=4
        while c<ne[k]:
            m=pc(inter[c,0]^(inter[c,2]&c0))+pc(inter[c,1]^(inter[c,3]&c1))
            o=clamp[c]-m; an+=(o if o>0 else int32(0)); c+=1
        acc=ap-an; votes[k]=acc
        if acc>bv: bv=acc; bk=int32(k)
    return bk


# ── H=4 kernel ────────────────────────────────────────────────────────────
@njit(cache=True, nogil=True, fastmath=True, boundscheck=False)
def _predict_h4(row, feat_idx, thresh, inter, clamp,
                ps, pe, ns, ne, N, K, votes):
    c0=uint64(0);c1=uint64(0);c2=uint64(0);c3=uint64(0)
    for i in range(N):
        if row[feat_idx[i]] >= thresh[i]:
            b=uint64(1)<<uint64(i&63); ci=i>>6
            if ci==0: c0|=b
            elif ci==1: c1|=b
            elif ci==2: c2|=b
            else: c3|=b
    bv=int32(-2_000_000_000); bk=int32(0)
    for k in range(K):
        ap=int32(0); c=ps[k]
        while c+4<=pe[k]:
            for off in range(4):
                ma=(pc(inter[c+off,0]^(inter[c+off,4]&c0))+pc(inter[c+off,1]^(inter[c+off,5]&c1))+
                    pc(inter[c+off,2]^(inter[c+off,6]&c2))+pc(inter[c+off,3]^(inter[c+off,7]&c3)))
                o=clamp[c+off]-ma; ap+=(o if o>0 else int32(0))
            c+=4
        while c<pe[k]:
            m=(pc(inter[c,0]^(inter[c,4]&c0))+pc(inter[c,1]^(inter[c,5]&c1))+
               pc(inter[c,2]^(inter[c,6]&c2))+pc(inter[c,3]^(inter[c,7]&c3)))
            o=clamp[c]-m; ap+=(o if o>0 else int32(0)); c+=1
        an=int32(0); c=ns[k]
        while c+4<=ne[k]:
            for off in range(4):
                ma=(pc(inter[c+off,0]^(inter[c+off,4]&c0))+pc(inter[c+off,1]^(inter[c+off,5]&c1))+
                    pc(inter[c+off,2]^(inter[c+off,6]&c2))+pc(inter[c+off,3]^(inter[c+off,7]&c3)))
                o=clamp[c+off]-ma; an+=(o if o>0 else int32(0))
            c+=4
        while c<ne[k]:
            m=(pc(inter[c,0]^(inter[c,4]&c0))+pc(inter[c,1]^(inter[c,5]&c1))+
               pc(inter[c,2]^(inter[c,6]&c2))+pc(inter[c,3]^(inter[c,7]&c3)))
            o=clamp[c]-m; an+=(o if o>0 else int32(0)); c+=1
        acc=ap-an; votes[k]=acc
        if acc>bv: bv=acc; bk=int32(k)
    return bk


# ── H=6 kernel ────────────────────────────────────────────────────────────
@njit(cache=True, nogil=True, fastmath=True, boundscheck=False)
def _predict_h6(row, feat_idx, thresh, inter, clamp,
                ps, pe, ns, ne, N, K, votes):
    c0=uint64(0);c1=uint64(0);c2=uint64(0);c3=uint64(0);c4=uint64(0);c5=uint64(0)
    for i in range(N):
        if row[feat_idx[i]] >= thresh[i]:
            b=uint64(1)<<uint64(i&63); ci=i>>6
            if ci==0: c0|=b
            elif ci==1: c1|=b
            elif ci==2: c2|=b
            elif ci==3: c3|=b
            elif ci==4: c4|=b
            else: c5|=b
    bv=int32(-2_000_000_000); bk=int32(0)
    for k in range(K):
        ap=int32(0); c=ps[k]
        while c+4<=pe[k]:
            for off in range(4):
                ma=(pc(inter[c+off,0]^(inter[c+off,6]&c0)) +pc(inter[c+off,1]^(inter[c+off,7]&c1))+
                    pc(inter[c+off,2]^(inter[c+off,8]&c2)) +pc(inter[c+off,3]^(inter[c+off,9]&c3))+
                    pc(inter[c+off,4]^(inter[c+off,10]&c4))+pc(inter[c+off,5]^(inter[c+off,11]&c5)))
                o=clamp[c+off]-ma; ap+=(o if o>0 else int32(0))
            c+=4
        while c<pe[k]:
            m=(pc(inter[c,0]^(inter[c,6]&c0))+pc(inter[c,1]^(inter[c,7]&c1))+
               pc(inter[c,2]^(inter[c,8]&c2))+pc(inter[c,3]^(inter[c,9]&c3))+
               pc(inter[c,4]^(inter[c,10]&c4))+pc(inter[c,5]^(inter[c,11]&c5)))
            o=clamp[c]-m; ap+=(o if o>0 else int32(0)); c+=1
        an=int32(0); c=ns[k]
        while c+4<=ne[k]:
            for off in range(4):
                ma=(pc(inter[c+off,0]^(inter[c+off,6]&c0)) +pc(inter[c+off,1]^(inter[c+off,7]&c1))+
                    pc(inter[c+off,2]^(inter[c+off,8]&c2)) +pc(inter[c+off,3]^(inter[c+off,9]&c3))+
                    pc(inter[c+off,4]^(inter[c+off,10]&c4))+pc(inter[c+off,5]^(inter[c+off,11]&c5)))
                o=clamp[c+off]-ma; an+=(o if o>0 else int32(0))
            c+=4
        while c<ne[k]:
            m=(pc(inter[c,0]^(inter[c,6]&c0))+pc(inter[c,1]^(inter[c,7]&c1))+
               pc(inter[c,2]^(inter[c,8]&c2))+pc(inter[c,3]^(inter[c,9]&c3))+
               pc(inter[c,4]^(inter[c,10]&c4))+pc(inter[c,5]^(inter[c,11]&c5)))
            o=clamp[c]-m; an+=(o if o>0 else int32(0)); c+=1
        acc=ap-an; votes[k]=acc
        if acc>bv: bv=acc; bk=int32(k)
    return bk


_KERNELS = {2: _predict_h2, 4: _predict_h4, 6: _predict_h6}


# ── model wrapper ─────────────────────────────────────────────────────────
class TMModel:
    __slots__ = ("L", "votes", "_fn")
    def __init__(self, fbz_path):
        self.L    = build_layout(read_fbz(fbz_path))
        self.votes = np.zeros(self.L["K"], np.int32)
        H = self.L["H"]
        if H not in _KERNELS:
            raise ValueError(f"H={H} not supported")
        self._fn = _KERNELS[H]

    def predict(self, row):
        L = self.L
        return int(self._fn(row, L["feat_idx"], L["thresh"], L["inter"], L["clamp"],
                            L["pos_start"], L["pos_end"],
                            L["neg_start"], L["neg_end"],
                            L["N"], L["K"], self.votes))


# ── data loaders ─────────────────────────────────────────────────────────
def load_X(name):
    p = f"{PROJ}/datasets/{name.lower()}_test/{name}_X_test_raw.bin"
    with open(p, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        X = np.frombuffer(f.read(n*d*4), dtype=np.float32).reshape(n, d).copy()
    return X

def load_y(name):
    return np.loadtxt(f"{PROJ}/datasets/{name.lower()}_test/{name}_Y_test.txt",
                      dtype=np.int32)

def macro_f1(y_true, y_pred, K):
    f1s = []
    for k in range(K):
        tp = int(((y_pred==k)&(y_true==k)).sum())
        fp = int(((y_pred==k)&(y_true!=k)).sum())
        fn = int(((y_pred!=k)&(y_true==k)).sum())
        p = tp/(tp+fp) if tp+fp else 0.0
        r = tp/(tp+fn) if tp+fn else 0.0
        f1s.append(2*p*r/(p+r) if p+r else 0.0)
    return float(np.mean(f1s))


# ── benchmark ─────────────────────────────────────────────────────────────
def run():
    print("=" * 70)
    print("  TM INFERENCE BENCHMARK (Numba V_BDC, single-threaded)")
    print(f"  models: {PROJ}/tm_models/")
    print("=" * 70)
    print(f"\n  {'Dataset':<10}  {'H':>2}  {'N':>4}  {'K':>3}"
          f"  {'µs/sample':>10}  {'Acc':>7}  {'F1':>7}")
    print("  " + "-" * 55)

    rows = []
    for stem, name in DATASETS:
        fbz_path = f"{PROJ}/tm_models/{stem}_model.fbz"
        if not os.path.exists(fbz_path):
            print(f"  {name:<10}  SKIP ({fbz_path} not found)")
            continue
        try:
            model = TMModel(fbz_path)
            X = load_X(name)
            y = load_y(name)
            n = len(X)

            # JIT warmup
            for i in range(5000):
                model.predict(X[i % n])

            # accuracy on full test set
            y_pred = np.array([model.predict(X[i]) for i in range(n)], np.int32)
            acc = float((y_pred == y).mean())
            f1  = macro_f1(y, y_pred, model.L["K"])

            # per-sample latency
            N_TIME = 3000
            t = time.perf_counter()
            for i in range(N_TIME):
                model.predict(X[i % n])
            us = (time.perf_counter() - t) / N_TIME * 1e6

            H = model.L["H"]; N_feat = model.L["N"]; K = model.L["K"]
            print(f"  {name:<10}  {H:>2}  {N_feat:>4}  {K:>3}"
                  f"  {us:>9.3f}   {acc:>6.4f}   {f1:>6.4f}")
            rows.append(dict(name=name, H=H, N=N_feat, K=K, us=us, acc=acc, f1=f1))
        except Exception as e:
            print(f"  {name:<10}  ERROR: {e}")

    print("\n\n  SUMMARY (sorted by µs/sample)")
    print("  " + "-" * 55)
    for r in sorted(rows, key=lambda x: x["us"]):
        print(f"  {r['name']:<10}  {r['H']:>2}  {r['N']:>4}  {r['K']:>3}"
              f"  {r['us']:>9.3f}   {r['acc']:>6.4f}   {r['f1']:>6.4f}")


if __name__ == "__main__":
    run()
