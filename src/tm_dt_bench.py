#!/usr/bin/env python3
"""
TM (Numba V_BDC) vs DT (Numba) inference benchmark — Raspberry Pi 5.

Runs both models on all four datasets and prints a side-by-side
latency / accuracy / F1 comparison with TM/DT speedup ratio.

TM models : PROJ/tm_models/{stem}_model.fbz
DT models : PROJ/ml_models/{stem}/DecisionTree.pkl
Test sets  : PROJ/ml_models/{stem}/testset.npz   (DT)
           : PROJ/datasets/{stem}_test/{Name}_*   (TM raw .bin)

Deps: numpy, numba, zstandard, scikit-learn (pkl unpickle only).
"""
import os, struct, time, pickle
os.environ.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})

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

N_WARMUP = 5_000
N_TIME   = 3_000


# ══════════════════════════════════════════════════════════════════════════
#  TM — FBZ reader + Numba V_BDC kernels
# ══════════════════════════════════════════════════════════════════════════

def _read_fbz(path):
    with open(path, "rb") as f:
        blob = f.read()
    hdr = "<4s B H H B I I I"
    magic, ver, N, K, _, total, comp_sz, _ = struct.unpack_from(hdr, blob, 0)
    assert magic == b"FBZ1" and ver == 1
    off = struct.calcsize(hdr)
    feat_idx = np.frombuffer(blob, dtype=np.int32,   count=N, offset=off).copy(); off += 4*N
    thresh   = np.frombuffer(blob, dtype=np.float32, count=N, offset=off).copy(); off += 4*N
    for _ in range(2):
        (n,) = struct.unpack_from("<H", blob, off); off += 2
        for _ in range(n):
            (ln,) = struct.unpack_from("<H", blob, off); off += 2 + ln
    bm = zstd.ZstdDecompressor().decompress(blob[off:off+comp_sz])
    H  = (N + 63) // 64;  cb = (N + 7) // 8
    lits_l, inv_l, clamp_l, sign_l, cls_l = [], [], [], [], []
    boff = 0
    for k in range(K):
        for pol in range(2):
            s = +1 if pol == 0 else -1
            (nc,) = struct.unpack_from("<H", bm, boff); boff += 2
            for _ in range(nc):
                clamp_l.append(bm[boff]); boff += 1
                pr = bm[boff:boff+cb]; boff += cb
                nr = bm[boff:boff+cb]; boff += cb
                pp = np.zeros(H*8, np.uint8); pp[:len(pr)] = np.frombuffer(pr, np.uint8)
                np_ = np.zeros(H*8, np.uint8); np_[:len(nr)] = np.frombuffer(nr, np.uint8)
                lits_l.append(pp.view(np.uint64).copy())
                inv_l .append(np_.view(np.uint64).copy())
                sign_l.append(s); cls_l.append(k)
    lits = np.stack(lits_l) if lits_l else np.zeros((0,H),np.uint64)
    inv  = np.stack(inv_l)  if inv_l  else np.zeros((0,H),np.uint64)
    return dict(N=N, K=K, H=H, feat_idx=feat_idx, thresh=thresh,
                lits=lits, inv=inv,
                clamp=np.asarray(clamp_l,np.int32),
                sign =np.asarray(sign_l, np.int32),
                cls  =np.asarray(cls_l,  np.int32))


def _build_tm_layout(m):
    N,K,H = m["N"],m["K"],m["H"]
    lits,inv = m["lits"],m["inv"]
    clamp,sign,cls = m["clamp"],m["sign"],m["cls"]
    xp = np.bitwise_xor(lits,inv)
    keys  = cls.astype(np.int64)*4+(1-(sign>0).astype(np.int64))
    order = np.argsort(keys, kind="stable")
    lits,xp = lits[order],xp[order]
    clamp,sign,cls = clamp[order],sign[order],cls[order]
    ps=np.zeros(K,np.int32); pe=np.zeros(K,np.int32)
    ns=np.zeros(K,np.int32); ne=np.zeros(K,np.int32)
    for k in range(K):
        pi=np.where((cls==k)&(sign>0))[0]; ni=np.where((cls==k)&(sign<0))[0]
        if len(pi): ps[k]=int(pi[0]);  pe[k]=int(pi[-1])+1
        if len(ni): ns[k]=int(ni[0]);  ne[k]=int(ni[-1])+1
    inter=np.empty((lits.shape[0],2*H),np.uint64)
    inter[:,:H]=lits; inter[:,H:]=xp
    return dict(N=N,K=K,H=H,feat_idx=m["feat_idx"],thresh=m["thresh"],
                inter=np.ascontiguousarray(inter),clamp=clamp,
                pos_start=ps,pos_end=pe,neg_start=ns,neg_end=ne)


@njit(cache=True, nogil=True, inline="always", boundscheck=False)
def _pc(v):
    v=v-((v>>uint64(1))&uint64(0x5555555555555555))
    v=(v&uint64(0x3333333333333333))+((v>>uint64(2))&uint64(0x3333333333333333))
    v=(v+(v>>uint64(4)))&uint64(0x0F0F0F0F0F0F0F0F)
    return int32((v*uint64(0x0101010101010101))>>uint64(56))


@njit(cache=True, nogil=True, fastmath=True, boundscheck=False)
def _tm_h2(row,fi,th,inter,cl,ps,pe,ns,ne,N,K,votes):
    c0=uint64(0);c1=uint64(0)
    for i in range(N):
        if row[fi[i]]>=th[i]:
            b=uint64(1)<<uint64(i&63)
            if(i>>6)==0:c0|=b
            else:c1|=b
    bv=int32(-2000000000);bk=int32(0)
    for k in range(K):
        ap=int32(0);c=ps[k]
        while c+4<=pe[k]:
            for o in range(4):
                m=_pc(inter[c+o,0]^(inter[c+o,2]&c0))+_pc(inter[c+o,1]^(inter[c+o,3]&c1))
                v=cl[c+o]-m;ap+=(v if v>0 else int32(0))
            c+=4
        while c<pe[k]:
            m=_pc(inter[c,0]^(inter[c,2]&c0))+_pc(inter[c,1]^(inter[c,3]&c1))
            v=cl[c]-m;ap+=(v if v>0 else int32(0));c+=1
        an=int32(0);c=ns[k]
        while c+4<=ne[k]:
            for o in range(4):
                m=_pc(inter[c+o,0]^(inter[c+o,2]&c0))+_pc(inter[c+o,1]^(inter[c+o,3]&c1))
                v=cl[c+o]-m;an+=(v if v>0 else int32(0))
            c+=4
        while c<ne[k]:
            m=_pc(inter[c,0]^(inter[c,2]&c0))+_pc(inter[c,1]^(inter[c,3]&c1))
            v=cl[c]-m;an+=(v if v>0 else int32(0));c+=1
        acc=ap-an;votes[k]=acc
        if acc>bv:bv=acc;bk=int32(k)
    return bk


@njit(cache=True, nogil=True, fastmath=True, boundscheck=False)
def _tm_h4(row,fi,th,inter,cl,ps,pe,ns,ne,N,K,votes):
    c0=uint64(0);c1=uint64(0);c2=uint64(0);c3=uint64(0)
    for i in range(N):
        if row[fi[i]]>=th[i]:
            b=uint64(1)<<uint64(i&63);ci=i>>6
            if ci==0:c0|=b
            elif ci==1:c1|=b
            elif ci==2:c2|=b
            else:c3|=b
    bv=int32(-2000000000);bk=int32(0)
    for k in range(K):
        ap=int32(0);c=ps[k]
        while c+4<=pe[k]:
            for o in range(4):
                m=(_pc(inter[c+o,0]^(inter[c+o,4]&c0))+_pc(inter[c+o,1]^(inter[c+o,5]&c1))+
                   _pc(inter[c+o,2]^(inter[c+o,6]&c2))+_pc(inter[c+o,3]^(inter[c+o,7]&c3)))
                v=cl[c+o]-m;ap+=(v if v>0 else int32(0))
            c+=4
        while c<pe[k]:
            m=(_pc(inter[c,0]^(inter[c,4]&c0))+_pc(inter[c,1]^(inter[c,5]&c1))+
               _pc(inter[c,2]^(inter[c,6]&c2))+_pc(inter[c,3]^(inter[c,7]&c3)))
            v=cl[c]-m;ap+=(v if v>0 else int32(0));c+=1
        an=int32(0);c=ns[k]
        while c+4<=ne[k]:
            for o in range(4):
                m=(_pc(inter[c+o,0]^(inter[c+o,4]&c0))+_pc(inter[c+o,1]^(inter[c+o,5]&c1))+
                   _pc(inter[c+o,2]^(inter[c+o,6]&c2))+_pc(inter[c+o,3]^(inter[c+o,7]&c3)))
                v=cl[c+o]-m;an+=(v if v>0 else int32(0))
            c+=4
        while c<ne[k]:
            m=(_pc(inter[c,0]^(inter[c,4]&c0))+_pc(inter[c,1]^(inter[c,5]&c1))+
               _pc(inter[c,2]^(inter[c,6]&c2))+_pc(inter[c,3]^(inter[c,7]&c3)))
            v=cl[c]-m;an+=(v if v>0 else int32(0));c+=1
        acc=ap-an;votes[k]=acc
        if acc>bv:bv=acc;bk=int32(k)
    return bk


@njit(cache=True, nogil=True, fastmath=True, boundscheck=False)
def _tm_h6(row,fi,th,inter,cl,ps,pe,ns,ne,N,K,votes):
    c0=uint64(0);c1=uint64(0);c2=uint64(0);c3=uint64(0);c4=uint64(0);c5=uint64(0)
    for i in range(N):
        if row[fi[i]]>=th[i]:
            b=uint64(1)<<uint64(i&63);ci=i>>6
            if ci==0:c0|=b
            elif ci==1:c1|=b
            elif ci==2:c2|=b
            elif ci==3:c3|=b
            elif ci==4:c4|=b
            else:c5|=b
    bv=int32(-2000000000);bk=int32(0)
    for k in range(K):
        ap=int32(0);c=ps[k]
        while c+4<=pe[k]:
            for o in range(4):
                m=(_pc(inter[c+o,0]^(inter[c+o,6]&c0)) +_pc(inter[c+o,1]^(inter[c+o,7]&c1))+
                   _pc(inter[c+o,2]^(inter[c+o,8]&c2)) +_pc(inter[c+o,3]^(inter[c+o,9]&c3))+
                   _pc(inter[c+o,4]^(inter[c+o,10]&c4))+_pc(inter[c+o,5]^(inter[c+o,11]&c5)))
                v=cl[c+o]-m;ap+=(v if v>0 else int32(0))
            c+=4
        while c<pe[k]:
            m=(_pc(inter[c,0]^(inter[c,6]&c0))+_pc(inter[c,1]^(inter[c,7]&c1))+
               _pc(inter[c,2]^(inter[c,8]&c2))+_pc(inter[c,3]^(inter[c,9]&c3))+
               _pc(inter[c,4]^(inter[c,10]&c4))+_pc(inter[c,5]^(inter[c,11]&c5)))
            v=cl[c]-m;ap+=(v if v>0 else int32(0));c+=1
        an=int32(0);c=ns[k]
        while c+4<=ne[k]:
            for o in range(4):
                m=(_pc(inter[c+o,0]^(inter[c+o,6]&c0)) +_pc(inter[c+o,1]^(inter[c+o,7]&c1))+
                   _pc(inter[c+o,2]^(inter[c+o,8]&c2)) +_pc(inter[c+o,3]^(inter[c+o,9]&c3))+
                   _pc(inter[c+o,4]^(inter[c+o,10]&c4))+_pc(inter[c+o,5]^(inter[c+o,11]&c5)))
                v=cl[c+o]-m;an+=(v if v>0 else int32(0))
            c+=4
        while c<ne[k]:
            m=(_pc(inter[c,0]^(inter[c,6]&c0))+_pc(inter[c,1]^(inter[c,7]&c1))+
               _pc(inter[c,2]^(inter[c,8]&c2))+_pc(inter[c,3]^(inter[c,9]&c3))+
               _pc(inter[c,4]^(inter[c,10]&c4))+_pc(inter[c,5]^(inter[c,11]&c5)))
            v=cl[c]-m;an+=(v if v>0 else int32(0));c+=1
        acc=ap-an;votes[k]=acc
        if acc>bv:bv=acc;bk=int32(k)
    return bk


_TM_KERN = {2: _tm_h2, 4: _tm_h4, 6: _tm_h6}


class _TMModel:
    __slots__ = ("L","votes","_fn")
    def __init__(self, path):
        self.L     = _build_tm_layout(_read_fbz(path))
        self.votes = np.zeros(self.L["K"], np.int32)
        self._fn   = _TM_KERN[self.L["H"]]
    def predict(self, row):
        L = self.L
        return int(self._fn(row, L["feat_idx"], L["thresh"], L["inter"], L["clamp"],
                            L["pos_start"], L["pos_end"],
                            L["neg_start"], L["neg_end"],
                            L["N"], L["K"], self.votes))


# ══════════════════════════════════════════════════════════════════════════
#  DT — sklearn pickle → numpy arrays → Numba tree walk
# ══════════════════════════════════════════════════════════════════════════

@njit(cache=True, nogil=True, boundscheck=False)
def _dt_predict(row, feat, left, right, thresh, leaf_cls):
    node = np.int32(0)
    while feat[node] >= 0:
        if row[feat[node]] <= thresh[node]:
            node = left[node]
        else:
            node = right[node]
    return leaf_cls[node]


class _DTModel:
    __slots__ = ("feat","left","right","thresh","leaf_cls")
    def __init__(self, pkl_path):
        dt = pickle.load(open(pkl_path, "rb"))
        t  = dt.tree_
        leaf          = (t.children_left == -1)
        self.feat     = np.where(leaf, -1, t.feature).astype(np.int32)
        self.left     = t.children_left.astype(np.int32)
        self.right    = t.children_right.astype(np.int32)
        self.thresh   = t.threshold.astype(np.float32)
        self.leaf_cls = t.value.reshape(t.node_count,-1).argmax(axis=1).astype(np.int32)
    def predict(self, row):
        return int(_dt_predict(row, self.feat, self.left, self.right,
                               self.thresh, self.leaf_cls))


# ══════════════════════════════════════════════════════════════════════════
#  data loaders
# ══════════════════════════════════════════════════════════════════════════

def _load_tm_data(name):
    p = f"{PROJ}/datasets/{name.lower()}_test/{name}_X_test_raw.bin"
    with open(p, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        X = np.frombuffer(f.read(n*d*4), dtype=np.float32).reshape(n,d).copy()
    y = np.loadtxt(f"{PROJ}/datasets/{name.lower()}_test/{name}_Y_test.txt",
                   dtype=np.int32)
    return X, y


def _load_dt_data(stem):
    d = np.load(f"{PROJ}/ml_models/{stem}/testset.npz")
    return d["X_te"].astype(np.float32), d["Y_te"].astype(np.int32)


# ══════════════════════════════════════════════════════════════════════════
#  metrics + timing helpers
# ══════════════════════════════════════════════════════════════════════════

def _macro_f1(y_true, y_pred, K):
    f1s = []
    for k in range(K):
        tp = int(((y_pred==k)&(y_true==k)).sum())
        fp = int(((y_pred==k)&(y_true!=k)).sum())
        fn = int(((y_pred!=k)&(y_true==k)).sum())
        p  = tp/(tp+fp) if tp+fp else 0.0
        r  = tp/(tp+fn) if tp+fn else 0.0
        f1s.append(2*p*r/(p+r) if p+r else 0.0)
    return float(np.mean(f1s))


def _bench(model, X, y):
    n = len(X); K = int(y.max())+1
    for i in range(N_WARMUP):
        model.predict(X[i % n])
    y_pred = np.array([model.predict(X[i]) for i in range(n)], np.int32)
    acc = float((y_pred == y).mean())
    f1  = _macro_f1(y, y_pred, K)
    t0  = time.perf_counter()
    for i in range(N_TIME):
        model.predict(X[i % n])
    us = (time.perf_counter()-t0)/N_TIME*1e6
    return us, acc, f1


# ══════════════════════════════════════════════════════════════════════════
#  main benchmark
# ══════════════════════════════════════════════════════════════════════════

def main():
    import platform
    print("=" * 78)
    print("  TM (Numba V_BDC)  vs  DT (Numba)  —  Raspberry Pi 5")
    print(f"  Platform : {platform.machine()}  |  Python {platform.python_version()}")
    print(f"  Warmup   : {N_WARMUP:,}   Timing: {N_TIME:,} samples each")
    print("=" * 78)

    print("\n  Compiling Numba kernels …", flush=True)

    rows = []
    for stem, name in DATASETS:
        print(f"\n  ── {name} ", end="", flush=True)

        tm_model = _TMModel(f"{PROJ}/tm_models/{stem}_model.fbz")
        dt_model = _DTModel(f"{PROJ}/ml_models/{stem}/DecisionTree.pkl")

        X_tm, y_tm = _load_tm_data(name)
        X_dt, y_dt = _load_dt_data(stem)

        print("(TM…)", end=" ", flush=True)
        tm_us, tm_acc, tm_f1 = _bench(tm_model, X_tm, y_tm)

        print("(DT…)", end=" ", flush=True)
        dt_us, dt_acc, dt_f1 = _bench(dt_model, X_dt, y_dt)

        gap = tm_us / dt_us
        rows.append((name, tm_us, tm_acc, tm_f1, dt_us, dt_acc, dt_f1, gap))
        print("done")

    # ── results table ────────────────────────────────────────────────────
    w = 86
    print("\n\n" + "=" * w)
    print("  RESULTS — Raspberry Pi 5  (single-threaded, µs/sample)")
    print("=" * w)
    print(f"  {'Dataset':<10}  {'TM µs':>8}  {'TM Acc':>7}  {'TM F1':>7}"
          f"  ||  {'DT µs':>8}  {'DT Acc':>7}  {'DT F1':>7}  {'Gap(TM/DT)':>11}")
    print("  " + "-" * (w-2))
    for name, tm_us, tm_acc, tm_f1, dt_us, dt_acc, dt_f1, gap in rows:
        faster = f"{gap:.2f}×  ({'TM faster' if gap < 1 else 'DT faster'})"
        print(f"  {name:<10}  {tm_us:>8.3f}  {tm_acc:>7.4f}  {tm_f1:>7.4f}"
              f"  ||  {dt_us:>8.3f}  {dt_acc:>7.4f}  {dt_f1:>7.4f}  {faster:>14}")
    print("=" * w)

    print(f"\n  {'Dataset':<10}  {'TM µs':>8}  {'DT µs':>8}  {'Gap':>8}")
    print("  " + "-" * 42)
    for name, tm_us, *_, dt_us, _, _, gap in rows:
        print(f"  {name:<10}  {tm_us:>8.3f}  {dt_us:>8.3f}  {gap:>7.2f}×")


if __name__ == "__main__":
    main()
