#!/usr/bin/env python3
"""
Pure-numpy ML inference benchmark — no sklearn, no numba.

Loads .npz model files from  PROJ/ml_models/{stem}/npz_models/
(produced by export_ml_to_npz.py) and measures:
  • per-sample latency (µs)
  • accuracy + macro-F1 on the full test set

Model type is detected from keys in the .npz:
  DT     — feat, thresh, left, right, leaf_cls         (no tree_roots)
  RF     — + tree_roots
  LINEAR — coef, intercept
  GNB    — theta, var, class_log_prior
  MLP    — n_layers, W0, b0, …
  KNN    — X_train, y_train, k, n_cls

Scaler params (scaler_mean / scaler_scale) are embedded and applied
automatically when present.

Usage:
  python ml_numpy_infer.py
"""
import os, sys, time
import numpy as np

os.environ.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})

PROJ = "/home/reddy/pi_zero2w_deploy"

KNN_MAX_EVAL = 500    # samples used for kNN accuracy (full dataset is O(n_train*n_test))
KNN_MAX_TIME = 100   # samples used for kNN latency timing

DATASETS = [
    ("wustl",  "WUSTL-EHMS"),
    ("nslkdd", "NSLKDD"),
    ("toniot", "TonIoT"),
    ("medsec", "MedSec-25"),
]


# ── type detection ────────────────────────────────────────────────────────

def detect_type(d):
    if "tree_roots" in d: return "RF"
    if "feat"       in d: return "DT"
    if "theta"      in d: return "GNB"
    if "W0"         in d: return "MLP"
    if "X_train"    in d: return "KNN"
    if "coef"       in d: return "LINEAR"
    return None


# ── predict functions (single 1-D float32 row) ───────────────────────────

def _scale(row, d):
    return (row - d["scaler_mean"]) / d["scaler_scale"] if "scaler_mean" in d else row


def predict_dt(row, d):
    feat = d["feat"]; thresh = d["thresh"]
    left = d["left"]; right  = d["right"]; lc = d["leaf_cls"]
    node = 0
    while feat[node] >= 0:
        node = int(left[node]) if row[feat[node]] <= thresh[node] else int(right[node])
    return int(lc[node])


def predict_rf(row, d):
    feat = d["feat"]; thresh = d["thresh"]
    left = d["left"]; right  = d["right"]; lc = d["leaf_cls"]
    n_cls = int(lc.max()) + 1
    votes = np.zeros(n_cls, np.int32)
    for start in d["tree_roots"]:
        node = int(start)
        while feat[node] >= 0:
            node = int(left[node]) if row[feat[node]] <= thresh[node] else int(right[node])
        votes[lc[node]] += 1
    return int(np.argmax(votes))


def predict_linear(row, d):
    x = _scale(row, d)
    scores = d["coef"] @ x + d["intercept"]
    return int(scores[0] >= 0) if len(scores) == 1 else int(np.argmax(scores))


def predict_gnb(row, d):
    x = _scale(row, d)
    log_p = d["class_log_prior"] - 0.5 * np.sum(
        np.log(2.0 * np.pi * d["var"]) + (x - d["theta"]) ** 2 / d["var"], axis=1)
    return int(np.argmax(log_p))


def predict_mlp(row, d):
    x = _scale(row, d)
    n = int(d["n_layers"])
    for i in range(n):
        x = x @ d[f"W{i}"] + d[f"b{i}"]
        if i < n - 1:
            np.maximum(x, 0.0, out=x)
    return int(np.argmax(x))


def predict_knn(row, d):
    x = _scale(row, d)
    dists = ((d["X_train"] - x) ** 2).sum(axis=1)
    k     = int(d["k"])
    nn    = np.argpartition(dists, k)[:k]
    return int(np.bincount(d["y_train"][nn], minlength=int(d["n_cls"])).argmax())


_PREDICT = {
    "DT":     predict_dt,
    "RF":     predict_rf,
    "LINEAR": predict_linear,
    "GNB":    predict_gnb,
    "MLP":    predict_mlp,
    "KNN":    predict_knn,
}


def predict_one(row, d, mtype):
    return _PREDICT[mtype](row, d)


# ── metrics ───────────────────────────────────────────────────────────────

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


# ── per-dataset benchmark ─────────────────────────────────────────────────

def run_dataset(stem, ds_name):
    npz_dir = os.path.join(PROJ, "ml_models", stem, "npz_models")
    if not os.path.isdir(npz_dir):
        print(f"\n── {ds_name}: {npz_dir} not found — run export_ml_to_npz.py first")
        return []

    data  = np.load(os.path.join(npz_dir, "testset.npz"))
    X     = data["X_te"].astype(np.float32)
    Y     = data["Y_te"].astype(np.int32)
    n, nf = X.shape
    K     = int(Y.max()) + 1

    print(f"\n── {ds_name}  (n={n}, features={nf}, classes={K})")
    print(f"  {'Model':<28}  {'Type':>6}  {'µs/sample':>10}  {'Acc':>7}  {'F1':>7}")
    print("  " + "-" * 65)

    rows = []
    for fname in sorted(os.listdir(npz_dir)):
        if not fname.endswith(".npz") or fname == "testset.npz":
            continue
        name = fname[:-4]
        d    = dict(np.load(os.path.join(npz_dir, fname), allow_pickle=False))
        mt   = detect_type(d)
        if mt is None:
            print(f"  {name:<28}  UNKNOWN format — skip")
            continue

        # warmup (fewer for kNN to avoid minutes of wait)
        n_warm = min(50 if mt == "KNN" else 300, n)
        for i in range(n_warm):
            predict_one(X[i], d, mt)

        # accuracy (cap kNN — distance to full training set is O(n_train * n_test))
        n_eval = min(KNN_MAX_EVAL, n) if mt == "KNN" else n
        y_pred = np.array([predict_one(X[i], d, mt) for i in range(n_eval)], np.int32)
        acc    = float((y_pred == Y[:n_eval]).mean())
        f1     = macro_f1(Y[:n_eval], y_pred, K)

        # per-sample latency
        N_TIME = min(KNN_MAX_TIME if mt == "KNN" else 2000, n)
        t0 = time.perf_counter()
        for i in range(N_TIME):
            predict_one(X[i % n], d, mt)
        us = (time.perf_counter() - t0) / N_TIME * 1e6

        tag = f"  [n={n_eval}]" if mt == "KNN" else ""
        print(f"  {name:<28}  {mt:>6}  {us:>9.2f}   {acc:>6.4f}   {f1:>6.4f}{tag}")
        rows.append(dict(name=name, mtype=mt, us=us, acc=acc, f1=f1))

    return rows


# ── main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  PURE-NUMPY ML INFERENCE BENCHMARK")
    print(f"  PROJ: {PROJ}")
    print("=" * 70)

    summary = {}
    for stem, ds_name in DATASETS:
        summary[ds_name] = run_dataset(stem, ds_name)

    print("\n\n" + "=" * 70)
    print("  SUMMARY  (sorted by µs/sample)")
    print("=" * 70)
    for ds_name, rows in summary.items():
        if not rows:
            continue
        print(f"\n  {ds_name}")
        print(f"  {'Model':<28}  {'Type':>6}  {'µs/samp':>8}  {'Acc':>7}  {'F1':>7}")
        print("  " + "-" * 60)
        for r in sorted(rows, key=lambda x: x["us"]):
            print(f"  {r['name']:<28}  {r['mtype']:>6}  "
                  f"{r['us']:>7.2f}   {r['acc']:>6.4f}   {r['f1']:>6.4f}")


if __name__ == "__main__":
    main()
