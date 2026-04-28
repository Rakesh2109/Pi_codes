#!/usr/bin/env python3
"""
Convert sklearn .pkl models → self-contained numpy .npz bundles.

Source : PROJ/ml_models/{stem}/*.pkl
Output : PROJ/ml_models/{stem}/npz_models/{name}.npz

Scaler parameters are embedded in each .npz that needs scaling so
every file is fully self-contained for Pi inference (no sklearn needed).

Supported types:
  DecisionTreeClassifier   → feat, thresh, left, right, leaf_cls
  RandomForestClassifier   → feat, thresh, left, right, leaf_cls, tree_roots
  LogisticRegression       → coef, intercept         [+ scaler_*]
  LinearSVC                → coef, intercept         [+ scaler_*]
  GaussianNB               → theta, var, class_log_prior  [+ scaler_*]
  MLPClassifier            → n_layers, W0/b0 …       [+ scaler_*]
  KNeighborsClassifier     → X_train, y_train, k, n_cls  [+ scaler_*]
  XGBClassifier            → SKIP (not exportable to pure numpy)

Usage:
  python export_ml_to_npz.py          # uses default PROJ
"""
import os, sys, pickle, shutil
import numpy as np

PROJ = "/home/reddy/pi_zero2w_deploy"

DATASETS = [
    ("wustl",  "WUSTL-EHMS"),
    ("nslkdd", "NSLKDD"),
    ("toniot", "TonIoT"),
    ("medsec", "MedSec-25"),
]

NEEDS_SCALING = {"LogisticRegression", "LinearSVM", "NaiveBayes", "kNN_5"}


# ── per-type helpers ──────────────────────────────────────────────────────

def _dt_arrays(tree):
    t = tree.tree_
    leaf     = t.children_left == -1
    feat     = np.where(leaf, -1, t.feature).astype(np.int32)
    thresh   = t.threshold.astype(np.float32)
    left     = t.children_left.astype(np.int32)
    right    = t.children_right.astype(np.int32)
    leaf_cls = t.value.reshape(t.node_count, -1).argmax(axis=1).astype(np.int32)
    return feat, thresh, left, right, leaf_cls


def export_dt(model, out_dir, name):
    feat, thresh, left, right, leaf_cls = _dt_arrays(model)
    np.savez(os.path.join(out_dir, f"{name}.npz"),
             feat=feat, thresh=thresh, left=left, right=right, leaf_cls=leaf_cls)


def export_rf(model, out_dir, name):
    all_f, all_t, all_l, all_r, all_c, roots = [], [], [], [], [], []
    offset = 0
    for est in model.estimators_:
        feat, thresh, left, right, lcls = _dt_arrays(est)
        gl = np.where(left  == -1, np.int32(-1), (left  + offset).astype(np.int32))
        gr = np.where(right == -1, np.int32(-1), (right + offset).astype(np.int32))
        all_f.append(feat); all_t.append(thresh)
        all_l.append(gl);   all_r.append(gr); all_c.append(lcls)
        roots.append(offset); offset += est.tree_.node_count
    np.savez(os.path.join(out_dir, f"{name}.npz"),
             feat=np.concatenate(all_f), thresh=np.concatenate(all_t),
             left=np.concatenate(all_l), right=np.concatenate(all_r),
             leaf_cls=np.concatenate(all_c),
             tree_roots=np.array(roots, np.int32))


def _scaler_kw(scaler):
    if scaler is None:
        return {}
    return dict(scaler_mean=scaler.mean_.astype(np.float32),
                scaler_scale=scaler.scale_.astype(np.float32))


def export_linear(model, out_dir, name, scaler=None):
    np.savez(os.path.join(out_dir, f"{name}.npz"),
             coef=model.coef_.astype(np.float32),
             intercept=model.intercept_.astype(np.float32),
             **_scaler_kw(scaler))


def export_gnb(model, out_dir, name, scaler=None):
    var = (model.var_ if hasattr(model, "var_") else model.sigma_).astype(np.float32)
    clp = (model.class_log_prior_ if hasattr(model, "class_log_prior_")
           else np.log(model.class_prior_)).astype(np.float32)
    np.savez(os.path.join(out_dir, f"{name}.npz"),
             theta=model.theta_.astype(np.float32),
             var=np.maximum(var, 1e-9),
             class_log_prior=clp,
             **_scaler_kw(scaler))


def export_mlp(model, out_dir, name, scaler=None):
    kw = {"n_layers": np.array(len(model.coefs_), np.int32)}
    for i, (W, b) in enumerate(zip(model.coefs_, model.intercepts_)):
        kw[f"W{i}"] = W.astype(np.float32)
        kw[f"b{i}"] = b.astype(np.float32)
    np.savez(os.path.join(out_dir, f"{name}.npz"), **kw, **_scaler_kw(scaler))


def export_knn(model, out_dir, name, scaler=None):
    n_cls = int(model._y.max()) + 1
    np.savez(os.path.join(out_dir, f"{name}.npz"),
             X_train=model._fit_X.astype(np.float32),
             y_train=model._y.astype(np.int32),
             k=np.array(model.n_neighbors, np.int32),
             n_cls=np.array(n_cls, np.int32),
             **_scaler_kw(scaler))


_EXPORTERS = {
    "DecisionTreeClassifier":  (export_dt,     False),
    "RandomForestClassifier":  (export_rf,     False),
    "LogisticRegression":      (export_linear, True),
    "LinearSVC":               (export_linear, True),
    "GaussianNB":              (export_gnb,    True),
    "MLPClassifier":           (export_mlp,    True),
    "KNeighborsClassifier":    (export_knn,    True),
}


# ── dataset loop ──────────────────────────────────────────────────────────

def process_dataset(stem, ds_name):
    src_dir = os.path.join(PROJ, "ml_models", stem)
    out_dir = os.path.join(src_dir, "npz_models")
    if not os.path.isdir(src_dir):
        print(f"  skip {stem}: {src_dir} not found"); return
    os.makedirs(out_dir, exist_ok=True)

    ts = os.path.join(src_dir, "testset.npz")
    if os.path.exists(ts):
        shutil.copy2(ts, os.path.join(out_dir, "testset.npz"))

    scaler_path = os.path.join(src_dir, "scaler.pkl")
    scaler = pickle.load(open(scaler_path, "rb")) if os.path.exists(scaler_path) else None

    print(f"\n── {ds_name} ──────────────────────────────────────────")
    for fname in sorted(os.listdir(src_dir)):
        if not fname.endswith(".pkl") or fname == "scaler.pkl":
            continue
        name  = fname[:-4]
        model = pickle.load(open(os.path.join(src_dir, fname), "rb"))
        cls   = type(model).__name__

        if cls == "XGBClassifier":
            print(f"  {name:<28}  SKIP (XGBoost — not exportable to pure numpy)")
            continue

        entry = _EXPORTERS.get(cls)
        if entry is None:
            print(f"  {name:<28}  SKIP (unknown type: {cls})"); continue

        fn, needs_sc = entry
        sc = scaler if (needs_sc and name in NEEDS_SCALING) else None
        try:
            if needs_sc:
                fn(model, out_dir, name, sc)
            else:
                fn(model, out_dir, name)
            kb = os.path.getsize(os.path.join(out_dir, f"{name}.npz")) / 1024
            print(f"  {name:<28}  {cls:<28}  {kb:>8.1f} KB")
        except Exception as e:
            print(f"  {name:<28}  FAILED: {e}")

    print(f"  → {out_dir}")


def main():
    print("=" * 72)
    print("  EXPORT sklearn .pkl → numpy .npz")
    print(f"  PROJ: {PROJ}")
    print("=" * 72)
    for stem, ds_name in DATASETS:
        process_dataset(stem, ds_name)
    print("\nDone.")


if __name__ == "__main__":
    main()
