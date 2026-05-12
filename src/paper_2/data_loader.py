#!/usr/bin/env python3
"""
Dataset loaders and unified preprocessing.

Same preprocessing for all datasets and all models:
  1. Load raw features, label-encode target
  2. Drop ID/non-feature columns
  3. One-hot encode categorical columns (where applicable)
  4. Stratified 80/20 split (seed=42)
  5. Replace inf with NaN, fill NaN with train-only median
  6. Drop constant columns (train-only std check)
  7. Drop highly correlated columns (|r| > 0.99, train-only)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import (
    TEST_SIZE,
    RANDOM_STATE,
    CORRELATION_THRESHOLD,
    DATA_ROOT,
    NSLKDD_ROOT,
)


# ─── Loaders ────────────────────────────────────────────────────
def load_nslkdd():
    """NSL-KDD (5 categories: Normal, DoS, Probe, R2L, U2R)."""
    cols = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
        "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
        "root_shell","su_attempted","num_root","num_file_creations","num_shells",
        "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
        "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
        "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
        "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
        "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
        "label","difficulty"
    ]
    tr_path = os.path.join(NSLKDD_ROOT, "KDDTrain+.txt")
    te_path = os.path.join(NSLKDD_ROOT, "KDDTest+.txt")
    if not os.path.exists(tr_path) or not os.path.exists(te_path):
        raise FileNotFoundError(
            "NSL-KDD files not found. Set PAPER2_NSLKDD_ROOT to the folder "
            "containing KDDTrain+.txt and KDDTest+.txt."
        )
    tr = pd.read_csv(tr_path, header=None, names=cols)
    te = pd.read_csv(te_path, header=None, names=cols)
    df = pd.concat([tr, te], ignore_index=True)

    # 5-class mapping
    dos = {"back","land","neptune","pod","smurf","teardrop","apache2","udpstorm","processtable","worm"}
    probe = {"satan","ipsweep","nmap","portsweep","mscan","saint"}
    r2l = {"ftp_write","guess_passwd","imap","multihop","phf","spy","warezclient",
           "warezmaster","sendmail","named","snmpgetattack","snmpguess","xlock","xsnoop","httptunnel"}
    u2r = {"buffer_overflow","loadmodule","perl","rootkit","ps","sqlattack","xterm"}

    def group(l):
        if l == "normal": return "Normal"
        if l in dos: return "DoS"
        if l in probe: return "Probe"
        if l in r2l: return "R2L"
        if l in u2r: return "U2R"
        return "Normal"

    df["category"] = df["label"].apply(group)
    le = LabelEncoder()
    y = le.fit_transform(np.array(df["category"].tolist(), dtype=str)).astype(np.int32)

    df = df.drop(columns=["label", "difficulty", "category"])
    cat_cols = ["protocol_type", "service", "flag"]
    num_cols = [c for c in df.columns if c not in cat_cols]
    X_cat = pd.get_dummies(df[cat_cols], dtype=np.float64)
    X_num = df[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    X = np.hstack([X_num, X_cat.to_numpy()])
    feat_names = num_cols + list(X_cat.columns)
    return X, y, feat_names, list(le.classes_), "nslkdd"


def load_toniot():
    """TON_IoT (10 attack categories)."""
    path = os.path.join(DATA_ROOT, "TON_IoT", "ton_iot_network.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "TON_IoT file not found. Set PAPER2_DATA_ROOT or place the file at "
            f"{path}."
        )
    df = pd.read_csv(path)
    le = LabelEncoder()
    y = le.fit_transform(np.array(df["type"].tolist(), dtype=str)).astype(np.int32)
    cat_cols = ["proto", "service", "conn_state"]
    num_cols = [c for c in df.columns if c not in cat_cols + ["type"]]
    X_cat = pd.get_dummies(df[cat_cols], dtype=np.float64)
    X_num = df[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    X = np.hstack([X_num, X_cat.to_numpy()])
    feat_names = num_cols + list(X_cat.columns)
    return X, y, feat_names, list(le.classes_), "ton_iot"


def load_medsec():
    """MedSec-25 (5 attack classes)."""
    path = os.path.join(DATA_ROOT, "medsec", "MedSec-25.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "MedSec-25 file not found. Set PAPER2_DATA_ROOT or place the file at "
            f"{path}."
        )
    df = pd.read_csv(path)
    le = LabelEncoder()
    y = le.fit_transform(np.array(df["Label"].tolist(), dtype=str)).astype(np.int32)
    drop_cols = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Src Port", "Dst Port", "Label"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    X = df.to_numpy(dtype=np.float64)
    feat_names = list(df.columns)
    return X, y, feat_names, list(le.classes_), "medsec"


# ─── Unified preprocessing ──────────────────────────────────────
def preprocess(X, y, feat_names):
    """Split + clean. Returns Xtr, Xte, ytr, yte, feat_names."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Inf -> NaN -> train median
    X_tr = np.where(np.isinf(X_tr), np.nan, X_tr)
    X_te = np.where(np.isinf(X_te), np.nan, X_te)
    col_med = np.nanmedian(X_tr, axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)
    for j in range(X_tr.shape[1]):
        X_tr[np.isnan(X_tr[:, j]), j] = col_med[j]
        X_te[np.isnan(X_te[:, j]), j] = col_med[j]

    # Drop constant columns (train-only)
    keep = [j for j in range(X_tr.shape[1]) if np.std(X_tr[:, j]) > 1e-12]
    feat_names = [feat_names[j] for j in keep]
    X_tr, X_te = X_tr[:, keep], X_te[:, keep]

    # Drop highly correlated columns (train-only)
    corr = np.nan_to_num(np.corrcoef(X_tr.T), nan=0.0)
    drop_c = set()
    for i in range(corr.shape[0]):
        if i in drop_c:
            continue
        for j in range(i + 1, corr.shape[1]):
            if j not in drop_c and abs(corr[i, j]) > CORRELATION_THRESHOLD:
                drop_c.add(j)
    if drop_c:
        keep_idx = sorted(set(range(X_tr.shape[1])) - drop_c)
        feat_names = [feat_names[j] for j in keep_idx]
        X_tr, X_te = X_tr[:, keep_idx], X_te[:, keep_idx]

    return X_tr, X_te, y_tr, y_te, feat_names


def load_iotm():
    """IoTM (6 classes: DDoS, DoS, MQTT, Recon, Benign, Spoofing).

    Uses the dataset's own Train.csv + Test.csv concatenated; downstream
    preprocess() re-splits 80/20 stratified with seed=42 for consistency.
    """
    tr_path = os.path.join(DATA_ROOT, "IoTM", "Train.csv")
    te_path = os.path.join(DATA_ROOT, "IoTM", "Test.csv")
    if not os.path.exists(tr_path) or not os.path.exists(te_path):
        raise FileNotFoundError(
            "IoTM files not found. Set PAPER2_DATA_ROOT or place Train.csv and "
            f"Test.csv under {os.path.join(DATA_ROOT, 'IoTM')}."
        )
    tr = pd.read_csv(tr_path)
    te = pd.read_csv(te_path)
    df = pd.concat([tr, te], ignore_index=True)
    le = LabelEncoder()
    y = le.fit_transform(np.array(df["class_name"].tolist(), dtype=str)).astype(np.int32)
    df = df.drop(columns=["class_name"])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    X = df.to_numpy(dtype=np.float64)
    feat_names = list(df.columns)
    return X, y, feat_names, list(le.classes_), "iotm"


def load_wustl():
    """WUSTL-EHMS-2020 (3 classes: normal, Data Alteration, Spoofing)."""
    path = os.path.join(
        DATA_ROOT, "WUSTL", "wustl-ehms-2020_with_attacks_categories.csv"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            "WUSTL file not found. Set PAPER2_DATA_ROOT or place the file at "
            f"{path}."
        )
    df = pd.read_csv(path)
    le = LabelEncoder()
    y = le.fit_transform(np.array(df["Attack Category"].tolist(), dtype=str)).astype(np.int32)
    drop_cols = ["Dir", "Flgs", "SrcAddr", "DstAddr", "Sport", "Dport",
                 "SrcMac", "DstMac", "Attack Category", "Label"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    X = df.to_numpy(dtype=np.float64)
    feat_names = list(df.columns)
    return X, y, feat_names, list(le.classes_), "wustl"


# ─── Registry ───────────────────────────────────────────────────
LOADERS = {
    "load_nslkdd": load_nslkdd,
    "load_toniot": load_toniot,
    "load_medsec": load_medsec,
    "load_iotm":   load_iotm,
    "load_wustl":  load_wustl,
}


def load_and_preprocess(loader_name):
    """One-call loader + preprocessing."""
    X, y, feat_names, class_names, short_id = LOADERS[loader_name]()
    X_tr, X_te, y_tr, y_te, feat_names = preprocess(X, y, feat_names)
    return {
        "X_train": X_tr, "X_test": X_te,
        "y_train": y_tr, "y_test": y_te,
        "feature_names": feat_names,
        "class_names": class_names,
        "short_id": short_id,
    }
