#!/usr/bin/env python3
"""
Single source of truth: datasets, TM hyperparameters, and paths.

All runners read from here so a single edit changes all experiments.
"""

import os

# ─── Paths ──────────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
DATA_ROOT = os.environ.get("PAPER2_DATA_ROOT", "/IoT/Datasets")
NSLKDD_ROOT = os.environ.get(
    "PAPER2_NSLKDD_ROOT",
    "/root/.cache/kagglehub/datasets/hassan06/nslkdd/versions/1",
)
TM_JULIA_SRC = os.environ.get(
    "PAPER2_TSETLIN_PATH", "/IoT/FuzzyPatternTM/src/Tsetlin.jl"
)
TMP_DIR = os.environ.get("PAPER2_TMP_DIR", "/tmp/glade_benchmark")
os.makedirs(TMP_DIR, exist_ok=True)

# ─── Datasets ───────────────────────────────────────────────────
# Each entry: (loader_name, short_id, human_name)
DATASETS = [
    ("load_nslkdd",  "nslkdd",  "NSL-KDD"),
    ("load_toniot",  "ton_iot", "TON_IoT"),
    ("load_medsec",  "medsec",  "MedSec-25"),
    ("load_wustl",   "wustl",   "WUSTL-EHMS-2020"),
]

# ─── Train/test split ───────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ─── Preprocessing ──────────────────────────────────────────────
CORRELATION_THRESHOLD = 0.99  # drop |r| > threshold (train-only)

# ─── GLADE binarizer ────────────────────────────────────────────
GLADE_N_BINS = 15  # Single hyperparameter for all datasets

# ─── TM hyperparameters (per-dataset, paper Table 1) ────────────
#   C  = clauses per class
#   T  = voting-margin threshold
#   S  = specificity
#   L  = max literals per clause
#   LF = literal forgetting count
#   E  = training epochs
TM_PARAMS_PER_DATASET = {
    "nslkdd":  {"CLAUSES": 90,  "T": 12, "S": 200, "L": 40, "LF": 8,  "EPOCHS": 80,
                "STATES_NUM": 256, "INCLUDE_LIMIT": 200},
    "ton_iot": {"CLAUSES": 100, "T": 15, "S": 20,  "L": 25, "LF": 8,  "EPOCHS": 200,
                "STATES_NUM": 256, "INCLUDE_LIMIT": 200},
    "medsec":  {"CLAUSES": 80,  "T": 12, "S": 75,  "L": 30, "LF": 8,  "EPOCHS": 300,
                "STATES_NUM": 256, "INCLUDE_LIMIT": 200},
    "wustl":   {"CLAUSES": 60,  "T": 8,  "S": 300, "L": 50, "LF": 15, "EPOCHS": 200,
                "STATES_NUM": 256, "INCLUDE_LIMIT": 200},
}


def get_tm_params(short_id):
    """Return per-dataset TM hyperparameters."""
    return TM_PARAMS_PER_DATASET[short_id]

# Julia TM threads
JULIA_THREADS = "32"

# ─── ML model hyperparameters ───────────────────────────────────
# Two parallel suites (paper Section: Model Configuration):
#
#   1. Library-default suite (NOT TM-matched):
#        ML_TREE_MODELS / ML_OTHER_MODELS / TINYML_MODELS
#        Standard out-of-the-box hyperparameters.
#
#   2. TM-matched variants (capacity bounded by shared TM_PARAMS):
#        ML_MODELS_TM_MATCHED
#        Tree n_estimators = C, MLP hidden = (C,) or (2C, C),
#        DecisionTree max_depth = L.
#
# `run_ml.py` / `run_tinyml.py` consume the default suite.
# `run_ml_tmmatched.py`              consumes the TM-matched suite.


# ── Library-default suite (paper Table: Library defaults) ───────
#   XGBoost          n=100, depth=6, lr=0.1
#   RandomForest     n=100, depth unconstrained
#   DecisionTree     depth unconstrained, Gini
#   MLP_tiny         hidden=(32,),     ReLU, 200 epochs
#   MLP_small        hidden=(64,),     ReLU, 200 epochs
#   MLP_med          hidden=(128, 64), ReLU, 200 epochs
#   kNN_5            k=5, uniform, Minkowski (sklearn defaults)
#   LogisticRegression  L2, C=1.0, LBFGS
#   LinearSVM        C=1.0, squared hinge
#   GaussianNB       default priors

ML_TREE_MODELS = [
    ("XGBoost",      "xgboost", dict(n_estimators=100, max_depth=6,    learning_rate=0.1)),
    ("RandomForest", "rf",      dict(n_estimators=100, max_depth=None)),
]

ML_OTHER_MODELS = [
    ("DecisionTree",       "dt",     dict(max_depth=None, criterion="gini")),
    ("kNN_5",              "knn",    dict(n_neighbors=5, weights="uniform", metric="minkowski")),
    ("LogisticRegression", "logreg", dict(C=1.0, solver="lbfgs", max_iter=1000)),
    ("LinearSVM",          "svm",    dict(C=1.0, loss="squared_hinge", max_iter=1000, dual=False)),
    ("GaussianNB",         "nb",     dict()),
]

TINYML_MODELS = [
    ("MLP_tiny",  "mlp", dict(hidden_layer_sizes=(32,),     activation="relu", max_iter=200)),
    ("MLP_small", "mlp", dict(hidden_layer_sizes=(64,),     activation="relu", max_iter=200)),
    ("MLP_med",   "mlp", dict(hidden_layer_sizes=(128, 64), activation="relu", max_iter=200)),
]


# ── TM-matched variants (paper Table: TM-matched variants) ──────
# Per-dataset, derived from TM_PARAMS_PER_DATASET.
#   XGBoost_Cmatched       n_estimators = C
#   RandomForest_Cmatched  n_estimators = C
#   MLP_C                  hidden = (C,)
#   MLP_2C                 hidden = (2C, C)
#   DecisionTree_Lmatched  max_depth = L

def make_ml_models_tm_matched(short_id):
    """Build the TM-matched ML suite for a specific dataset."""
    p = TM_PARAMS_PER_DATASET[short_id]
    C = p["CLAUSES"]
    L = p["L"]
    return [
        ("XGBoost_Cmatched",      "xgboost", dict(n_estimators=C, max_depth=6, learning_rate=0.1)),
        ("RandomForest_Cmatched", "rf",      dict(n_estimators=C, max_depth=None)),
        ("DecisionTree_Lmatched", "dt",      dict(max_depth=L, criterion="gini")),
        ("MLP_C",                 "mlp",     dict(hidden_layer_sizes=(C,),    activation="relu", max_iter=200)),
        ("MLP_2C",                "mlp",     dict(hidden_layer_sizes=(2*C, C), activation="relu", max_iter=200)),
    ]
