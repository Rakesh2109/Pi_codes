from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]
Int32Array = NDArray[np.int32]


@njit(cache=True, nogil=True)
def _predict_one(
    row: Float32Array,
    feature: Int32Array,
    left: Int32Array,
    right: Int32Array,
    threshold: Float32Array,
    leaf_class: Int32Array,
) -> np.int32:
    node = np.int32(0)
    while feature[node] >= 0:
        node = left[node] if row[feature[node]] <= threshold[node] else right[node]
    return leaf_class[node]


@njit(cache=True, nogil=True)
def _predict_batch(
    rows: Float32Array,
    feature: Int32Array,
    left: Int32Array,
    right: Int32Array,
    threshold: Float32Array,
    leaf_class: Int32Array,
) -> Int32Array:
    out = np.empty(rows.shape[0], dtype=np.int32)
    for i in range(rows.shape[0]):
        out[i] = _predict_one(rows[i], feature, left, right, threshold, leaf_class)
    return out


@dataclass(frozen=True, slots=True)
class DecisionTreeModel:
    feature: Int32Array
    left: Int32Array
    right: Int32Array
    threshold: Float32Array
    leaf_class: Int32Array
    n_classes: int

    @classmethod
    def from_pickle(cls, path: Path) -> DecisionTreeModel:
        tree = _load_trusted_pickle(path)
        raw = tree.tree_
        leaf = raw.children_left == -1
        values = raw.value.reshape(raw.node_count, -1)
        return cls(
            feature=np.where(leaf, -1, raw.feature).astype(np.int32),
            left=raw.children_left.astype(np.int32),
            right=raw.children_right.astype(np.int32),
            threshold=raw.threshold.astype(np.float32),
            leaf_class=values.argmax(axis=1).astype(np.int32),
            n_classes=int(values.shape[1]),
        )

    def predict(self, row: Float32Array) -> int:
        return int(
            _predict_one(
                row,
                self.feature,
                self.left,
                self.right,
                self.threshold,
                self.leaf_class,
            )
        )

    def predict_batch(self, rows: Float32Array) -> Int32Array:
        x = np.ascontiguousarray(rows, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"predict_batch() expects a 2D matrix, got shape {x.shape}")
        return _predict_batch(
            x,
            self.feature,
            self.left,
            self.right,
            self.threshold,
            self.leaf_class,
        )


def _load_trusted_pickle(path: Path) -> object:
    try:
        from sklearn.exceptions import InconsistentVersionWarning
    except Exception:  # pragma: no cover - sklearn import error is raised by pickle.
        InconsistentVersionWarning = Warning

    with path.open("rb") as f, warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        return pickle.load(f)
