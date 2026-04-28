from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

KBinsStrategy = Literal["uniform", "quantile"]


class KBinsBooleanizer:
    """Thermometer-style KBins discretizer for Tsetlin Machine inputs.

    This mirrors the threshold-learning part of scikit-learn's
    ``KBinsDiscretizer`` for the two strategies that produce useful ordered
    thresholds for TM booleanization:

    - ``uniform``: evenly spaced bins between feature min and max.
    - ``quantile``: bins with approximately equal sample counts.

    The output is not one-hot. It is an ordered Boolean/thermometer encoding:
    ``X[:, feature] >= bin_edge`` for every learned internal bin edge.
    """

    def __init__(self, n_bins: int = 5, strategy: KBinsStrategy = "quantile"):
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        if strategy not in {"uniform", "quantile"}:
            raise ValueError("strategy must be 'uniform' or 'quantile'")
        self.n_bins = int(n_bins)
        self.strategy: KBinsStrategy = strategy
        self.bin_edges_: list[np.ndarray] = []
        self._n_features = 0
        self._fitted = False

    def fit(self, X: np.ndarray) -> KBinsBooleanizer:
        X = self._as_2d_float(X)
        self._n_features = int(X.shape[1])
        self.bin_edges_ = []

        for feature in range(X.shape[1]):
            col = X[:, feature]
            if self.strategy == "uniform":
                edges = self._uniform_edges(col)
            else:
                edges = self._quantile_edges(col)
            self.bin_edges_.append(edges.astype(np.float32, copy=False))

        self._fitted = True
        return self

    def transform(self, X: np.ndarray, pack_bits: bool = False) -> np.ndarray:
        self._require_fitted()
        X = self._as_2d_float(X)
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"expected {self._n_features} features, got {X.shape[1]}"
            )

        n_bits = self.n_bits
        out = np.empty((X.shape[0], n_bits), dtype=np.uint8)
        offset = 0
        for feature, edges in enumerate(self.bin_edges_):
            width = edges.size
            if width:
                out[:, offset : offset + width] = X[:, [feature]] >= edges
            offset += width

        if not pack_bits:
            return out

        pad = (-n_bits) % 8
        if pad:
            out = np.pad(out, ((0, 0), (0, pad)), mode="constant")
        return np.packbits(out, axis=1)

    def fit_transform(self, X: np.ndarray, pack_bits: bool = False) -> np.ndarray:
        return self.fit(X).transform(X, pack_bits=pack_bits)

    def save_json(self, path: str | Path) -> None:
        self._require_fitted()
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> KBinsBooleanizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "version": "KBinsBooleanizer",
            "n_bins": self.n_bins,
            "strategy": self.strategy,
            "n_features_in": self._n_features,
            "bin_edges": [edges.tolist() for edges in self.bin_edges_],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> KBinsBooleanizer:
        obj = cls(n_bins=int(payload["n_bins"]), strategy=payload["strategy"])
        obj.bin_edges_ = [
            np.asarray(edges, dtype=np.float32) for edges in payload["bin_edges"]
        ]
        obj._n_features = int(payload["n_features_in"])
        obj._fitted = True
        return obj

    @property
    def n_bits(self) -> int:
        self._require_fitted()
        return int(sum(edges.size for edges in self.bin_edges_))

    @property
    def n_features_in(self) -> int:
        return int(self._n_features)

    @property
    def feature_indices(self) -> np.ndarray:
        self._require_fitted()
        return np.repeat(
            np.arange(self._n_features, dtype=np.int32),
            [edges.size for edges in self.bin_edges_],
        )

    @property
    def thresholds(self) -> np.ndarray:
        self._require_fitted()
        if not self.bin_edges_:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self.bin_edges_).astype(np.float32, copy=False)

    def _uniform_edges(self, col: np.ndarray) -> np.ndarray:
        lo = float(np.min(col))
        hi = float(np.max(col))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.empty(0, dtype=np.float32)
        return np.linspace(lo, hi, self.n_bins + 1, dtype=np.float64)[1:-1]

    def _quantile_edges(self, col: np.ndarray) -> np.ndarray:
        if float(np.max(col)) <= float(np.min(col)):
            return np.empty(0, dtype=np.float32)
        percentiles = np.linspace(0.0, 100.0, self.n_bins + 1)[1:-1]
        edges = np.percentile(col, percentiles)
        return np.unique(edges)

    @staticmethod
    def _as_2d_float(X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"expected a 2D matrix, got shape {arr.shape}")
        return arr

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("KBinsBooleanizer is not fitted")
