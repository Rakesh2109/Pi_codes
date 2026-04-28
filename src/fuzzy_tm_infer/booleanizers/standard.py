from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class StandardBinarizer:
    """TMU-style threshold booleanizer.

    For each feature, learn sorted non-minimum unique values as thermometer
    thresholds. If a feature has more thresholds than ``max_bits_per_feature``,
    sample them evenly across the unique-value list.
    """

    def __init__(self, max_bits_per_feature: int = 25):
        if max_bits_per_feature < 1:
            raise ValueError("max_bits_per_feature must be >= 1")
        self.max_bits_per_feature = int(max_bits_per_feature)
        self.unique_values: list[np.ndarray] = []
        self.number_of_features = 0
        self._n_features = 0
        self._fitted = False

    def fit(self, X: np.ndarray) -> StandardBinarizer:
        X = self._as_2d(X)
        self._n_features = int(X.shape[1])
        self.unique_values = []
        self.number_of_features = 0

        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])[1:]
            if values.size > self.max_bits_per_feature:
                values = self._sample_thresholds(values, self.max_bits_per_feature)
            values = np.asarray(values, dtype=X.dtype)
            self.unique_values.append(values)
            self.number_of_features += int(values.size)

        self._fitted = True
        return self

    def transform(self, X: np.ndarray, pack_bits: bool = False) -> np.ndarray:
        self._require_fitted()
        X = self._as_2d(X)
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"expected {self._n_features} features, got {X.shape[1]}"
            )

        out = np.zeros((X.shape[0], self.number_of_features), dtype=np.uint8)
        offset = 0
        for feature, thresholds in enumerate(self.unique_values):
            width = thresholds.size
            if width:
                out[:, offset : offset + width] = X[:, [feature]] >= thresholds
            offset += width

        if not pack_bits:
            return out

        pad = (-out.shape[1]) % 8
        if pad:
            out = np.pad(out, ((0, 0), (0, pad)), mode="constant")
        return np.packbits(out, axis=1)

    def fit_transform(self, X: np.ndarray, pack_bits: bool = False) -> np.ndarray:
        return self.fit(X).transform(X, pack_bits=pack_bits)

    def save_json(self, path: str | Path) -> None:
        self._require_fitted()
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> StandardBinarizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "version": "StandardBinarizer",
            "max_bits_per_feature": self.max_bits_per_feature,
            "n_features_in": self._n_features,
            "number_of_features": self.number_of_features,
            "unique_values": [values.tolist() for values in self.unique_values],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StandardBinarizer:
        obj = cls(max_bits_per_feature=int(payload["max_bits_per_feature"]))
        obj.unique_values = [
            np.asarray(values, dtype=np.float64) for values in payload["unique_values"]
        ]
        obj._n_features = int(payload["n_features_in"])
        obj.number_of_features = int(payload["number_of_features"])
        obj._fitted = True
        return obj

    @staticmethod
    def _sample_thresholds(values: np.ndarray, n: int) -> np.ndarray:
        selected = []
        step = float(values.size) / float(n)
        pos = 0.0
        while int(pos) < values.size and len(selected) < n:
            selected.append(values[int(pos)])
            pos += step
        return np.asarray(selected, dtype=values.dtype)

    @staticmethod
    def _as_2d(X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError(f"expected a 2D matrix, got shape {arr.shape}")
        return arr

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("StandardBinarizer is not fitted")
