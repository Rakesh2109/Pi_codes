from __future__ import annotations

import numpy as np


class ThermometerBinarizer:
    """Fixed-threshold thermometer encoder for dense numeric tensors.

    The default range matches TMU's color thermometer convention for image
    bytes: thresholds are evenly spaced inside ``[0, 255]`` and exclude both
    endpoints.
    """

    def __init__(self, resolution: int = 8, value_min: float = 0.0, value_max: float = 255.0):
        if resolution < 2:
            raise ValueError("resolution must be >= 2")
        if value_max <= value_min:
            raise ValueError("value_max must be greater than value_min")
        self.resolution = int(resolution)
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        self.thresholds = np.linspace(
            self.value_min,
            self.value_max,
            self.resolution + 1,
            dtype=np.float32,
        )[1:-1]

    def fit(self, X: np.ndarray) -> ThermometerBinarizer:
        np.asarray(X)
        return self

    def transform(self, X: np.ndarray, pack_bits: bool = False) -> np.ndarray:
        X = np.asarray(X)
        out = (X[..., np.newaxis] >= self.thresholds).astype(np.uint8)

        if X.ndim >= 4 and X.shape[-1] == 3:
            n_rows, height, width, channels, n_thresholds = out.shape
            out = out.transpose(0, 1, 2, 4, 3).reshape(
                n_rows,
                height,
                width,
                channels * n_thresholds,
            )

        if not pack_bits:
            return out

        flat = out.reshape(out.shape[0], -1)
        pad = (-flat.shape[1]) % 8
        if pad:
            flat = np.pad(flat, ((0, 0), (0, pad)), mode="constant")
        return np.packbits(flat, axis=1)

    def fit_transform(self, X: np.ndarray, pack_bits: bool = False) -> np.ndarray:
        return self.fit(X).transform(X, pack_bits=pack_bits)
