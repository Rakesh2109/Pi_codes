from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import ASSETS_DIR, ML_MODELS_DIR

Float32Array = NDArray[np.float32]
Int32Array = NDArray[np.int32]


def load_X(dataset_name: str, assets_dir: Path = ASSETS_DIR) -> Float32Array:
    path = (
        assets_dir
        / "datasets"
        / f"{dataset_name.lower()}_test"
        / f"{dataset_name}_X_test_raw.bin"
    )
    with path.open("rb") as f:
        n_rows, n_features = struct.unpack("<II", f.read(8))
        data: Float32Array = np.frombuffer(
            f.read(n_rows * n_features * 4), dtype=np.float32
        )
    return data.reshape(n_rows, n_features).copy()


def load_y(dataset_name: str, assets_dir: Path = ASSETS_DIR) -> Int32Array:
    path = (
        assets_dir
        / "datasets"
        / f"{dataset_name.lower()}_test"
        / f"{dataset_name}_Y_test.txt"
    )
    return np.loadtxt(path, dtype=np.int32)


def model_path(stem: str, assets_dir: Path = ASSETS_DIR) -> Path:
    return assets_dir / "tm_models" / f"{stem}_model.fbz"


def dt_model_path(stem: str, assets_dir: Path = ML_MODELS_DIR) -> Path:
    return assets_dir / stem / "DecisionTree.pkl"


def load_dt_testset(
    stem: str,
    assets_dir: Path = ML_MODELS_DIR,
) -> tuple[Float32Array, Int32Array]:
    with np.load(assets_dir / stem / "testset.npz") as data:
        return data["X_te"].astype(np.float32), data["Y_te"].astype(np.int32)
