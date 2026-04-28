from __future__ import annotations

import ctypes as ct
import re
import subprocess
from contextlib import suppress
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..._logging import logger

Version = str
Float32Array = NDArray[np.float32]
Int32Array = NDArray[np.int32]


_HERE = Path(__file__).resolve().parent
_FUZZY_TM_DIR = _HERE / "fuzzy_tm"
DEFAULT_VERSION = "v17"


class FuzzyTMModel:
    """NumPy-friendly wrapper for the self-contained Fuzzy TM inference libraries."""

    __slots__ = ("version", "_lib", "_handle", "_n_literals", "_n_classes", "_h_words")

    version: Version
    _lib: ct.CDLL
    _handle: ct.c_void_p
    _n_literals: int
    _n_classes: int
    _h_words: int

    def __init__(
        self,
        model_path: str | Path,
        *,
        version: Version = DEFAULT_VERSION,
        build: bool = True,
    ) -> None:
        self.version = version
        lib = _load_library(version, build=build)
        handle = lib.fuzzy_tm_model_load(str(model_path).encode())
        if not handle:
            raise RuntimeError(f"failed to load Fuzzy TM model: {model_path}")

        self._lib = lib
        self._handle = ct.c_void_p(handle)
        self._n_literals = int(lib.fuzzy_tm_n_literals(self._handle))
        self._n_classes = int(lib.fuzzy_tm_n_classes(self._handle))
        self._h_words = int(lib.fuzzy_tm_h_words(self._handle))

    @classmethod
    def from_fbz(
        cls,
        path: str | Path,
        *,
        version: Version = DEFAULT_VERSION,
        build: bool = True,
    ) -> FuzzyTMModel:
        return cls(path, version=version, build=build)

    @property
    def n_literals(self) -> int:
        return self._n_literals

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def h_words(self) -> int:
        return self._h_words

    def calibrate(self, rows: Float32Array, *, verbose: bool = False) -> None:
        x = _as_float32_matrix(rows)
        self._lib.fuzzy_tm_model_calibrate(
            self._handle,
            _float_ptr(x),
            ct.c_uint32(x.shape[0]),
            ct.c_uint32(x.shape[1]),
            ct.c_int(1 if verbose else 0),
        )

    def predict(self, row: Float32Array) -> int:
        x = np.ascontiguousarray(row, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError(f"predict() expects a 1D row, got shape {x.shape}")
        return int(self._lib.fuzzy_tm_predict_row(self._handle, _float_ptr(x)))

    def predict_batch(self, rows: Float32Array, *, calibrate: bool = False) -> Int32Array:
        x = _as_float32_matrix(rows)
        if calibrate:
            self.calibrate(x)
        out = np.empty(x.shape[0], dtype=np.int32)
        self._lib.fuzzy_tm_predict_batch(
            self._handle,
            _float_ptr(x),
            ct.c_uint32(x.shape[0]),
            ct.c_uint32(x.shape[1]),
            out.ctypes.data_as(ct.POINTER(ct.c_int32)),
        )
        return out

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._lib.fuzzy_tm_model_free(handle)
            self._handle = ct.c_void_p()

    def __enter__(self) -> FuzzyTMModel:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


def _version_dir(version: Version) -> Path:
    directory = _FUZZY_TM_DIR / version
    if not directory.is_dir():
        choices = ", ".join(available_versions()) or "<none>"
        raise ValueError(f"unknown Fuzzy TM version {version!r}; available: {choices}")
    return directory


def available_versions() -> list[str]:
    if not _FUZZY_TM_DIR.is_dir():
        return []
    return sorted(
        (
            path.name
            for path in _FUZZY_TM_DIR.iterdir()
            if path.is_dir() and (path / "Makefile").exists()
        ),
        key=_version_sort_key,
    )


def build_library(version: Version = DEFAULT_VERSION) -> Path:
    directory = _version_dir(version)
    logger.info("building native Fuzzy TM library {}", directory)
    subprocess.run(["make", "-C", str(directory), "lib"], check=True)
    return directory / "libfuzzy_tm_infer.so"


def _load_library(version: Version, *, build: bool) -> ct.CDLL:
    lib_path = _version_dir(version) / "libfuzzy_tm_infer.so"
    if build and not lib_path.exists():
        build_library(version)
    if not lib_path.exists():
        raise FileNotFoundError(f"{lib_path} does not exist; run build_library({version!r})")

    lib = ct.CDLL(str(lib_path))
    lib.fuzzy_tm_model_load.argtypes = [ct.c_char_p]
    lib.fuzzy_tm_model_load.restype = ct.c_void_p
    lib.fuzzy_tm_model_free.argtypes = [ct.c_void_p]
    lib.fuzzy_tm_model_free.restype = None
    lib.fuzzy_tm_n_literals.argtypes = [ct.c_void_p]
    lib.fuzzy_tm_n_literals.restype = ct.c_uint16
    lib.fuzzy_tm_n_classes.argtypes = [ct.c_void_p]
    lib.fuzzy_tm_n_classes.restype = ct.c_uint16
    lib.fuzzy_tm_h_words.argtypes = [ct.c_void_p]
    lib.fuzzy_tm_h_words.restype = ct.c_uint32
    lib.fuzzy_tm_model_calibrate.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_float),
        ct.c_uint32,
        ct.c_uint32,
        ct.c_int,
    ]
    lib.fuzzy_tm_model_calibrate.restype = None
    lib.fuzzy_tm_predict_row.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float)]
    lib.fuzzy_tm_predict_row.restype = ct.c_int32
    lib.fuzzy_tm_predict_batch.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_float),
        ct.c_uint32,
        ct.c_uint32,
        ct.POINTER(ct.c_int32),
    ]
    lib.fuzzy_tm_predict_batch.restype = None
    return lib


def _version_sort_key(version: str) -> tuple[int, str]:
    match = re.fullmatch(r"v(\d+)", version)
    if match:
        return int(match.group(1)), version
    return 10**9, version


def _as_float32_matrix(rows: Float32Array) -> Float32Array:
    x = np.ascontiguousarray(rows, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected a 2D NumPy array, got shape {x.shape}")
    return x


def _float_ptr(x: Float32Array) -> ct.POINTER(ct.c_float):
    return x.ctypes.data_as(ct.POINTER(ct.c_float))
