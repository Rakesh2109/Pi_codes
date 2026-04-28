from __future__ import annotations

import ctypes as ct
import subprocess
from collections.abc import Callable, Iterable, Iterator
from contextlib import suppress
from pathlib import Path
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from ..._logging import logger
from ...booleanizers.glade import GLADEBooleanizer

Float64Array = NDArray[np.float64]
Float32Array = NDArray[np.float32]
UInt8Array = NDArray[np.uint8]
UInt32Array = NDArray[np.uint32]
T = TypeVar("T")

_HERE = Path(__file__).resolve().parent
_GLADE_DIR = _HERE / "booleanizers" / "glade_v2"
_LIB_NAME = "libglade_v2.so"


class FastGLADEBooleanizer:
    """GLADE v2 with Python fitting and native C transform kernels."""

    __slots__ = (
        "_chunk_features4",
        "_chunk_features8",
        "_chunk_indices4",
        "_chunk_thresholds4",
        "_feature_indices",
        "_lib",
        "_model",
        "_n_features",
        "_thresholds32",
        "_thresholds64",
    )

    _chunk_features4: UInt32Array | None
    _chunk_features8: UInt32Array | None
    _chunk_indices4: UInt32Array | None
    _chunk_thresholds4: Float32Array | None
    _feature_indices: UInt32Array | None
    _thresholds32: Float32Array | None
    _thresholds64: Float64Array | None
    _n_features: int
    _model: GLADEBooleanizer
    _lib: ct.CDLL

    def __init__(self, n_bins: int = 15, *, build: bool = True) -> None:
        self._model = GLADEBooleanizer(n_bins=n_bins)
        self._chunk_features4 = None
        self._chunk_features8 = None
        self._chunk_indices4 = None
        self._chunk_thresholds4 = None
        self._feature_indices = None
        self._thresholds32 = None
        self._thresholds64 = None
        self._n_features = 0
        self._lib = _load_library(build=build)

    @property
    def backend(self) -> str:
        return self._lib.glade_v2_backend().decode()

    @property
    def n_bits(self) -> int:
        return int(self._thresholds32.size) if self._thresholds32 is not None else 0

    @property
    def n_features_in(self) -> int:
        return int(self._n_features)

    @property
    def feature_indices(self) -> NDArray[np.int32]:
        self._require_fitted()
        assert self._feature_indices is not None
        return self._feature_indices.astype(np.int32, copy=True)

    @property
    def thresholds(self) -> NDArray[np.float32]:
        self._require_fitted()
        assert self._thresholds32 is not None
        return self._thresholds32.copy()

    def fit(self, X: np.ndarray) -> FastGLADEBooleanizer:
        self._model.fit(X)
        self._sync_from_model(self._model)
        return self

    def transform(self, X: np.ndarray, pack_bits: bool = False) -> UInt8Array:
        x = self._as_transform_matrix(X)
        out = self.empty_output(x.shape[0], pack_bits=pack_bits)
        return self.transform_into(x, out, pack_bits=pack_bits)

    def transform_into(
        self,
        X: np.ndarray,
        out: UInt8Array,
        *,
        pack_bits: bool = False,
        trusted: bool = True,
    ) -> UInt8Array:
        """Transform rows into an existing output array.

        ``trusted=True`` uses the C fast path that skips repeated feature-index
        validation. Exactness is unchanged because the wrapper has already
        validated row shape, output shape, and owns the fitted threshold arrays.
        """
        self._require_fitted()
        x = self._as_transform_matrix(X)
        y = np.asarray(out)
        expected = self.output_shape(x.shape[0], pack_bits=pack_bits)
        if y.shape != expected:
            raise ValueError(f"expected output shape {expected}, got {y.shape}")
        if y.dtype != np.uint8:
            raise ValueError(f"expected uint8 output, got {y.dtype}")
        if not y.flags.c_contiguous:
            raise ValueError("output array must be C-contiguous")

        assert self._feature_indices is not None
        assert self._chunk_features4 is not None
        assert self._chunk_features8 is not None
        assert self._chunk_indices4 is not None
        assert self._chunk_thresholds4 is not None
        assert self._thresholds32 is not None
        assert self._thresholds64 is not None

        n_rows = int(x.shape[0])
        n_bits = int(self._thresholds32.size)
        if pack_bits:
            if x.dtype == np.float64:
                rc = self._lib.glade_v2_transform_packed(
                    _double_ptr(x),
                    ct.c_uint32(n_rows),
                    ct.c_uint32(self._n_features),
                    _uint32_ptr(self._feature_indices),
                    _double_ptr(self._thresholds64),
                    ct.c_uint32(n_bits),
                    _uint8_ptr(y),
                )
            elif trusted and self.backend == "neon":
                rc = self._lib.glade_v2_transform_packed_chunks4_f32_unchecked(
                    _float_ptr(x),
                    ct.c_uint32(n_rows),
                    ct.c_uint32(self._n_features),
                    _uint32_ptr(self._feature_indices),
                    _uint32_ptr(self._chunk_features4),
                    _uint32_ptr(self._chunk_indices4),
                    _float_ptr(self._thresholds32),
                    _float_ptr(self._chunk_thresholds4),
                    ct.c_uint32(n_bits),
                    _uint8_ptr(y),
                )
            elif trusted:
                rc = self._lib.glade_v2_transform_packed_chunked_f32_unchecked(
                    _float_ptr(x),
                    ct.c_uint32(n_rows),
                    ct.c_uint32(self._n_features),
                    _uint32_ptr(self._feature_indices),
                    _uint32_ptr(self._chunk_features4),
                    _uint32_ptr(self._chunk_features8),
                    _float_ptr(self._thresholds32),
                    ct.c_uint32(n_bits),
                    _uint8_ptr(y),
                )
            else:
                rc = self._lib.glade_v2_transform_packed_chunked_f32(
                    _float_ptr(x),
                    ct.c_uint32(n_rows),
                    ct.c_uint32(self._n_features),
                    _uint32_ptr(self._feature_indices),
                    _uint32_ptr(self._chunk_features4),
                    _uint32_ptr(self._chunk_features8),
                    _float_ptr(self._thresholds32),
                    ct.c_uint32(n_bits),
                    _uint8_ptr(y),
                )
        else:
            if x.dtype == np.float64:
                rc = self._lib.glade_v2_transform_u8(
                    _double_ptr(x),
                    ct.c_uint32(n_rows),
                    ct.c_uint32(self._n_features),
                    _uint32_ptr(self._feature_indices),
                    _double_ptr(self._thresholds64),
                    ct.c_uint32(n_bits),
                    _uint8_ptr(y),
                )
            elif trusted and self.backend == "neon":
                rc = self._lib.glade_v2_transform_u8_chunks4_f32_unchecked(
                    _float_ptr(x),
                    ct.c_uint32(n_rows),
                    ct.c_uint32(self._n_features),
                    _uint32_ptr(self._feature_indices),
                    _uint32_ptr(self._chunk_features4),
                    _uint32_ptr(self._chunk_indices4),
                    _float_ptr(self._thresholds32),
                    _float_ptr(self._chunk_thresholds4),
                    ct.c_uint32(n_bits),
                    _uint8_ptr(y),
                )
            elif trusted:
                rc = self._lib.glade_v2_transform_u8_chunked_f32_unchecked(
                    _float_ptr(x),
                    ct.c_uint32(n_rows),
                    ct.c_uint32(self._n_features),
                    _uint32_ptr(self._feature_indices),
                    _uint32_ptr(self._chunk_features4),
                    _uint32_ptr(self._chunk_features8),
                    _float_ptr(self._thresholds32),
                    ct.c_uint32(n_bits),
                    _uint8_ptr(y),
                )
            else:
                rc = self._lib.glade_v2_transform_u8_chunked_f32(
                    _float_ptr(x),
                    ct.c_uint32(n_rows),
                    ct.c_uint32(self._n_features),
                    _uint32_ptr(self._feature_indices),
                    _uint32_ptr(self._chunk_features4),
                    _uint32_ptr(self._chunk_features8),
                    _float_ptr(self._thresholds32),
                    ct.c_uint32(n_bits),
                    _uint8_ptr(y),
                )
        if rc != 0:
            raise RuntimeError(f"fast GLADE transform failed with code {rc}")
        return y

    def output_shape(self, n_rows: int, *, pack_bits: bool = False) -> tuple[int, int]:
        self._require_fitted()
        n = int(n_rows)
        if n < 0:
            raise ValueError("n_rows must be non-negative")
        n_bits = self.n_bits
        return (n, (n_bits + 7) // 8) if pack_bits else (n, n_bits)

    def empty_output(self, n_rows: int, *, pack_bits: bool = False) -> UInt8Array:
        return np.empty(self.output_shape(n_rows, pack_bits=pack_bits), dtype=np.uint8)

    def transform_stream(
        self,
        rows: np.ndarray | Iterable[np.ndarray],
        *,
        batch_size: int = 4096,
        pack_bits: bool = False,
    ) -> Iterator[UInt8Array]:
        """Yield transformed chunks while keeping memory bounded."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if isinstance(rows, np.ndarray):
            for start in range(0, rows.shape[0], batch_size):
                yield self.transform(rows[start : start + batch_size], pack_bits=pack_bits)
            return

        for batch in rows:
            yield self.transform(batch, pack_bits=pack_bits)

    def consume_stream(
        self,
        rows: np.ndarray | Iterable[np.ndarray],
        consumer: Callable[[UInt8Array], T],
        *,
        batch_size: int = 4096,
        pack_bits: bool = False,
    ) -> Iterator[T]:
        """Booleanize streaming chunks and pass each chunk to a consumer."""
        for chunk in self.transform_stream(
            rows,
            batch_size=batch_size,
            pack_bits=pack_bits,
        ):
            yield consumer(chunk)

    def fit_transform(self, X: np.ndarray, pack_bits: bool = False) -> UInt8Array:
        return self.fit(X).transform(X, pack_bits=pack_bits)

    @property
    def profile_enabled(self) -> bool:
        return bool(self._lib.glade_v2_profile_enabled())

    def profile_reset(self) -> None:
        self._lib.glade_v2_profile_reset()

    def profile_snapshot(self) -> dict[str, int]:
        count = int(self._lib.glade_v2_profile_field_count())
        values = np.zeros(count, dtype=np.uint64)
        rc = int(
            self._lib.glade_v2_profile_read(
                values.ctypes.data_as(ct.POINTER(ct.c_uint64)),
                ct.c_uint32(count),
            )
        )
        if rc < 0:
            raise RuntimeError(f"fast GLADE profile read failed with code {rc}")
        return {
            self._lib.glade_v2_profile_field_name(ct.c_uint32(i)).decode(): int(
                values[i]
            )
            for i in range(count)
        }

    @classmethod
    def from_booleanizer(
        cls, model: GLADEBooleanizer, *, build: bool = True
    ) -> FastGLADEBooleanizer:
        obj = cls(n_bins=model.n_bins, build=build)
        obj._sync_from_model(model)
        return obj

    @classmethod
    def load_json(
        cls, path: str | Path, *, build: bool = True
    ) -> FastGLADEBooleanizer:
        return cls.from_booleanizer(GLADEBooleanizer.load_json(path), build=build)

    def save_json(self, path: str | Path, quantise_int16: bool = False) -> None:
        self._model.save_json(path, quantise_int16=quantise_int16)

    def _sync_from_model(self, model: GLADEBooleanizer) -> None:
        self._model = model
        n_features = int(model.n_features_in)
        self._feature_indices = np.ascontiguousarray(
            model.feature_indices, dtype=np.uint32
        )
        if self._feature_indices.ndim != 1:
            raise ValueError("feature indices must be a 1D array")
        if self._feature_indices.size and int(self._feature_indices.max()) >= n_features:
            raise ValueError("feature indices exceed fitted feature count")
        self._chunk_features4 = _make_chunk_features4(self._feature_indices)
        self._chunk_features8 = _make_chunk_features8(self._feature_indices)
        self._thresholds32 = np.ascontiguousarray(model.thresholds, dtype=np.float32)
        if self._thresholds32.ndim != 1:
            raise ValueError("thresholds must be a 1D array")
        if self._thresholds32.size != self._feature_indices.size:
            raise ValueError("threshold and feature-index counts differ")
        n_chunks4 = int(self._feature_indices.size) // 4
        self._chunk_indices4 = np.ascontiguousarray(
            self._feature_indices[: n_chunks4 * 4],
            dtype=np.uint32,
        )
        self._chunk_thresholds4 = np.ascontiguousarray(
            self._thresholds32[: n_chunks4 * 4],
            dtype=np.float32,
        )
        self._thresholds64 = np.ascontiguousarray(self._thresholds32, dtype=np.float64)
        self._n_features = n_features

    def _require_fitted(self) -> None:
        if (
            self._feature_indices is None
            or self._chunk_features4 is None
            or self._chunk_features8 is None
            or self._chunk_indices4 is None
            or self._chunk_thresholds4 is None
            or self._thresholds32 is None
            or self._thresholds64 is None
        ):
            raise RuntimeError("FastGLADEBooleanizer is not fitted")

    def _as_transform_matrix(self, X: np.ndarray) -> Float32Array | Float64Array:
        source = np.asarray(X)
        dtype = np.float64 if source.dtype == np.float64 else np.float32
        x = np.ascontiguousarray(source, dtype=dtype)
        if x.ndim != 2:
            raise ValueError(f"expected a 2D matrix, got shape {x.shape}")
        if x.shape[1] != self._n_features:
            raise ValueError(f"expected {self._n_features} features, got {x.shape[1]}")
        return x


def build_glade_library() -> Path:
    logger.info("building Fast GLADE v2 library {}", _GLADE_DIR)
    subprocess.run(["make", "-C", str(_GLADE_DIR), "lib"], check=True)
    return _GLADE_DIR / _LIB_NAME


def _load_library(*, build: bool) -> ct.CDLL:
    lib_path = _GLADE_DIR / _LIB_NAME
    if build and not lib_path.exists():
        build_glade_library()
    if not lib_path.exists():
        raise FileNotFoundError(f"{lib_path} does not exist; run build_glade_library()")

    lib = ct.CDLL(str(lib_path))
    lib.glade_v2_backend.argtypes = []
    lib.glade_v2_backend.restype = ct.c_char_p
    lib.glade_v2_profile_field_count.argtypes = []
    lib.glade_v2_profile_field_count.restype = ct.c_uint32
    lib.glade_v2_profile_field_name.argtypes = [ct.c_uint32]
    lib.glade_v2_profile_field_name.restype = ct.c_char_p
    lib.glade_v2_profile_enabled.argtypes = []
    lib.glade_v2_profile_enabled.restype = ct.c_int
    lib.glade_v2_profile_reset.argtypes = []
    lib.glade_v2_profile_reset.restype = None
    lib.glade_v2_profile_read.argtypes = [
        ct.POINTER(ct.c_uint64),
        ct.c_uint32,
    ]
    lib.glade_v2_profile_read.restype = ct.c_int
    lib.glade_v2_transform_u8.argtypes = _transform_argtypes()
    lib.glade_v2_transform_u8.restype = ct.c_int
    lib.glade_v2_transform_u8_f32.argtypes = _transform_argtypes_f32()
    lib.glade_v2_transform_u8_f32.restype = ct.c_int
    lib.glade_v2_transform_u8_chunked_f32.argtypes = _transform_chunked_argtypes_f32()
    lib.glade_v2_transform_u8_chunked_f32.restype = ct.c_int
    lib.glade_v2_transform_u8_chunked_f32_unchecked.argtypes = (
        _transform_chunked_argtypes_f32()
    )
    lib.glade_v2_transform_u8_chunked_f32_unchecked.restype = ct.c_int
    lib.glade_v2_transform_u8_chunks4_f32_unchecked.argtypes = (
        _transform_chunks4_argtypes_f32()
    )
    lib.glade_v2_transform_u8_chunks4_f32_unchecked.restype = ct.c_int
    lib.glade_v2_transform_packed.argtypes = _transform_argtypes()
    lib.glade_v2_transform_packed.restype = ct.c_int
    lib.glade_v2_transform_packed_f32.argtypes = _transform_argtypes_f32()
    lib.glade_v2_transform_packed_f32.restype = ct.c_int
    lib.glade_v2_transform_packed_chunked_f32.argtypes = (
        _transform_chunked_argtypes_f32()
    )
    lib.glade_v2_transform_packed_chunked_f32.restype = ct.c_int
    lib.glade_v2_transform_packed_chunked_f32_unchecked.argtypes = (
        _transform_chunked_argtypes_f32()
    )
    lib.glade_v2_transform_packed_chunked_f32_unchecked.restype = ct.c_int
    lib.glade_v2_transform_packed_chunks4_f32_unchecked.argtypes = (
        _transform_chunks4_argtypes_f32()
    )
    lib.glade_v2_transform_packed_chunks4_f32_unchecked.restype = ct.c_int
    return lib


def _transform_argtypes() -> list[object]:
    return [
        ct.POINTER(ct.c_double),
        ct.c_uint32,
        ct.c_uint32,
        ct.POINTER(ct.c_uint32),
        ct.POINTER(ct.c_double),
        ct.c_uint32,
        ct.POINTER(ct.c_uint8),
    ]


def _transform_argtypes_f32() -> list[object]:
    return [
        ct.POINTER(ct.c_float),
        ct.c_uint32,
        ct.c_uint32,
        ct.POINTER(ct.c_uint32),
        ct.POINTER(ct.c_float),
        ct.c_uint32,
        ct.POINTER(ct.c_uint8),
    ]


def _transform_chunked_argtypes_f32() -> list[object]:
    return [
        ct.POINTER(ct.c_float),
        ct.c_uint32,
        ct.c_uint32,
        ct.POINTER(ct.c_uint32),
        ct.POINTER(ct.c_uint32),
        ct.POINTER(ct.c_uint32),
        ct.POINTER(ct.c_float),
        ct.c_uint32,
        ct.POINTER(ct.c_uint8),
    ]


def _transform_chunks4_argtypes_f32() -> list[object]:
    return [
        ct.POINTER(ct.c_float),
        ct.c_uint32,
        ct.c_uint32,
        ct.POINTER(ct.c_uint32),
        ct.POINTER(ct.c_uint32),
        ct.POINTER(ct.c_uint32),
        ct.POINTER(ct.c_float),
        ct.POINTER(ct.c_float),
        ct.c_uint32,
        ct.POINTER(ct.c_uint8),
    ]


def _double_ptr(x: Float64Array) -> ct.POINTER(ct.c_double):
    return x.ctypes.data_as(ct.POINTER(ct.c_double))


def _float_ptr(x: Float32Array) -> ct.POINTER(ct.c_float):
    return x.ctypes.data_as(ct.POINTER(ct.c_float))


def _uint32_ptr(x: UInt32Array) -> ct.POINTER(ct.c_uint32):
    return x.ctypes.data_as(ct.POINTER(ct.c_uint32))


def _uint8_ptr(x: UInt8Array) -> ct.POINTER(ct.c_uint8):
    return x.ctypes.data_as(ct.POINTER(ct.c_uint8))


def fast_glade_available(*, build: bool = False) -> bool:
    with suppress(Exception):
        _load_library(build=build)
        return True
    return False


def _make_chunk_features4(feature_indices: UInt32Array) -> UInt32Array:
    n_chunks = int(feature_indices.size) // 4
    mixed = np.iinfo(np.uint32).max
    out = np.empty(n_chunks, dtype=np.uint32)
    for chunk in range(n_chunks):
        start = chunk * 4
        values = feature_indices[start : start + 4]
        first = int(values[0])
        out[chunk] = first if bool(np.all(values == first)) else mixed
    return np.ascontiguousarray(out, dtype=np.uint32)


def _make_chunk_features8(feature_indices: UInt32Array) -> UInt32Array:
    n_chunks = int(feature_indices.size) // 8
    mixed = np.iinfo(np.uint32).max
    out = np.empty(n_chunks, dtype=np.uint32)
    for chunk in range(n_chunks):
        start = chunk * 8
        values = feature_indices[start : start + 8]
        first = int(values[0])
        out[chunk] = first if bool(np.all(values == first)) else mixed
    return np.ascontiguousarray(out, dtype=np.uint32)
