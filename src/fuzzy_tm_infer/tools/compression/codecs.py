from __future__ import annotations

import lzma
import struct
import zlib
from dataclasses import dataclass
from os import PathLike
from time import perf_counter
from typing import Protocol

import zstandard as zstd


@dataclass(frozen=True, slots=True)
class FBZPayload:
    compressed: bytes
    uncompressed: bytes


@dataclass(frozen=True, slots=True)
class CodecResult:
    name: str
    raw_bytes: int
    compressed_bytes: int
    ratio: float
    encode_us: float
    decode_us: float


class ByteCodec(Protocol):
    name: str

    def encode(self, data: bytes) -> bytes: ...

    def decode(self, data: bytes) -> bytes: ...


class PackBitsRLECodec:
    name = "rle-packbits"

    def encode(self, data: bytes) -> bytes:
        return rle_pack(data)

    def decode(self, data: bytes) -> bytes:
        return rle_unpack(data)


class PackBitsCodec(PackBitsRLECodec):
    name = "packbits"


class ZstdCodec:
    name = "zstd"

    def __init__(self, level: int = 22) -> None:
        self.level = level
        self.name = f"zstd-{level}"
        self._compressor = zstd.ZstdCompressor(level=level)
        self._decompressor = zstd.ZstdDecompressor()

    def encode(self, data: bytes) -> bytes:
        return self._compressor.compress(data)

    def decode(self, data: bytes) -> bytes:
        return self._decompressor.decompress(data)


class Lz4Codec:
    name = "lz4-hc"

    def encode(self, data: bytes) -> bytes:
        import lz4.frame  # type: ignore[import-not-found, import-untyped]

        return lz4.frame.compress(data, compression_level=16)

    def decode(self, data: bytes) -> bytes:
        import lz4.frame  # type: ignore[import-not-found, import-untyped]

        return lz4.frame.decompress(data)


class BrotliCodec:
    name = "brotli-11"

    def encode(self, data: bytes) -> bytes:
        import brotli  # type: ignore[import-not-found, import-untyped]

        return brotli.compress(data, quality=11)

    def decode(self, data: bytes) -> bytes:
        import brotli  # type: ignore[import-not-found, import-untyped]

        return brotli.decompress(data)


class LzmaCodec:
    name = "lzma-9e"

    def encode(self, data: bytes) -> bytes:
        return lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)

    def decode(self, data: bytes) -> bytes:
        return lzma.decompress(data)


class ZlibCodec:
    name = "zlib-9"

    def encode(self, data: bytes) -> bytes:
        return zlib.compress(data, level=9)

    def decode(self, data: bytes) -> bytes:
        return zlib.decompress(data)


class LibDeflateCodec:
    name = "libdeflate"

    def encode(self, data: bytes) -> bytes:
        import cramjam  # type: ignore[import-not-found, import-untyped]

        return bytes(cramjam.deflate.compress(data, level=9))

    def decode(self, data: bytes) -> bytes:
        import cramjam  # type: ignore[import-not-found, import-untyped]

        return bytes(cramjam.deflate.decompress(data))


class IsalDeflateCodec:
    name = "isal-deflate"

    def encode(self, data: bytes) -> bytes:
        from isal import igzip  # type: ignore[import-not-found, import-untyped]

        return igzip.compress(data, compresslevel=3)

    def decode(self, data: bytes) -> bytes:
        from isal import igzip  # type: ignore[import-not-found, import-untyped]

        return igzip.decompress(data)


class SnappyCodec:
    name = "snappy"

    def encode(self, data: bytes) -> bytes:
        import snappy  # type: ignore[import-not-found, import-untyped]

        return snappy.compress(data)

    def decode(self, data: bytes) -> bytes:
        import snappy  # type: ignore[import-not-found, import-untyped]

        return snappy.decompress(data)


class Blosc2Codec:
    name = "blosc2-zstd"

    def encode(self, data: bytes) -> bytes:
        import blosc2  # type: ignore[import-not-found, import-untyped]

        return blosc2.compress(data, typesize=1, clevel=9, codec=blosc2.Codec.ZSTD)

    def decode(self, data: bytes) -> bytes:
        import blosc2  # type: ignore[import-not-found, import-untyped]

        return blosc2.decompress(data)


class ZopfliCodec:
    name = "zopfli-zlib"

    def encode(self, data: bytes) -> bytes:
        import zopfli.zlib  # type: ignore[import-not-found, import-untyped]

        return zopfli.zlib.compress(data)

    def decode(self, data: bytes) -> bytes:
        return zlib.decompress(data)


def read_fbz_payload(path: str | PathLike[str]) -> FBZPayload:
    with open(path, "rb") as f:
        blob = f.read()

    hdr = "<4s B H H B I I I"
    magic, ver, n_literals, _n_classes, _cmax, _total, comp_sz, _uncomp_sz = (
        struct.unpack_from(hdr, blob, 0)
    )
    if magic != b"FBZ1" or ver != 1:
        raise ValueError(f"{path} is not an FBZ1 model")

    off = struct.calcsize(hdr)
    off += 4 * n_literals
    off += 4 * n_literals

    for _ in range(2):
        (n_strings,) = struct.unpack_from("<H", blob, off)
        off += 2
        for _ in range(n_strings):
            (length,) = struct.unpack_from("<H", blob, off)
            off += 2 + length

    compressed = blob[off : off + comp_sz]
    return FBZPayload(
        compressed=compressed,
        uncompressed=zstd.ZstdDecompressor().decompress(compressed),
    )


def rle_pack(data: bytes) -> bytes:
    out = bytearray()
    i = 0
    length = len(data)

    while i < length:
        run = 1
        while run < 128 and i + run < length and data[i + run] == data[i]:
            run += 1

        if run >= 3:
            out.append(257 - run)
            out.append(data[i])
            i += run
            continue

        literal_start = i
        literal_end = min(length, i + 128)
        i += run
        while i < literal_end:
            run = 1
            while run < 128 and i + run < length and data[i + run] == data[i]:
                run += 1
            if run >= 3:
                break
            i += run

        literal = data[literal_start:i]
        out.append(len(literal) - 1)
        out.extend(literal)

    return bytes(out)


def rle_unpack(data: bytes) -> bytes:
    out = bytearray()
    i = 0
    length = len(data)

    while i < length:
        control = data[i]
        i += 1
        if control < 128:
            n = control + 1
            out.extend(data[i : i + n])
            i += n
        elif control > 128:
            repeat = 257 - control
            out.extend([data[i]] * repeat)
            i += 1

    return bytes(out)


packbits_encode = rle_pack
packbits_decode = rle_unpack


def benchmark_codec(codec: ByteCodec, data: bytes, repeats: int = 20) -> CodecResult:
    encoded = codec.encode(data)
    if codec.decode(encoded) != data:
        raise ValueError(f"{codec.name} failed round-trip verification")

    t0 = perf_counter()
    for _ in range(repeats):
        encoded = codec.encode(data)
    encode_us = (perf_counter() - t0) / repeats * 1e6

    t0 = perf_counter()
    for _ in range(repeats):
        decoded = codec.decode(encoded)
    decode_us = (perf_counter() - t0) / repeats * 1e6
    if decoded != data:
        raise ValueError(f"{codec.name} failed timed round-trip verification")

    return CodecResult(
        name=codec.name,
        raw_bytes=len(data),
        compressed_bytes=len(encoded),
        ratio=len(encoded) / len(data) if data else 0.0,
        encode_us=encode_us,
        decode_us=decode_us,
    )
