#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from fuzzy_tm_infer.assets import ensure_assets
    from fuzzy_tm_infer.config import DATASETS
    from fuzzy_tm_infer.data import model_path
    from fuzzy_tm_infer.tools.compression.codecs import (
        Blosc2Codec,
        BrotliCodec,
        ByteCodec,
        CodecResult,
        IsalDeflateCodec,
        LibDeflateCodec,
        Lz4Codec,
        LzmaCodec,
        PackBitsRLECodec,
        SnappyCodec,
        ZlibCodec,
        ZopfliCodec,
        ZstdCodec,
        benchmark_codec,
        read_fbz_payload,
    )
else:
    from ...assets import ensure_assets
    from ...config import DATASETS
    from ...data import model_path
    from .codecs import (
        Blosc2Codec,
        BrotliCodec,
        ByteCodec,
        CodecResult,
        IsalDeflateCodec,
        LibDeflateCodec,
        Lz4Codec,
        LzmaCodec,
        PackBitsRLECodec,
        SnappyCodec,
        ZlibCodec,
        ZopfliCodec,
        ZstdCodec,
        benchmark_codec,
        read_fbz_payload,
    )


@dataclass(frozen=True, slots=True)
class TableRow:
    dataset: str
    codec: str
    raw_bytes: int | None
    compressed_bytes: int | None
    ratio: float | None
    encode_us: float | None
    decode_us: float | None
    error: str | None = None


def _codecs() -> tuple[ByteCodec, ...]:
    return (
        PackBitsRLECodec(),
        ZstdCodec(level=22),
        Lz4Codec(),
        BrotliCodec(),
        LzmaCodec(),
        ZlibCodec(),
        LibDeflateCodec(),
        IsalDeflateCodec(),
        SnappyCodec(),
        Blosc2Codec(),
        ZopfliCodec(),
    )


def _row_from_result(dataset: str, result: CodecResult) -> TableRow:
    return TableRow(
        dataset=dataset,
        codec=result.name,
        raw_bytes=result.raw_bytes,
        compressed_bytes=result.compressed_bytes,
        ratio=result.ratio,
        encode_us=result.encode_us,
        decode_us=result.decode_us,
    )


def _row_from_error(dataset: str, codec: ByteCodec, error: Exception) -> TableRow:
    return TableRow(
        dataset=dataset,
        codec=codec.name,
        raw_bytes=None,
        compressed_bytes=None,
        ratio=None,
        encode_us=None,
        decode_us=None,
        error=type(error).__name__,
    )


def collect_compression_results(repeats: int = 20) -> list[TableRow]:
    ensure_assets()
    rows: list[TableRow] = []
    for stem, name in DATASETS:
        payload = read_fbz_payload(model_path(stem))
        for codec in _codecs():
            try:
                rows.append(
                    _row_from_result(
                        name,
                        benchmark_codec(codec, payload.uncompressed, repeats),
                    )
                )
            except ImportError as error:
                rows.append(_row_from_error(name, codec, error))
    return rows


def _format_int(value: int | None) -> str:
    return f"{value}" if value is not None else "SKIP"


def _format_float(value: float | None, digits: int) -> str:
    return f"{value:.{digits}f}" if value is not None else "-"


def _best_size_rows(rows: list[TableRow]) -> set[TableRow]:
    best: set[TableRow] = set()
    datasets = {row.dataset for row in rows}
    for dataset in datasets:
        valid = [
            row
            for row in rows
            if row.dataset == dataset and row.compressed_bytes is not None
        ]
        if not valid:
            continue
        sizes = [row.compressed_bytes for row in valid if row.compressed_bytes is not None]
        min_size = min(sizes)
        best.update(row for row in valid if row.compressed_bytes == min_size)
    return best


def print_compression_table(rows: list[TableRow]) -> None:
    best_rows = _best_size_rows(rows)
    print("=" * 82)
    print("  FBZ BITMASK COMPRESSION COMPARISON")
    print("  * = smallest compressed size for that dataset")
    print("=" * 82)
    print(
        f"\n  {'Dataset':<10}  {'Codec':<13}  {'Raw B':>9}"
        f"  {'Cmp B':>9}  {'Ratio':>7}  {'Enc us':>10}  {'Dec us':>10}"
    )
    print("  " + "-" * 80)

    for row in rows:
        marker = "*" if row in best_rows else " "
        print(
            f"{marker} {row.dataset:<10}  {row.codec:<13}  {_format_int(row.raw_bytes):>9}"
            f"  {_format_int(row.compressed_bytes):>9}  {_format_float(row.ratio, 3):>7}"
            f"  {_format_float(row.encode_us, 2):>10}"
            f"  {(row.error or _format_float(row.decode_us, 2)):>10}"
        )


def save_table_png(rows: list[TableRow], out_path: Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore[import-not-found, import-untyped]

    best_rows = _best_size_rows(rows)
    headers = ("Dataset", "Codec", "Raw B", "Cmp B", "Ratio", "Enc us", "Dec us")
    table_rows = [
        (
            row.dataset,
            row.codec,
            _format_int(row.raw_bytes),
            _format_int(row.compressed_bytes),
            _format_float(row.ratio, 3),
            _format_float(row.encode_us, 2),
            row.error or _format_float(row.decode_us, 2),
        )
        for row in rows
    ]

    width = 10.8
    height = max(6.0, 0.22 * (len(table_rows) + 1) + 0.55)
    fig, ax = plt.subplots(figsize=(width, height), dpi=180)
    ax.axis("off")
    ax.set_title(
        "FBZ Bitmask Compression Comparison (bold = smallest compressed size)",
        fontsize=11,
        pad=4,
    )

    table = ax.table(
        cellText=table_rows,
        colLabels=headers,
        cellLoc="right",
        colLoc="right",
        colWidths=(0.12, 0.19, 0.11, 0.11, 0.09, 0.15, 0.15),
        bbox=(0.01, 0.0, 0.98, 0.95),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.8)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d7de")
        cell.set_linewidth(0.45)
        if row_idx == 0:
            cell.set_facecolor("#24292f")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f6f8fa")
        if row_idx > 0 and rows[row_idx - 1] in best_rows:
            cell.set_facecolor("#fff4ce")
            cell.get_text().set_weight("bold")
        if col_idx in (0, 1):
            cell.get_text().set_ha("left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def run_compression_comparison(
    repeats: int = 20,
    png_path: Path | None = None,
) -> None:
    rows = collect_compression_results(repeats)
    print_compression_table(rows)
    if png_path is not None:
        save_table_png(rows, png_path)
        print(f"\nPNG table written to: {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare compression codecs on local FBZ bitmask payloads."
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--png",
        type=Path,
        default=Path("compression_table.png"),
        help="PNG output path. Use --no-png to disable image output.",
    )
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args()

    run_compression_comparison(
        repeats=args.repeats,
        png_path=None if args.no_png else args.png,
    )


if __name__ == "__main__":
    main()
