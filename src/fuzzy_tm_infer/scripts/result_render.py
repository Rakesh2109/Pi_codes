#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fuzzy_tm_infer._logging import add_logging_args, configure_from_args, logger
else:
    from .._logging import add_logging_args, configure_from_args, logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render result CSV tables to LaTeX booktabs tables."
    )
    parser.add_argument("result_dir", type=Path, help="structured result directory")
    parser.add_argument(
        "--table",
        action="append",
        default=[],
        help="table ID to render; defaults to all tables/*.csv",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_from_args(args, default_level="INFO")

    result_dir = args.result_dir
    tables_dir = result_dir / "tables"
    latex_dir = result_dir / "latex"
    if not tables_dir.is_dir():
        raise SystemExit(f"missing tables directory: {tables_dir}")
    latex_dir.mkdir(exist_ok=True)

    csv_paths = (
        [tables_dir / f"{table_id}.csv" for table_id in args.table]
        if args.table
        else sorted(tables_dir.glob("*.csv"))
    )
    if not csv_paths:
        raise SystemExit(f"no CSV tables found in {tables_dir}")

    for csv_path in csv_paths:
        if not csv_path.exists():
            raise SystemExit(f"missing CSV table: {csv_path}")
        latex_path = latex_dir / f"{csv_path.stem}.tex"
        latex_path.write_text(_csv_to_latex(csv_path), encoding="utf-8")
        logger.info("rendered {}", latex_path)


def _csv_to_latex(path: Path) -> str:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise SystemExit(f"empty CSV table: {path}")

    n_cols = len(rows[0])
    if n_cols == 0:
        raise SystemExit(f"CSV table has no columns: {path}")
    for row in rows:
        if len(row) != n_cols:
            raise SystemExit(f"ragged CSV table: {path}")

    align = "l" + "r" * (n_cols - 1)
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule"]
    lines.append(" & ".join(_latex_cell(cell) for cell in rows[0]) + r" \\")
    lines.append("\\midrule")
    for row in rows[1:]:
        lines.append(" & ".join(_latex_cell(cell) for cell in row) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _latex_cell(value: str) -> str:
    text = value.strip()
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


if __name__ == "__main__":
    main()
