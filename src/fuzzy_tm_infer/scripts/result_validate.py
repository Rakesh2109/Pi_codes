#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
import tomllib
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fuzzy_tm_infer._logging import add_logging_args, configure_from_args, logger
else:
    from .._logging import add_logging_args, configure_from_args, logger


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"
ALLOWED_ROOT_FILES = {"README.md", "SPEC.md", "INDEX.md"}
ALLOWED_ROOT_DIRS = {"formats", "templates"}
FORMAT_IDS = {"benchmark.table.v1", "analysis.note.v1"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate structured result records.")
    parser.add_argument("paths", nargs="*", type=Path, help="result dirs; defaults to results/")
    add_logging_args(parser)
    args = parser.parse_args()
    configure_from_args(args, default_level="INFO")

    errors: list[str] = []
    paths = args.paths or [RESULTS_DIR]
    for path in paths:
        if path == RESULTS_DIR:
            _validate_root(path, errors)
            for child in sorted(path.iterdir()):
                if child.is_dir() and child.name not in ALLOWED_ROOT_DIRS:
                    _validate_result_dir(child, errors)
        else:
            _validate_result_dir(path, errors)

    if errors:
        for error in errors:
            logger.error(error)
        raise SystemExit(1)
    logger.success("results validation passed")


def _validate_root(path: Path, errors: list[str]) -> None:
    if not path.is_dir():
        errors.append(f"missing results directory: {path}")
        return
    for child in sorted(path.iterdir()):
        if child.is_file() and child.name not in ALLOWED_ROOT_FILES:
            errors.append(f"flat file not allowed in results root: {child}")
        if child.is_dir() and child.name in ALLOWED_ROOT_DIRS:
            continue


def _validate_result_dir(path: Path, errors: list[str]) -> None:
    if not path.is_dir():
        errors.append(f"result path is not a directory: {path}")
        return
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}_[a-z0-9]+(?:_[a-z0-9]+)*", path.name):
        errors.append(f"bad result directory name: {path}")

    manifest_path = path / "manifest.toml"
    readme_path = path / "README.md"
    raw_dir = path / "raw"
    for required in (manifest_path, readme_path, raw_dir):
        if not required.exists():
            errors.append(f"missing required result artifact: {required}")
    if not manifest_path.exists():
        return

    try:
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        errors.append(f"invalid TOML in {manifest_path}: {exc}")
        return

    _validate_manifest(path, manifest, errors)


def _validate_manifest(path: Path, manifest: dict[str, object], errors: list[str]) -> None:
    for key in (
        "schema_version",
        "format_id",
        "date",
        "title",
        "command_id",
        "command",
        "platform",
        "status",
        "exact",
        "raw_dir",
    ):
        if key not in manifest:
            errors.append(f"{path}: manifest missing {key}")

    format_id = manifest.get("format_id")
    if format_id not in FORMAT_IDS:
        errors.append(f"{path}: unsupported format_id {format_id!r}")

    status = manifest.get("status")
    if status not in {"draft", "accepted", "rejected", "archived"}:
        errors.append(f"{path}: invalid status {status!r}")

    if format_id == "benchmark.table.v1":
        _validate_benchmark(path, manifest, errors)


def _validate_benchmark(path: Path, manifest: dict[str, object], errors: list[str]) -> None:
    tables = manifest.get("tables")
    if not isinstance(tables, list) or not tables:
        errors.append(f"{path}: benchmark result must declare non-empty tables list")
        return
    if not (path / "tables").is_dir():
        errors.append(f"{path}: missing tables/")
    if not (path / "latex").is_dir():
        errors.append(f"{path}: missing latex/")

    for table in tables:
        if not isinstance(table, str) or not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", table):
            errors.append(f"{path}: invalid table id {table!r}")
            continue
        csv_path = path / "tables" / f"{table}.csv"
        tex_path = path / "latex" / f"{table}.tex"
        if not csv_path.exists():
            errors.append(f"{path}: missing {csv_path.relative_to(path)}")
        else:
            _validate_csv(csv_path, errors)
        if not tex_path.exists():
            errors.append(f"{path}: missing {tex_path.relative_to(path)}")
        else:
            text = tex_path.read_text(encoding="utf-8")
            for token in ("\\toprule", "\\midrule", "\\bottomrule"):
                if token not in text:
                    errors.append(f"{tex_path}: missing {token}")


def _validate_csv(path: Path, errors: list[str]) -> None:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        errors.append(f"{path}: empty CSV")
        return
    width = len(rows[0])
    if width == 0:
        errors.append(f"{path}: CSV has no columns")
    for row in rows:
        if len(row) != width:
            errors.append(f"{path}: ragged CSV row")
            return


if __name__ == "__main__":
    main()
