#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fuzzy_tm_infer._logging import add_logging_args, configure_from_args, logger
else:
    from .._logging import add_logging_args, configure_from_args, logger


ROOT = Path(__file__).resolve().parents[1]
FUZZY_TM_DIR = ROOT / "algorithms" / "c" / "fuzzy_tm"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a new native Fuzzy TM backend version from an existing one."
    )
    parser.add_argument("--version", required=True, help="new version id, for example v20")
    parser.add_argument("--base", default="v17", help="base version to copy, default v17")
    add_logging_args(parser)
    args = parser.parse_args()
    configure_from_args(args, default_level="INFO")

    version = _validate_version(args.version)
    base = _validate_version(args.base)
    source = FUZZY_TM_DIR / base
    target = FUZZY_TM_DIR / version

    if not source.is_dir():
        raise SystemExit(f"base backend does not exist: {source}")
    if target.exists():
        raise SystemExit(f"target backend already exists: {target}")

    shutil.copytree(source, target)
    _rewrite_text_files(target, base, version)

    logger.success(f"created {target.relative_to(ROOT)} from {source.relative_to(ROOT)}")
    logger.info("next steps:")
    logger.info(f"  1. edit algorithms/c/fuzzy_tm/{version}/README.md")
    logger.info(f"  2. implement the hypothesis in algorithms/c/fuzzy_tm/{version}/")
    logger.info(f"  3. run: make -C src/fuzzy_tm_infer native-{version}")
    logger.info("  4. run: make -C src/fuzzy_tm_infer verify-native")


def _validate_version(value: str) -> str:
    if not re.fullmatch(r"v[0-9]+", value):
        raise SystemExit(f"version must match vNN, got {value!r}")
    return value


def _rewrite_text_files(root: Path, base: str, version: str) -> None:
    replacements = {
        base: version,
        base.upper(): version.upper(),
        f"tm_{base}": f"tm_{version}",
    }

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        updated = text
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
