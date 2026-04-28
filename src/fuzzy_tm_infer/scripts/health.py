#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fuzzy_tm_infer._logging import add_logging_args, configure_from_args, logger
else:
    from .._logging import add_logging_args, configure_from_args, logger


REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local project health checks.")
    parser.add_argument("--native", action="store_true", help="also run native exactness")
    add_logging_args(parser)
    args = parser.parse_args()
    configure_from_args(args, default_level="INFO")

    _run([sys.executable, "-m", "compileall", "-q", "src/fuzzy_tm_infer"])
    _run([sys.executable, "-m", "fuzzy_tm_infer.scripts.result_validate"])
    _run_stale_name_check()
    if args.native:
        _run(
            [
                "make",
                "-C",
                "src/fuzzy_tm_infer",
                "clean-native",
                "native-v17",
                "native-v19",
                "verify-native",
            ]
        )
    logger.success("health check passed")


def _run(cmd: list[str]) -> None:
    logger.info("running {}", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _run_stale_name_check() -> None:
    pattern = "|".join(
        (
            "tm_" "local_infer",
            "tm-" "local-infer",
            "src/tm_" "local",
            "libtm_" "local",
            "100\\.98",
        )
    )
    cmd = [
        "rg",
        "-n",
        "--glob",
        "!ansible/rpi.env",
        pattern,
        "README.md",
        "SPEC.MD",
        "AGENTS.md",
        "ansible",
        "src/fuzzy_tm_infer",
        "pyproject.toml",
        ".gitignore",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if result.returncode == 0:
        raise SystemExit("stale names found:\n" + result.stdout)
    if result.returncode not in {0, 1}:
        raise SystemExit(result.stderr or "stale-name check failed")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
