#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path
from textwrap import dedent

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fuzzy_tm_infer._logging import add_logging_args, configure_from_args, logger
else:
    from .._logging import add_logging_args, configure_from_args, logger


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a structured result directory with CSV and LaTeX artifacts."
    )
    parser.add_argument("--slug", required=True, help="lowercase result slug")
    parser.add_argument("--title", required=True, help="human-readable result title")
    parser.add_argument(
        "--format-id",
        default="benchmark.table.v1",
        help="result format ID from results/formats",
    )
    parser.add_argument(
        "--experiment-type",
        default="generic",
        help="benchmark experiment type, for example full_comparison",
    )
    parser.add_argument("--table-id", default="metrics", help="primary table ID")
    parser.add_argument("--command-id", default="", help="ID from EXPERIMENT_COMMANDS.md")
    parser.add_argument("--command", default="", help="exact command used for the run")
    parser.add_argument("--platform", default="", help="platform summary")
    parser.add_argument("--date", default=date.today().isoformat(), help="YYYY-MM-DD")
    add_logging_args(parser)
    args = parser.parse_args()
    configure_from_args(args, default_level="INFO")

    slug = _validate_slug(args.slug)
    run_date = _validate_date(args.date)
    target = RESULTS_DIR / f"{run_date}_{slug}"
    if target.exists():
        raise SystemExit(f"result directory already exists: {target}")

    table_id = _validate_table_id(args.table_id)
    format_id = _validate_format_id(args.format_id)
    experiment_type = _validate_table_id(args.experiment_type)
    is_benchmark = format_id == "benchmark.table.v1"
    (target / "raw").mkdir(parents=True)
    if is_benchmark:
        (target / "tables").mkdir()
        (target / "latex").mkdir()
    _write_manifest(
        target,
        args.title,
        format_id,
        experiment_type if is_benchmark else "",
        table_id if is_benchmark else "",
        args.command_id,
        args.command,
        args.platform,
        run_date,
    )
    _write_readme(
        target,
        args.title,
        format_id,
        experiment_type,
        table_id,
        args.command_id,
        args.command,
        args.platform,
    )
    if is_benchmark:
        _write_metrics(target, table_id)
        _write_table(target, table_id)

    logger.success("created {}", target.relative_to(REPO_ROOT))
    if is_benchmark:
        logger.info("fill tables/{}.csv and keep latex/{}.tex LaTeX-compatible", table_id, table_id)
    else:
        logger.info("write the analysis note in README.md and place supporting material under raw/")


def _validate_slug(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", value):
        raise SystemExit(
            "slug must be lowercase ASCII words separated by underscores, "
            f"got {value!r}"
        )
    return value


def _validate_date(value: str) -> str:
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"date must be YYYY-MM-DD, got {value!r}") from exc
    return value


def _validate_format_id(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9]+(?:\.[a-z0-9]+)*\.v[0-9]+", value):
        raise SystemExit(f"format ID must look like benchmark.table.v1, got {value!r}")
    if value not in {"benchmark.table.v1", "analysis.note.v1"}:
        raise SystemExit(f"unsupported format ID {value!r}; use benchmark.table.v1 or analysis.note.v1")
    return value


def _validate_table_id(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", value):
        raise SystemExit(f"table ID must be lowercase words separated by underscores, got {value!r}")
    return value


def _write_manifest(
    target: Path,
    title: str,
    format_id: str,
    experiment_type: str,
    table_id: str,
    command_id: str,
    command: str,
    platform: str,
    run_date: str,
) -> None:
    text = dedent(
        f"""\
        schema_version = "1"
        format_id = "{_toml_string(format_id)}"
        date = "{run_date}"
        title = "{_toml_string(title)}"
        command_id = "{_toml_string(command_id)}"
        command = "{_toml_string(command)}"
        platform = "{_toml_string(platform)}"
        status = "draft"
        exact = false
        raw_dir = "raw"
        """
    )
    if experiment_type:
        text += f'experiment_type = "{_toml_string(experiment_type)}"\n'
    if table_id:
        text += (
            f'tables = ["{_toml_string(table_id)}"]\n'
            f'primary_csv = "tables/{_toml_string(table_id)}.csv"\n'
            f'primary_latex = "latex/{_toml_string(table_id)}.tex"\n'
        )
    (target / "manifest.toml").write_text(text, encoding="utf-8")


def _write_readme(
    target: Path,
    title: str,
    format_id: str,
    experiment_type: str,
    table_id: str,
    command_id: str,
    command: str,
    platform: str,
) -> None:
    text = dedent(
        f"""\
        # {title}

        ## Summary

        Short result summary.

        ## Hypothesis

        What was expected to happen?

        ## Format

        ```text
        {format_id}
        ```

        ## Command

        Command ID:

        ```text
        {command_id}
        ```

        Command:

        ```bash
        {command}
        ```

        ## Platform

        ```text
        {platform}
        ```

        ## Correctness

        - Exact:
        - Validation command:

        ## Artifacts

        See:

        ```text
        raw/
        ```

        ## Decision

        - accepted / rejected / archived

        Why:
        """
    )
    if experiment_type:
        text = text.replace(
            "## Command",
            f"Experiment type:\n\n```text\n{experiment_type}\n```\n\n## Command",
        )
    if table_id:
        text = text.replace(
            "raw/\n",
            f"tables/{table_id}.csv\nlatex/{table_id}.tex\nraw/\n",
        )
    (target / "README.md").write_text(text, encoding="utf-8")


def _write_metrics(target: Path, table_id: str) -> None:
    (target / "tables" / f"{table_id}.csv").write_text(
        "dataset,baseline,candidate,speedup,exact\n"
        "WUSTL,,,,false\n"
        "NSLKDD,,,,false\n"
        "TonIoT,,,,false\n"
        "MedSec,,,,false\n",
        encoding="utf-8",
    )


def _write_table(target: Path, table_id: str) -> None:
    (target / "latex" / f"{table_id}.tex").write_text(
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        "Dataset & Baseline & Candidate & Speedup & Exact \\\\\n"
        "\\midrule\n"
        "WUSTL &  &  &  &  \\\\\n"
        "NSLKDD &  &  &  &  \\\\\n"
        "TonIoT &  &  &  &  \\\\\n"
        "MedSec &  &  &  &  \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n",
        encoding="utf-8",
    )


def _toml_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


if __name__ == "__main__":
    main()
