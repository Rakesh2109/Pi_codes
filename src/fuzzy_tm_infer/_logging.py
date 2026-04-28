from __future__ import annotations

import os
import sys
from argparse import ArgumentParser, Namespace

from loguru import logger

_FORMAT = "<level>{message}</level>"
_DEFAULT_LEVEL = os.environ.get("FUZZY_TM_LOG_LEVEL", "WARNING").upper()

logger.remove()
logger.add(sys.stderr, level=_DEFAULT_LEVEL, format=_FORMAT)


def add_logging_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase log verbosity; use twice for trace logs",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only show errors",
    )


def configure_logging(
    *,
    verbose: int = 0,
    quiet: bool = False,
    default_level: str = "WARNING",
) -> None:
    level = _selected_level(verbose=verbose, quiet=quiet, default_level=default_level)
    logger.remove()
    logger.add(sys.stderr, level=level, format=_FORMAT)


def configure_from_args(args: Namespace, *, default_level: str = "WARNING") -> None:
    configure_logging(
        verbose=int(getattr(args, "verbose", 0) or 0),
        quiet=bool(getattr(args, "quiet", False)),
        default_level=default_level,
    )


def _selected_level(*, verbose: int, quiet: bool, default_level: str) -> str:
    if quiet:
        return "ERROR"

    env_level = os.environ.get("FUZZY_TM_LOG_LEVEL")
    if env_level:
        return env_level.upper()

    if verbose >= 2:
        return "TRACE"
    if verbose == 1:
        return "DEBUG"
    return default_level.upper()
