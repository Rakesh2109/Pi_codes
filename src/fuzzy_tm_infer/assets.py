from __future__ import annotations

import os
from pathlib import Path
from zipfile import ZipFile

from ._logging import logger
from .config import ASSETS_DIR, HERE, ML_MODELS_DIR

_ML_DATASETS = ("wustl", "nslkdd", "toniot", "medsec")
_ML_FILES = ("DecisionTree.pkl", "testset.npz")


def ensure_assets() -> None:
    """Extract root-level archives into this folder if assets are not present."""
    model_marker = ASSETS_DIR / "tm_models" / "wustl_model.fbz"
    data_marker = ASSETS_DIR / "datasets" / "wustl_test" / "WUSTL_X_test_raw.bin"
    if model_marker.exists() and data_marker.exists():
        return

    archives = (_archive("datasets.zip"), _archive("tm_models.zip"))
    missing = [str(path) for path in archives if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing extracted assets and source archive(s): " + ", ".join(missing)
        )

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        logger.info("extracting assets from {}", archive)
        with ZipFile(archive) as zf:
            zf.extractall(ASSETS_DIR)


def ensure_ml_models() -> None:
    """Extract Decision Tree models and matching test sets on demand."""
    markers = [
        ML_MODELS_DIR / stem / filename
        for stem in _ML_DATASETS
        for filename in _ML_FILES
    ]
    if all(path.exists() for path in markers):
        return

    archive = _ml_models_archive()
    if archive is None:
        raise FileNotFoundError(
            "Missing ML model assets. Expected ml_models.zip in src/ or repo root."
        )

    ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("extracting Decision Tree assets from {}", archive)
    with ZipFile(archive) as zf:
        names = set(zf.namelist())
        for stem in _ML_DATASETS:
            (ML_MODELS_DIR / stem).mkdir(parents=True, exist_ok=True)
            for filename in _ML_FILES:
                member = _zip_member(names, f"{stem}/{filename}")
                if member not in names:
                    raise FileNotFoundError(f"{archive} does not contain {member}")
                target = ML_MODELS_DIR / stem / filename
                if target.exists():
                    continue
                target.write_bytes(zf.read(member))


def _ml_models_archive() -> Path | None:
    return _archive("ml_models.zip", required=False)


def _archive(name: str, *, required: bool = True) -> Path | None:
    env_root = os.environ.get("FUZZY_TM_INFER_ARCHIVE_DIR")
    roots = [
        Path(env_root) if env_root else None,
        HERE.parents[0],
        HERE.parents[1],
        HERE.parents[1] / "archives" / "source_assets",
        HERE.parents[1] / "archives",
        Path.cwd(),
        Path.cwd() / "src",
        Path.cwd() / "archives" / "source_assets",
        Path.cwd() / "archives",
    ]
    for root in roots:
        if root is None:
            continue
        path = root / name
        if path.exists():
            return path
    if required:
        return Path.cwd() / name
    return None


def _zip_member(names: set[str], suffix: str) -> str:
    if suffix in names:
        return suffix
    prefixed = f"ml_models/{suffix}"
    if prefixed in names:
        return prefixed
    return suffix
