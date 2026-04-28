from __future__ import annotations

from pathlib import Path
from typing import Final

HERE: Final[Path] = Path(__file__).resolve().parent
ASSETS_DIR: Final[Path] = HERE / "assets"
ML_MODELS_DIR: Final[Path] = ASSETS_DIR / "ml_models"


DATASETS: Final[tuple[tuple[str, str], ...]] = (
    ("wustl", "WUSTL"),
    ("nslkdd", "NSLKDD"),
    ("toniot", "TonIoT"),
    ("medsec", "MedSec"),
)
