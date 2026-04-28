from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "FuzzyTMModel",
    "GLADEBooleanizer",
    "GLADEv2",
    "KBinsBooleanizer",
    "StandardBinarizer",
    "ThermometerBinarizer",
    "TMModel",
]

if TYPE_CHECKING:
    from .algorithms import TMModel
    from .algorithms.c import FuzzyTMModel
    from .booleanizers import (
        GLADEBooleanizer,
        GLADEv2,
        KBinsBooleanizer,
        StandardBinarizer,
        ThermometerBinarizer,
    )


def __getattr__(name: str) -> object:
    if name == "FuzzyTMModel":
        from .algorithms.c import FuzzyTMModel

        return FuzzyTMModel
    if name == "TMModel":
        from .algorithms import TMModel

        return TMModel
    if name in {
        "GLADEBooleanizer",
        "GLADEv2",
        "KBinsBooleanizer",
        "StandardBinarizer",
        "ThermometerBinarizer",
    }:
        from .booleanizers import (
            GLADEBooleanizer,
            GLADEv2,
            KBinsBooleanizer,
            StandardBinarizer,
            ThermometerBinarizer,
        )

        return {
            "GLADEBooleanizer": GLADEBooleanizer,
            "GLADEv2": GLADEv2,
            "KBinsBooleanizer": KBinsBooleanizer,
            "StandardBinarizer": StandardBinarizer,
            "ThermometerBinarizer": ThermometerBinarizer,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
