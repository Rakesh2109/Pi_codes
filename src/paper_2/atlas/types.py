"""Serialisable record types for the GLADE+FPTM atlas export.

These mirror the dataclasses used by ``sonnets-project/TMAtlas`` so the JSON
shape stays familiar, with a few additions that the Fuzzy Pattern Tsetlin
Machine needs (per-clause polarity, the literal-sum ``clamp``).  Each record
exposes ``to_dict()`` returning JSON-ready primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Literal:
    """A single included literal: ``feature operator threshold``."""

    feature: str
    operator: str  # "≥" for an asserted bit, "<" for a negated bit
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "operator": self.operator,
            "threshold": float(self.threshold),
        }


@dataclass(frozen=True, slots=True)
class ClauseWeight:
    """Vote contribution of a clause towards one class."""

    value: float
    polarity: str  # "positive" | "negative"

    def to_dict(self) -> dict[str, Any]:
        return {"value": float(self.value), "polarity": self.polarity}


@dataclass(frozen=True, slots=True)
class Clause:
    """One FPTM clause (a clause belongs to exactly one class and polarity).

    ``literals`` lists only the *included* literals.  ``clamp`` is the
    fuzzy-pattern literal-sum cap: the clause emits ``max(clamp - v, 0)`` for a
    sample with ``v`` violated literals, so the clause vote lies in ``[0, clamp]``.
    ``weights`` is kept TMAtlas-shaped (``{class_name: ClauseWeight}``) and holds
    only the owning class.
    """

    id: int
    cls: str
    polarity: str  # "positive" | "negative"
    clamp: int
    literals: list[Literal]
    weights: dict[str, ClauseWeight]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": int(self.id),
            "class": self.cls,
            "polarity": self.polarity,
            "clamp": int(self.clamp),
            "nLiterals": len(self.literals),
            "literals": [lit.to_dict() for lit in self.literals],
            "weights": {k: w.to_dict() for k, w in self.weights.items()},
        }


@dataclass(frozen=True, slots=True)
class FeatureClass:
    """Definition of one original (pre-binarisation) input feature."""

    name: str
    type: str  # "binary" | "continuous"
    range: tuple[float, float]
    thresholds: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "range": [float(self.range[0]), float(self.range[1])],
            "thresholds": [float(t) for t in self.thresholds],
        }


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """High-level description of the model."""

    type: str  # "classification"
    task: str  # "binary" | "multiclass"
    classes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "task": self.task, "classes": list(self.classes)}


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Training hyper-parameters and shape of the FPTM."""

    num_clauses_per_class: int
    num_classes: int
    num_literals: int
    T: float | None = None
    s: float | None = None
    L: int | None = None
    LF: int | None = None
    epochs: int | None = None
    weighted_clauses: bool = False
    created: str | None = None
    binariser: str = "GLADE"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "numClausesPerClass": int(self.num_clauses_per_class),
            "numClauses": int(self.num_clauses_per_class) * int(self.num_classes),
            "numClasses": int(self.num_classes),
            "numLiterals": int(self.num_literals),
            "T": self.T,
            "s": self.s,
            "weightedClauses": bool(self.weighted_clauses),
            "binariser": self.binariser,
            "created": self.created,
        }
        if self.L is not None:
            d["L"] = int(self.L)
        if self.LF is not None:
            d["LF"] = int(self.LF)
        if self.epochs is not None:
            d["epochs"] = int(self.epochs)
        return d
