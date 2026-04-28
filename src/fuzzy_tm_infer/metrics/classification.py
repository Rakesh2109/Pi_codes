from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def macro_f1(
    y_true: NDArray[np.int32],
    y_pred: NDArray[np.int32],
    n_classes: int,
) -> float:
    f1s: list[float] = []
    for k in range(n_classes):
        tp = int(((y_pred == k) & (y_true == k)).sum())
        fp = int(((y_pred == k) & (y_true != k)).sum())
        fn = int(((y_pred != k) & (y_true == k)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return float(np.mean(f1s))
