"""Numerically stable regression metrics."""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


def regression_metrics(predictions: ArrayLike, targets: ArrayLike) -> dict[str, float]:
    pred, true = np.asarray(predictions, dtype=float).reshape(-1), np.asarray(targets, dtype=float).reshape(-1)
    if pred.shape != true.shape or not pred.size: raise ValueError("predictions and targets must be non-empty and equal-sized")
    residual = pred - true
    return {"MAE": float(np.mean(np.abs(residual))), "RMSE": float(np.sqrt(np.mean(residual ** 2))),
            "R2": float(1 - np.sum(residual ** 2) / max(np.sum((true - true.mean()) ** 2), 1e-12))}
