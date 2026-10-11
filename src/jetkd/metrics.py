"""AUC, BCE, accuracy and background rejection from logits."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def rejection(fpr: np.ndarray, tpr: np.ndarray, eps_s: float) -> float:
    """1/fpr at signal efficiency eps_s, interpolating the ROC curve linearly in tpr."""
    order = np.lexsort((fpr, tpr))
    tpr, fpr = tpr[order], fpr[order]
    keep = np.r_[True, tpr[1:] != tpr[:-1]]   # smallest fpr at each tpr
    fpr_at = float(np.interp(eps_s, tpr[keep], fpr[keep]))
    return 1.0 / fpr_at if fpr_at > 0 else float("inf")


def binary_metrics(logits, labels, signal_efficiencies=(0.3, 0.5, 0.8)) -> dict:
    z = np.asarray(logits, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    fpr, tpr, _ = roc_curve(y, z)
    out = {
        "n": int(z.size),
        "loss": float(np.mean(np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z))))),
        "accuracy": float(accuracy_score(y, (z >= 0.0).astype(np.float64))),
        "auc": float(roc_auc_score(y, z)),
    }
    for eps in signal_efficiencies:
        out[f"rej{round(eps * 100):02d}"] = rejection(fpr, tpr, eps)
    return out
