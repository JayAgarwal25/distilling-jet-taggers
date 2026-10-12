"""Teacher logits: one logit per jet and split, in the dataset's row order.

Distillation reads a teacher only through these files, so any model can be a teacher:
a jetkd model (`model_logits`) or another project's predictions (`from_probabilities`).
Before a file is written, `check_rows` makes sure its rows are the dataset's rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from jetkd.data import TopTaggingSplit, collate


def logits_path(directory: str | Path, teacher: str, seed: int, split: str) -> Path:
    return Path(directory) / f"{teacher}_s{seed}_{split}.npz"


def save_logits(path: str | Path, logits: np.ndarray, labels: np.ndarray) -> None:
    logits = np.asarray(logits, dtype=np.float32)
    if not np.isfinite(logits).all():
        raise ValueError("non-finite logits")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, logits=logits, sids=np.arange(len(logits)),
                        labels=np.asarray(labels, dtype=np.float32))


def from_probabilities(p_signal: np.ndarray, p_background: np.ndarray) -> np.ndarray:
    """Logits from the two stored class probabilities.

    Taking both columns, not 1 - p, keeps the digits of saturated scores.
    """
    p_signal = np.asarray(p_signal).astype(np.float64)
    p_background = np.asarray(p_background).astype(np.float64)
    if (p_signal <= 0).any() or (p_background <= 0).any():
        raise ValueError("a stored probability is zero; the logit cannot be recovered")
    return (np.log(p_signal) - np.log(p_background)).astype(np.float32)


def check_rows(labels: np.ndarray, split: str, data: str | Path,
               jet_pt: np.ndarray | None = None, pt_outliers: int = 500) -> None:
    """Refuse a foreign file whose rows are not the dataset's rows, in order.

    Labels must agree everywhere. With `jet_pt`, the jet pT must also agree to 1e-4
    except for at most `pt_outliers` jets: pipelines that keep fewer constituents
    (128 for weaver's ParT) change the pT of the largest jets slightly.
    """
    with np.load(data) as f:
        ours = f[f"labels_{split}"]
        k = f[f"kinematics_{split}"] if jet_pt is not None else None
    if len(labels) != len(ours) or not np.array_equal(np.asarray(labels).astype(ours.dtype), ours):
        raise ValueError(f"{split}: labels differ from the dataset; rows are not aligned")
    if jet_pt is not None:
        pt = np.hypot(k[:, :, 1].sum(1), k[:, :, 2].sum(1))
        n = int((np.abs(pt - jet_pt) > 1e-4 * np.maximum(jet_pt, 1e-9)).sum())
        if n > pt_outliers:
            raise ValueError(f"{split}: {n} jets differ in pT; rows are not aligned")


@torch.no_grad()
def model_logits(model: torch.nn.Module, split: str, data: str | Path, device: str = "cuda",
                 batch_size: int = 512, workers: int = 4):
    """Logits and labels of `model` on a whole split, in row order."""
    model.eval().to(device)
    loader = DataLoader(TopTaggingSplit(split, data), batch_size=batch_size,
                        num_workers=workers, collate_fn=collate)
    z, y = [], []
    for batch in loader:
        z.append(model(batch.to(device)).float().cpu())
        y.append(batch.label)
    return torch.cat(z).numpy(), torch.cat(y).numpy()
