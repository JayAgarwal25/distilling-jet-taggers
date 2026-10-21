"""Teacher logits: one logit per jet and split, in the dataset's row order.

Distillation reads a teacher only through these files, so any model can be a teacher:
a jetkd model (`model_logits`) or another project's predictions (`from_probabilities`).
Before a file is written, `check_rows` makes sure its rows are the dataset's rows.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from jetkd.data import TopTaggingSplit, collate
from jetkd.log import duration, log

# ensemble teachers: the mean logit of several runs of one model; run ids give them seed 1001
ENSEMBLES = {"lgatr_slim_ens": ("lgatr_slim", (1001, 1002, 1003))}


def logits_path(directory: str | Path, teacher: str, seed: int, split: str) -> Path:
    return Path(directory) / f"{teacher}_s{seed}_{split}.npz"


def ensemble_logits(directory: str | Path, name: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Logits and labels of ensemble `name` from its members' logits in `directory`."""
    member, seeds = ENSEMBLES[name]
    zs, labels = [], None
    for s in seeds:
        with np.load(logits_path(directory, member, s, split)) as f:
            if labels is not None and not np.array_equal(f["labels"], labels):
                raise ValueError(f"{member} run {s}, {split}: labels differ between members")
            zs.append(f["logits"].astype(np.float64))
            labels = f["labels"]
    return np.mean(zs, axis=0).astype(np.float32), labels


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


@contextmanager
def deterministic():
    """Deterministic GPU kernels: the sums over particles otherwise use atomics, so two
    evaluations of one checkpoint differ in the last bits and rej@0.3 in its last digits."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(before)


@torch.no_grad()
def model_logits(model: torch.nn.Module, split: str, data: str | Path | Dataset,
                 device: str = "cuda", batch_size: int = 512, workers: int = 4,
                 progress: bool = False):
    """Logits and labels of `model` on a whole split, in row order.

    `data` is the dataset file or an already loaded `TopTaggingSplit` of `split`. With
    `progress`, a line is printed after every tenth of the batches.
    """
    model.eval().to(device)
    dataset = data if isinstance(data, Dataset) else TopTaggingSplit(split, data)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, collate_fn=collate)
    if progress:
        log(f"{split}: {len(dataset):,} jets in {len(loader):,} batches of {batch_size}")
    z, y = [], []
    t0, every = time.time(), max(1, len(loader) // 10)
    with deterministic():
        for i, batch in enumerate(loader, 1):
            z.append(model(batch.to(device)).float().cpu())
            y.append(batch.label)
            if progress and (i % every == 0 or i == len(loader)):
                spent = time.time() - t0
                log(f"{split}: batch {i:,}/{len(loader):,} after {duration(spent)}, "
                    f"time estimate {duration(spent / i * len(loader))}")
    return torch.cat(z).numpy(), torch.cat(y).numpy()
