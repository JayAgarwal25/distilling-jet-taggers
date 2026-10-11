"""The top tagging dataset and the batch format every model takes.

The data are the top tagging reference sample of Kasieczka et al. (Zenodo 2603256) in
the single-file layout used by L-GATr and LLoCa. For each split, `kinematics_<split>`
holds up to 200 constituents per jet as (E, px, py, pz), zero padded, and
`labels_<split>` is 1 for top jets. `scripts/get_data.py` downloads the file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

SPLITS = ("train", "val", "test")

# a slot holds a particle when all four |p_mu| exceed this, as in LLoCa
PARTICLE_EPS = 1e-5


@dataclass(frozen=True)
class JetBatch:
    p4: torch.Tensor     # (B, 200, 4) as (E, px, py, pz), zero padded
    mask: torch.Tensor   # (B, 200), True for particles
    label: torch.Tensor  # (B,), 1 for top
    sid: torch.Tensor    # (B,), row index in the split

    def to(self, device) -> JetBatch:
        return JetBatch(self.p4.to(device), self.mask.to(device), self.label.to(device),
                        self.sid.to(device))


def particle_mask(p4: torch.Tensor) -> torch.Tensor:
    return (p4.abs() > PARTICLE_EPS).all(dim=-1)


class TopTaggingSplit(Dataset):
    """One split, held in memory.

    Items are (p4, label, row index). The row index identifies a jet in every
    prediction file and teacher logit file. `limit` keeps the first rows only.
    """

    def __init__(self, split: str, path: str = "data/toptagging_full.npz",
                 limit: int | None = None):
        with np.load(path) as f:
            self.p4 = torch.from_numpy(f[f"kinematics_{split}"][:limit].astype(np.float32))
            self.label = torch.from_numpy(f[f"labels_{split}"][:limit].astype(np.float32))

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, i: int):
        return self.p4[i], self.label[i], i


def collate(items) -> JetBatch:
    p4 = torch.stack([it[0] for it in items])
    label = torch.stack([it[1] for it in items])
    sid = torch.tensor([it[2] for it in items])
    return JetBatch(p4, particle_mask(p4), label, sid)
