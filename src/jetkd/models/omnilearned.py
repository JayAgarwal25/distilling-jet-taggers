"""OmniLearned-S (PET2, small) built from the omnilearned package, for operation counts and timing.

The setting of tagging-guide's pet2_s and its PET2Wrapper (github.com/heidelberg-hepml/tagging-guide).
Its trained weights are OmniLearned's own; the paper times this architecture with random weights.
"""

from __future__ import annotations

import torch
from torch import nn

from jetkd.data import JetBatch
from jetkd.features import STANDARDISATION
from jetkd.models.inputs import embed


class OmniLearnedS(nn.Module):
    def __init__(self):
        super().__init__()
        from omnilearned.network import PET2

        self.net = PET2(input_dim=7, num_classes=1, use_int=True, local_int=False, int_type="lhc", K=10,
                        num_coord=2, conditional=False, pid=False, add_info=False, mode="classifier",
                        base_dim=128, num_transformers=8, num_transformers_head=2, num_tokens=4,
                        num_heads=8, mlp_ratio=2)

    # tagging-guide's PET2Wrapper on our features; same logits (verified bit for bit)
    def forward(self, batch: JetBatch) -> torch.Tensor:
        _, feats, _ = embed(batch.p4, spurions=False, dtype=torch.float32)
        mean, scale = STANDARDISATION[0]
        feats[..., 0] = feats[..., 0] / scale + mean   # PET2 takes log pT as it is
        feats = feats[..., [5, 4, 0, 1, 2, 3, 6]]   # deta, dphi, log pT first, for its local features
        return self.net(x=feats, y=None)["y_pred"][:, 0]
