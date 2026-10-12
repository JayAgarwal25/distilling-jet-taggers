"""L-GATr and L-GATr-slim taggers, read out from a global token.

Particles and spurions enter as four-vectors with the seven particle features as
scalars. L-GATr embeds the vectors in multivectors; L-GATr-slim keeps them as vectors.
This writes out the path of tagging-guide's LGATrWrapper and LGATrSlimWrapper that our
models take. Verified to reproduce their initial weights, logits and our trained
checkpoints bit for bit.
"""

from __future__ import annotations

import torch
from lgatr import LGATr, embed_vector, extract_scalar
from lgatr.nets.slim import LGATrSlim
from torch import nn

from jetkd.data import JetBatch
from jetkd.models.inputs import add_global_token, attention_mask, embed, pack


def tokens(p4: torch.Tensor, dtype: torch.dtype):
    """Vectors (tokens, 1, 4), scalars (tokens, 8), jet index and global token positions."""
    p4, feats, mask = embed(p4, spurions=True, dtype=dtype)
    [v, s], jet, ptr = pack(mask, p4.unsqueeze(2).float(), feats)
    return add_global_token(v, s, jet, ptr)


class LGATrSlimTagger(nn.Module):
    """`float64` computes the features from float64 momenta, as for the L-GATr-slim teacher."""

    def __init__(self, hidden_v_channels: int, hidden_s_channels: int, num_heads: int,
                 num_blocks: int, mlp_ratio: int, float64: bool = False):
        super().__init__()
        self.dtype = torch.float64 if float64 else torch.float32
        self.net = LGATrSlim(num_blocks=num_blocks, in_v_channels=1, out_v_channels=0,
                             hidden_v_channels=hidden_v_channels, in_s_channels=8,
                             out_s_channels=1, hidden_s_channels=hidden_s_channels,
                             num_heads=num_heads, mlp_ratio=mlp_ratio, attn_ratio=1)

    def forward(self, batch: JetBatch) -> torch.Tensor:
        v, s, jet, first = tokens(batch.p4, self.dtype)
        _, out = self.net(v.unsqueeze(0), s.unsqueeze(0), **attention_mask(jet, s.dtype))
        return out.squeeze(0)[first, 0]


class LGATrTagger(nn.Module):
    """The L-GATr teacher. Its features are computed from float64 momenta."""

    def __init__(self):
        super().__init__()
        self.net = LGATr(
            num_blocks=12, in_mv_channels=1, out_mv_channels=1, hidden_mv_channels=16,
            in_s_channels=8, out_s_channels=None, hidden_s_channels=32,
            attention=dict(num_heads=8, multi_query=False, attn_ratio=2, head_scale=True),
            mlp=dict(nonlinearity="gelu", mlp_ratio=2),
            primitives=dict(sparse_gp=True, sparse_linear=False, bivector=True,
                            geometric_product=True, subgroup=True),
        )

    def forward(self, batch: JetBatch) -> torch.Tensor:
        v, s, jet, first = tokens(batch.p4, torch.float64)
        mv, _ = self.net(embed_vector(v).unsqueeze(0), s.unsqueeze(0),
                         **attention_mask(jet, s.dtype))
        return extract_scalar(mv).squeeze(0)[first, 0, 0]
