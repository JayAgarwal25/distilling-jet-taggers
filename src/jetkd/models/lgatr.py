"""L-GATr and L-GATr-slim taggers, read out from a global token.

Particles and spurions enter as four-vectors with the seven particle features as
scalars. L-GATr embeds the vectors in multivectors; L-GATr-slim keeps them as vectors.
"""

from __future__ import annotations

import torch
from lgatr import LGATr, embed_vector, extract_scalar
from lgatr.nets.slim import LGATrSlim
from torch import nn

from jetkd.data import JetBatch
from jetkd.models.inputs import add_global_token, attention_mask, embed, pack


# tagging-guide's LGATrWrapper and LGATrSlimWrapper (identity frames, global token),
# written out; same initial weights, logits and checkpoints (verified bit for bit)
def tokens(p4: torch.Tensor, dtype: torch.dtype):
    """Vectors (tokens, 1, 4), scalars (tokens, 8), jet index and global token positions."""
    p4, feats, mask = embed(p4, spurions=True, dtype=dtype)
    [v, s], jet, ptr = pack(mask, p4.unsqueeze(2).float(), feats)
    return add_global_token(v, s, jet, ptr)


def padded_tokens(p4: torch.Tensor, dtype: torch.dtype):
    """As `tokens`, padded: vectors (jets, tokens, 1, 4), scalars (jets, tokens, 8), attention mask.
    The global token comes first."""
    p4, feats, mask = embed(p4, spurions=True, dtype=dtype)
    v = torch.cat([torch.zeros_like(p4[:, :1]), p4], dim=1).unsqueeze(2).float()
    s = feats.new_zeros(feats.shape[0], feats.shape[1] + 1, feats.shape[2] + 1)
    s[:, 1:, :-1] = feats
    s[:, 0, -1] = 1.0
    mask = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)
    return v, s, mask


class LGATrSlimTagger(nn.Module):
    """`float64` computes the features from float64 momenta, as for the L-GATr-slim teacher."""

    padded = False   # jets padded to the longest one instead of packed; an execution setting, same weights

    def __init__(self, hidden_v_channels: int, hidden_s_channels: int, num_heads: int,
                 num_blocks: int, mlp_ratio: int, float64: bool = False):
        super().__init__()
        self.dtype = torch.float64 if float64 else torch.float32
        self.net = LGATrSlim(num_blocks=num_blocks, in_v_channels=1, out_v_channels=0,
                             hidden_v_channels=hidden_v_channels, in_s_channels=8,
                             out_s_channels=1, hidden_s_channels=hidden_s_channels,
                             num_heads=num_heads, mlp_ratio=mlp_ratio, attn_ratio=1)

    def forward(self, batch: JetBatch) -> torch.Tensor:
        if self.padded:   # tagging-guide's zero-padded path (verified bit for bit)
            v, s, mask = padded_tokens(batch.p4, self.dtype)
            _, out = self.net(v, s, attn_mask=mask.unsqueeze(1).unsqueeze(2))
            return out[:, 0, 0]
        v, s, jet, first = tokens(batch.p4, self.dtype)
        _, out = self.net(v.unsqueeze(0), s.unsqueeze(0), **attention_mask(jet, s.dtype))
        return out.squeeze(0)[first, 0]


class LGATrTagger(nn.Module):
    """The L-GATr teacher. Its features are computed from float64 momenta."""

    padded = False   # jets padded to the longest one instead of packed; an execution setting, same weights

    def __init__(self, hidden_mv_channels: int = 16, hidden_s_channels: int = 32,
                 num_blocks: int = 12, num_heads: int = 8, attn_ratio: int = 2, mlp_ratio: int = 2):
        super().__init__()
        self.net = LGATr(
            num_blocks=num_blocks, in_mv_channels=1, out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels, in_s_channels=8, out_s_channels=None,
            hidden_s_channels=hidden_s_channels,
            attention=dict(num_heads=num_heads, multi_query=False, attn_ratio=attn_ratio,
                           head_scale=True),
            mlp=dict(nonlinearity="gelu", mlp_ratio=mlp_ratio),
            primitives=dict(sparse_gp=True, sparse_linear=False, bivector=True,
                            geometric_product=True, subgroup=True),
        )

    def forward(self, batch: JetBatch) -> torch.Tensor:
        if self.padded:   # tagging-guide's zero-padded path (verified bit for bit)
            v, s, mask = padded_tokens(batch.p4, torch.float64)
            mv, _ = self.net(embed_vector(v), s, attn_mask=mask.unsqueeze(1).unsqueeze(2))
            return extract_scalar(mv)[:, 0, 0, 0]
        v, s, jet, first = tokens(batch.p4, torch.float64)
        mv, _ = self.net(embed_vector(v).unsqueeze(0), s.unsqueeze(0),
                         **attention_mask(jet, s.dtype))
        return extract_scalar(mv).squeeze(0)[first, 0, 0]
