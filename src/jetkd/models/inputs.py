"""Input pipelines shared by the taggers.

`embed` follows the top tagging wrappers of tagging-guide and `embed_sparse` the
embedding of lloca-experiments (both by Spinner et al.). Jets are packed into one
sequence of real tokens; attention is restricted to tokens of the same jet.
"""

from __future__ import annotations

import torch
from torch_geometric.utils import scatter

from jetkd.features import apply_mass_regulator, tagging_features

try:
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask
except ImportError:
    BlockDiagonalMask = None

# reference vectors for the L-GATr family: the two beam directions and time
SPURIONS = ((1.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, -1.0), (1.0, 0.0, 0.0, 0.0))


def embed(p4: torch.Tensor, spurions: bool, dtype: torch.dtype):
    """Padded jets to momenta, the seven features and a token mask.

    Spurions, if used, are prepended to every jet. The mass regulator acts on
    every token, and the features of spurions and padding are zero. Unlike
    tagging-guide's dense embedding, the features are computed after the mass
    regulator, as in our training; verified to reproduce the training inputs exactly.
    """
    p4 = p4.to(dtype)
    n_spurions = len(SPURIONS) if spurions else 0
    if spurions:
        ref = torch.tensor(SPURIONS, device=p4.device, dtype=dtype).expand(len(p4), -1, -1)
        p4 = torch.cat([ref, p4], dim=1)
    mask = (p4.abs() > 1e-5).any(dim=-1)
    n = int(mask.sum(dim=-1).max())
    p4, mask = p4[:, :n], mask[:, :n]

    p4 = apply_mass_regulator(p4, mask)
    particle = mask.clone()
    particle[:, :n_spurions] = False
    jet = (p4 * particle.unsqueeze(-1)).sum(dim=1, keepdim=True).expand_as(p4)
    feats = tagging_features(p4, jet)
    feats[~particle] = 0.0
    return p4, feats.float(), mask


def pack(mask: torch.Tensor, *tensors: torch.Tensor):
    """(B, N, ...) tensors to (tokens, ...) for the tokens in `mask`, with jet index and offsets."""
    counts = mask.sum(dim=-1)
    ptr = torch.zeros(len(counts) + 1, dtype=torch.long, device=mask.device)
    ptr[1:] = torch.cumsum(counts, dim=0)
    idx = mask.flatten().nonzero().squeeze(-1)
    jet = torch.repeat_interleave(torch.arange(len(counts), device=mask.device), counts)
    return [t.flatten(0, 1).index_select(0, idx) for t in tensors], jet, ptr


def add_global_token(vectors: torch.Tensor, scalars: torch.Tensor, jet: torch.Tensor,
                     ptr: torch.Tensor):
    """Put a global token in front of each jet, flagged by an extra scalar channel."""
    n_jets = len(ptr) - 1
    n = len(jet) + n_jets
    first = ptr[:-1] + torch.arange(n_jets, device=jet.device)
    rest = torch.arange(len(jet), device=jet.device) + jet + 1
    v = torch.zeros(n, *vectors.shape[1:], dtype=vectors.dtype, device=vectors.device)
    v[rest] = vectors
    s = torch.zeros(n, scalars.shape[1] + 1, dtype=scalars.dtype, device=scalars.device)
    s[rest, :-1] = scalars
    s[:, -1].index_fill_(0, first, 1.0)
    jet = torch.repeat_interleave(torch.arange(n_jets, device=jet.device), ptr.diff() + 1)
    return v, s, jet, first


def attention_mask(jet: torch.Tensor, dtype: torch.dtype) -> dict:
    """Keyword arguments that restrict attention to tokens of the same jet.

    On GPU with xformers installed this is xformers' block-diagonal mask, as in
    training. Otherwise it is the same mask written out densely (verified identical).
    """
    if jet.is_cuda and BlockDiagonalMask is not None:
        return {"attn_bias": BlockDiagonalMask.from_seqlens(torch.bincount(jet).tolist())}
    same = jet.unsqueeze(1) == jet.unsqueeze(0)
    mask = torch.zeros(same.shape, dtype=dtype, device=jet.device)
    return {"attn_mask": mask.masked_fill(~same, float("-inf"))}


def embed_sparse(p4: torch.Tensor, mask: torch.Tensor, dtype: torch.dtype):
    """Particles in `mask`, each jet led by the spurions, as one token sequence.

    Returns momenta (tokens, 4) after the mass regulator, the seven features in
    float32 (zero for spurions), the spurion flags, the jet index and the jet offsets.
    Verified to match lloca-experiments' embed_tagging_data bit for bit.
    """
    n_spurions = len(SPURIONS)
    ref = torch.tensor(SPURIONS, device=p4.device, dtype=dtype).expand(len(p4), -1, -1)
    p4 = torch.cat([ref, p4.to(dtype)], dim=1)
    mask = torch.cat([torch.ones_like(mask[:, :n_spurions]), mask], dim=1)
    spurion = torch.zeros_like(mask)
    spurion[:, :n_spurions] = True
    [p4, spurion], jet, ptr = pack(mask, p4, spurion)

    p4 = apply_mass_regulator(p4, torch.ones_like(spurion))
    particle = ~spurion
    jet_p4 = scatter(p4[particle], jet[particle], dim=0, reduce="sum").index_select(0, jet)
    feats = tagging_features(p4, jet_p4)
    feats[spurion] = 0.0
    return p4, feats.float(), spurion, jet, ptr
