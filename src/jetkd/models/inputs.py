"""Input pipeline shared by the Transformer and L-GATr taggers.

Adapted from the top tagging wrappers of tagging-guide (Spinner et al.). Jets are
packed into one sequence of real tokens and attention is restricted to tokens of
the same jet.
"""

from __future__ import annotations

import torch

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
    every token, and the features of spurions and padding are zero.
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
    training. Otherwise it is the equivalent dense additive mask.
    """
    if jet.is_cuda and BlockDiagonalMask is not None:
        return {"attn_bias": BlockDiagonalMask.from_seqlens(torch.bincount(jet).tolist())}
    same = jet.unsqueeze(1) == jet.unsqueeze(0)
    mask = torch.zeros(same.shape, dtype=dtype, device=jet.device)
    return {"attn_mask": mask.masked_fill(~same, float("-inf"))}
