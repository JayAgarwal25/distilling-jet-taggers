"""MIParT (Wu et al., arXiv:2406.12616) with the configuration options our runs use.

Adapted from the copy in lloca-experiments (Spinner et al.) of the official code
(github.com/USST-HEP/MIParT, MIT). Particle features and four-momenta enter; ParT's
pairwise features feed both the More-Interaction attention and the attention bias of
the standard blocks; class-attention blocks read out a class token. Only the options
our runs use are kept. Verified to reproduce the original's initial weights, logits and
our trained checkpoints bit for bit.
"""

from __future__ import annotations

import math
import random

import torch
from torch import nn
from torch_geometric.utils import to_dense_batch

from jetkd.data import JetBatch
from jetkd.models.inputs import embed_sparse


def pairwise_lv_fts(xi, xj, eps=1e-8):
    """ln kT, ln z, ln dR and ln m^2 of particle pairs, with x = (px, py, pz, E) on dim 1."""
    def pt_rap_phi(x):
        px, py, pz, e = x.split((1, 1, 1, 1), dim=1)
        pt = torch.sqrt(x[:, :2].square().sum(dim=1, keepdim=True))
        rap = 0.5 * torch.log(1 + (2 * pz) / (e - pz).clamp(min=1e-20))
        return pt, rap, torch.atan2(py, px)

    pti, rapi, phii = pt_rap_phi(xi)
    ptj, rapj, phij = pt_rap_phi(xj)
    dphi = (phii - phij + math.pi) % (2 * math.pi) - math.pi
    delta = ((rapi - rapj) ** 2 + dphi**2).sqrt()
    ptmin = torch.minimum(pti, ptj)
    xij = xi + xj
    m2 = (xij[:, 3:4].square() - xij[:, :3].square().sum(dim=1, keepdim=True)).clamp(min=eps)
    return torch.cat([
        torch.log((ptmin * delta).clamp(min=eps)),
        torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps)),
        torch.log(delta.clamp(min=eps)),
        torch.log(m2),
    ], dim=1)


def trim(x, v, mask, training):
    """ParT's sequence trimming: drop padding, and in training also a random few percent of
    particles. An augmentation that draws from Python's `random` and from torch."""
    if training:
        q = min(1, random.uniform(0.9, 1.02))
        maxlen = torch.quantile(mask.type_as(x).sum(dim=-1), q).long()
        rand = torch.rand_like(mask.type_as(x)).masked_fill_(~mask, -1)
        perm = rand.argsort(dim=-1, descending=True)
        mask = torch.gather(mask, -1, perm)
        x = torch.gather(x, -1, perm.expand_as(x))
        v = torch.gather(v, -1, perm.expand_as(v))
    else:
        maxlen = mask.sum(dim=-1).max()
    maxlen = max(maxlen, 1)
    return x[:, :, :maxlen], v[:, :, :maxlen], mask[:, :, :maxlen]


class Embed(nn.Module):
    def __init__(self, input_dim, dims):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        layers = []
        for dim in dims:
            layers += [nn.LayerNorm(input_dim), nn.Linear(input_dim, dim), nn.GELU()]
            input_dim = dim
        self.embed = nn.Sequential(*layers)

    def forward(self, x):   # (batch, channels, particles) -> (particles, batch, dim)
        return self.embed(self.input_bn(x).permute(2, 0, 1).contiguous())


class PairEmbed(nn.Module):
    """Pairwise features to MI weights (dims[-2] channels) and attention biases (dims[-1])."""

    def __init__(self, dims):
        super().__init__()
        layers, input_dim = [nn.BatchNorm1d(4)], 4
        for dim in dims[:3]:
            layers += [nn.Conv1d(input_dim, dim, 1), nn.BatchNorm1d(dim), nn.GELU()]
            input_dim = dim
        self.embed1 = nn.Sequential(*layers[:-1])
        self.embed2 = nn.Sequential(nn.Conv1d(dims[-2], dims[-1], 1), nn.BatchNorm1d(dims[-1]))
        self.out_dims = dims[-2], dims[-1]

    def forward(self, v):
        batch_size, _, n = v.size()
        with torch.no_grad():
            i, j = torch.tril_indices(n, n, offset=0, device=v.device)
            v = v.unsqueeze(-1).repeat(1, 1, 1, n)
            x = pairwise_lv_fts(v[:, :, i, j], v[:, :, j, i])
        out = []
        for embed, dim in zip((self.embed1, self.embed2), self.out_dims, strict=True):
            x = embed(x)
            y = torch.zeros(batch_size, dim, n, n, dtype=x.dtype, device=x.device)
            y[:, :, i, j] = x
            y[:, :, j, i] = x
            out.append(y)
        return out


class MIAttention(nn.Module):
    """Attention whose weights are given, computed from the pairwise features alone."""

    def __init__(self, dim, num_heads, attn_drop):
        super().__init__()
        self.num_heads = num_heads
        self.vv = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x, weights):
        x = x.transpose(0, 1)
        bsz, n, dim = x.shape
        head_dim = dim // self.num_heads
        v = self.vv(x).reshape(bsz, n, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.contiguous().view(bsz * self.num_heads, n, head_dim)
        out = torch.bmm(self.attn_drop(weights), v)
        return out.transpose(0, 1).contiguous().view(n, bsz, dim)


class Block(nn.Module):
    """ParT's block (normformer with head and residual scales); `mi=True` uses MI attention."""

    def __init__(self, embed_dim, num_heads, dropout=0.1, mi=False):
        super().__init__()
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        ffn_dim = embed_dim * 4
        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = (MIAttention(embed_dim, num_heads, dropout) if mi
                     else nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout))
        self.post_attn_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.act = nn.GELU()
        self.act_dropout = nn.Dropout(dropout)
        self.post_fc_norm = nn.LayerNorm(ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.c_attn = nn.Parameter(torch.ones(num_heads))
        self.w_resid = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None, mi_weights=None):
        if mi_weights is not None:
            residual = x
            x = self.attn(self.pre_attn_norm(x), mi_weights)
        elif x_cls is not None:
            with torch.no_grad():
                padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), 1)
            residual = x_cls
            u = self.pre_attn_norm(torch.cat((x_cls, x), dim=0))
            x = self.attn(x_cls, u, u, key_padding_mask=padding_mask)[0]
        else:
            residual = x
            x = self.pre_attn_norm(x)
            x = self.attn(x, x, x, key_padding_mask=padding_mask, attn_mask=attn_mask)[0]
        n = x.size(0)
        x = torch.einsum("tbhd,h->tbdh", x.view(n, -1, self.num_heads, self.head_dim), self.c_attn)
        x = self.dropout(self.post_attn_norm(x.reshape(n, -1, self.embed_dim)))
        x += residual

        residual = x
        x = self.act_dropout(self.act(self.fc1(self.pre_fc_norm(x))))
        x = self.dropout(self.fc2(self.post_fc_norm(x)))
        x += torch.mul(self.w_resid, residual)
        return x


class MIParT(nn.Module):
    """The network: (batch, 7, particles) features and (batch, 4, particles) momenta to logits."""

    def __init__(self, embed_dims=(128, 512, 64), pair_embed_dims=(64, 64, 64), num_heads=8,
                 num_MIlayers=5, num_layers=5, num_cls_layers=2):
        super().__init__()
        embed_dim = embed_dims[-1]
        self.num_mi_heads = pair_embed_dims[-1]
        self.embed = Embed(7, embed_dims)
        self.pair_embed = PairEmbed(list(pair_embed_dims) + [num_heads])
        self.blocks = nn.ModuleList(
            [Block(embed_dim, self.num_mi_heads, mi=True) for _ in range(num_MIlayers)])
        self.blocks2 = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(num_layers)])
        self.cls_blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, dropout=0.0) for _ in range(num_cls_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Sequential(nn.Linear(embed_dim, 1))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.calls = 0

    def trim(self, x, v, mask):
        if self.calls < 5:
            self.calls += 1
            return x, v, mask
        return trim(x, v, mask, self.training)

    def forward(self, x, v, mask):
        with torch.no_grad():
            x, v, mask = self.trim(x, v, mask.bool())
            padding = ~mask.squeeze(1)
        x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)
        n, bsz, _ = x.shape
        weights, bias = self.pair_embed(v)
        bias = bias.contiguous().view(-1, n, n)
        weights = weights.contiguous().view(bsz, self.num_mi_heads, n, n)
        weights = weights.masked_fill(padding[:, None, None, :], float("-inf"))
        weights = weights.view(bsz * self.num_mi_heads, n, n).softmax(dim=-1)
        for block in self.blocks:
            x = block(x, mi_weights=weights)
        for block in self.blocks2:
            x = block(x, padding_mask=padding, attn_mask=bias)
        cls = self.cls_token.expand(1, x.size(1), -1)
        for block in self.cls_blocks:
            cls = block(x, x_cls=cls, padding_mask=padding)
        return self.fc(self.norm(cls).squeeze(0))


def dense_inputs(batch: JetBatch, dtype: torch.dtype):
    """Features (B, 7, N), momenta (B, 4, N) as (px, py, pz, E) and mask (B, 1, N), as ParT takes them."""
    n = int(batch.mask.sum(dim=1).max())
    p4, feats, spurion, jet, _ = embed_sparse(batch.p4[:, :n], batch.mask[:, :n], dtype)
    particle = ~spurion
    x, mask = to_dense_batch(feats[particle], jet[particle])
    v, _ = to_dense_batch(p4[particle].float()[:, [1, 2, 3, 0]], jet[particle])
    return x.transpose(1, 2), v.transpose(1, 2), mask.unsqueeze(1).float()


class MIParTTagger(nn.Module):
    def __init__(self, float64: bool = False, **net):
        super().__init__()
        self.dtype = torch.float64 if float64 else torch.float32
        self.net = MIParT(**net)

    def forward(self, batch: JetBatch) -> torch.Tensor:
        return self.net(*dense_inputs(batch, self.dtype))[:, 0]
