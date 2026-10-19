"""A small Particle Transformer: self-attention over particles with a learned pairwise bias.

The particle features enter a pre-norm Transformer; an MLP turns the four pairwise
features into one additive attention bias per head, as in ParT (Qu, Li and Qian,
arXiv:2202.03772). The output is mean pooled over particles.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from jetkd.data import JetBatch
from jetkd.features import dense_tagging_features, pairwise_features


class PairBiasAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, bias, padding):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.d_head).unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head) + bias
        scores = scores.masked_fill(padding[:, None, None, :], torch.finfo(scores.dtype).min)
        w = self.dropout(F.softmax(scores, dim=-1))
        return self.out((w @ v).transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = PairBiasAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_dim, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias, padding):
        x = x + self.drop(self.attn(self.norm1(x), bias, padding))
        return x + self.drop(self.ffn(self.norm2(x)))


class ParTTagger(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 4, num_blocks: int = 4,
                 ffn_dim: int = 256, pair_hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(7, d_model)
        self.pair_mlp = nn.Sequential(nn.Linear(4, pair_hidden), nn.GELU(),
                                      nn.Linear(pair_hidden, num_heads))
        self.blocks = nn.ModuleList(
            [Block(d_model, num_heads, ffn_dim, dropout) for _ in range(num_blocks)])
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d_model, 1))

    def forward(self, batch: JetBatch) -> torch.Tensor:
        n = int(batch.mask.sum(dim=1).max())
        p4, mask = batch.p4[:, :n], batch.mask[:, :n]
        return self.body(dense_tagging_features(p4, mask), *pairwise_features(p4, mask), mask)

    def body(self, features, pair, pair_mask, mask) -> torch.Tensor:
        """The network on the particle and pair features."""
        x = self.input_proj(features)
        bias = self.pair_mlp(pair).permute(0, 3, 1, 2).masked_fill(~pair_mask.unsqueeze(1), 0.0)
        for block in self.blocks:
            x = block(x, bias, ~mask)
        x = self.out_norm(x)
        m = mask.unsqueeze(-1).to(x.dtype)
        return self.head((x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)).squeeze(-1)
