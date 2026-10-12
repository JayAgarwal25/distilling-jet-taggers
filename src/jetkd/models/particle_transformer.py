"""Particle Transformer (Qu, Li and Qian, arXiv:2202.03772), as our ParT teacher was trained.

Adapted from lloca 1.3.5 (Spinner et al.; MIT), whose version follows weaver-core's ParT.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from jetkd.data import JetBatch
from jetkd.models.mipart import dense_inputs, trim


def pairwise_features(xi, xj, eps=1e-8):
    """ln m^2, ln kT, ln z and ln dR of particle pairs, with x = (px, py, pz, E) on dim 1."""
    def pt_rap_phi(x):
        px, py, pz, e = x.split((1, 1, 1, 1), dim=1)
        pt = torch.sqrt(x[:, :2].square().sum(dim=1, keepdim=True))
        rap = 0.5 * torch.log((1 + (2 * pz) / (e - pz).clamp(min=1e-20)).clamp(min=1e-20))
        return pt, rap, torch.atan2(py, px)

    pti, rapi, phii = pt_rap_phi(xi)
    ptj, rapj, phij = pt_rap_phi(xj)
    xij = xi + xj
    m2 = (xij[:, 3:4].square() - xij[:, :3].square().sum(dim=1, keepdim=True)).clamp(min=eps)
    dphi = (phii - phij + math.pi) % (2 * math.pi) - math.pi
    delta = ((rapi - rapj) ** 2 + dphi**2).sqrt()
    ptmin = torch.minimum(pti, ptj)
    return torch.cat([
        torch.log(m2),
        torch.log((ptmin * delta).clamp(min=eps)),
        torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps)),
        torch.log(delta.clamp(min=eps)),
    ], dim=1)


class Embed(nn.Module):
    def __init__(self, input_dim, dims):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        layers = []
        for dim in dims:
            layers += [nn.LayerNorm(input_dim), nn.Linear(input_dim, dim), nn.GELU()]
            input_dim = dim
        self.embed = nn.Sequential(*layers)

    def forward(self, x):   # (batch, channels, particles) -> (batch, particles, dim)
        return self.embed(self.input_bn(x).transpose(1, 2).contiguous())


class PairEmbed(nn.Module):
    """Pairwise features of the particles in `mask` to one attention bias per head."""

    def __init__(self, dims):
        super().__init__()
        layers, input_dim = [nn.BatchNorm1d(4)], 4
        for dim in dims:
            layers += [nn.Conv1d(input_dim, dim, 1), nn.BatchNorm1d(dim), nn.GELU()]
            input_dim = dim
        self.embed = nn.Sequential(*layers)
        self.out_dim = dims[-1]

    def forward(self, v, mask):
        batch_size, _, n = v.size()
        with torch.no_grad():
            pairs = (mask.unsqueeze(-1) * mask.unsqueeze(-2)).float().tril()
            b, _, i, j = pairs.nonzero(as_tuple=True)
            x = pairwise_features(v.unsqueeze(-1), v.unsqueeze(-2))
            x = x.permute(0, 2, 3, 1)[b, i, j, :].T.unsqueeze(0).contiguous()
        elements = self.embed(x).squeeze(0).T
        y = torch.zeros(batch_size, n, n, self.out_dim, dtype=elements.dtype, device=elements.device)
        y[b, i, j, :] = elements
        y[b, j, i, :] = elements
        return y.permute(0, 3, 1, 2).contiguous()


# lloca's LLoCa attention with identity frames, which is plain scaled dot-product
# attention; verified bit for bit against lloca 1.3.5's Attention
class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.embed_dim, self.num_heads, self.dropout = embed_dim, num_heads, dropout
        self.head_dim = embed_dim // num_heads
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, padding, bias=None):
        bsz, tgt_len, _ = query.shape
        src_len = key.shape[1]
        mask = torch.zeros_like(padding, dtype=query.dtype).masked_fill_(padding, float("-inf"))
        mask = mask.reshape(bsz, src_len)[:, None, None, :].expand(-1, self.num_heads, -1, -1)
        if bias is not None:
            mask = bias + mask
        q, k, v = F._in_projection_packed(query, key, value, self.in_proj.weight, self.in_proj.bias)
        q, k, v = (t.reshape(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
                   for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, mask, self.dropout if self.training else 0.0)
        return self.out_proj(out.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim))


class Block(nn.Module):
    """ParT's block (normformer with head and residual scales), batch first."""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        ffn_dim = 4 * embed_dim
        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, dropout)
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

    def forward(self, x, padding, bias=None, x_cls=None):
        if x_cls is not None:
            with torch.no_grad():
                padding = torch.cat((torch.zeros_like(padding[:, :1]), padding), dim=1)
            residual = x_cls
            u = self.pre_attn_norm(torch.cat((x_cls, x), dim=1))
            x = self.attn(x_cls, u, u, padding)
        else:
            residual = x
            x = self.pre_attn_norm(x)
            x = self.attn(x, x, x, padding, bias)
        bsz, n, _ = x.size()
        x = torch.einsum("bthd,h->btdh", x.reshape(bsz, n, self.num_heads, self.head_dim),
                         self.c_attn)
        x = self.dropout(self.post_attn_norm(x.reshape(bsz, n, self.embed_dim)))
        x += residual

        residual = x
        x = self.act_dropout(self.act(self.fc1(self.pre_fc_norm(x))))
        x = self.dropout(self.fc2(self.post_fc_norm(x)))
        x += torch.mul(self.w_resid, residual)
        return x


# lloca 2.0's ParticleTransformer cannot run our checkpoint (it disables sequence trimming
# and orders the pairwise features differently), so this follows lloca 1.3.5. Verified:
# same initial weights and layer outputs; checkpoint logits agree to 1e-5 on CPU.
class ParticleTransformer(nn.Module):
    def __init__(self, embed_dims=(128, 512, 128), pair_embed_dims=(64, 64, 64), num_heads=8,
                 num_layers=8, num_cls_layers=2):
        super().__init__()
        embed_dim = embed_dims[-1]
        self.embed = Embed(7, embed_dims)
        self.pair_embed = PairEmbed((*pair_embed_dims, num_heads))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(num_layers)])
        self.cls_blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, dropout=0.0) for _ in range(num_cls_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Sequential(nn.Linear(embed_dim, 1))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self._init_weights()
        self.calls = 0

    def _init_weights(self):
        """MoCo v3 initialisation of every linear layer, then the deeper blocks scaled down."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "in_proj" in name:
                    val = math.sqrt(6.0 / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
                    nn.init.uniform_(module.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for i, block in enumerate(self.blocks, start=1):
            block.attn.out_proj.weight.data.div_(math.sqrt(2.0 * i))
            block.fc2.weight.data.div_(math.sqrt(2.0 * i))

    def forward(self, x, v, mask):
        with torch.no_grad():
            if self.calls < 5:   # as in the original, no trimming on the first five calls
                self.calls += 1
                mask = mask.bool()
            else:
                x, v, mask = trim(x, v, mask.bool(), self.training)
            padding = ~mask.squeeze(1)
        x = self.embed(x).masked_fill(~mask.transpose(1, 2), 0)
        bias = self.pair_embed(v, mask)
        for block in self.blocks:
            x = block(x, padding, bias)
        cls = self.cls_token.expand(x.size(0), 1, -1)
        for block in self.cls_blocks:
            cls = block(x, padding, x_cls=cls)
        return self.fc(self.norm(cls.squeeze(1)))


class ParTTeacher(nn.Module):
    """The ParT teacher; its features are computed from float64 momenta."""

    def __init__(self):
        super().__init__()
        self.net = ParticleTransformer()

    def forward(self, batch: JetBatch) -> torch.Tensor:
        return self.net(*dense_inputs(batch, torch.float64))[:, 0]
