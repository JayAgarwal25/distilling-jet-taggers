"""Deep Sets and Deep Sets with message passing (arXiv:2609.26899), for operation counts and timing.

Adapted from github.com/stanford-ai4physics/fm-distillation (MIT), keeping only the paths their
students use. Module names follow theirs, so their checkpoints load.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from jetkd.data import JetBatch


class DynamicTanh(nn.Module):
    def __init__(self, dim: int, alpha: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1,), alpha))
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return torch.tanh(self.alpha * x) * self.weight


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: int, n_out: int, norm: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(n_in, hidden)
        self.act = nn.GELU()
        self.norm = DynamicTanh(hidden) if norm else nn.Identity()
        self.fc2 = nn.Linear(hidden, n_out)

    def forward(self, x):
        return self.fc2(self.norm(self.act(self.fc1(x))))


def safe_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.where(x > 0, x, 1.0))


def pair_features(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """ln m^2, ln dR and ln kT of every pair, from (d eta, d phi, ln pT, ln E) per particle."""
    xi, xj = x.unsqueeze(2), x.unsqueeze(1)
    deta, dphi = xi[..., 0] - xj[..., 0], xi[..., 1] - xj[..., 1]
    pti, ptj = xi[..., 2].exp(), xj[..., 2].exp()
    m2 = 2 * pti * ptj * (torch.cosh(deta) - torch.cos(dphi))
    dr2 = deta.square() + dphi.square()
    kt = torch.minimum(pti, ptj) * dr2.sqrt()
    return torch.stack([safe_log(m2), 0.5 * safe_log(dr2), safe_log(kt)], -1) * mask


class DeepSetsInteraction(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.edge_mlp = MLP(2 * dim + 3, 2 * dim, dim)

    def forward(self, h, edges, edge_mask):
        n = h.shape[1]
        hi, hj = h.unsqueeze(2).expand(-1, -1, n, -1), h.unsqueeze(1).expand(-1, n, -1, -1)
        msg = self.edge_mlp(torch.cat([hi, hj, edges], -1)) * edge_mask
        return msg.sum(2) / edge_mask.sum(2).clamp(min=1.0)


class DeepSetsBody(nn.Module):
    def __init__(self, dim: int, phi_layers: int, interaction_layers: int, interaction_k: int):
        super().__init__()
        self.interaction_k = interaction_k
        self.embed = MLP(4, 2 * dim, dim, norm=True)
        self.phi_norms = nn.ModuleList(DynamicTanh(dim) for _ in range(phi_layers - 1))
        self.phi_blocks = nn.ModuleList(MLP(dim, 2 * dim, dim) for _ in range(phi_layers - 1))
        self.interaction_norms = nn.ModuleList(DynamicTanh(dim) for _ in range(interaction_layers))
        self.interaction_blocks = nn.ModuleList(DeepSetsInteraction(dim) for _ in range(interaction_layers))

    def forward(self, x):
        if self.interaction_k:
            x = x[:, :self.interaction_k]
        mask = (x[:, :, 2:3] != 0).float()   # padding is all zero
        h = self.embed(x) * mask
        if len(self.interaction_blocks):
            edge_mask = (mask @ mask.transpose(1, 2)).unsqueeze(-1)
            edges = pair_features(x, edge_mask)
            for norm, block in zip(self.interaction_norms, self.interaction_blocks, strict=True):
                h = h + block(norm(h), edges, edge_mask) * mask
        for norm, phi in zip(self.phi_norms, self.phi_blocks, strict=True):
            h = h + phi(norm(h)) * mask
        return (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)


class DeepSetsHead(nn.Module):
    def __init__(self, dim: int, rho_layers: int):
        super().__init__()
        self.rho_norms = nn.ModuleList(DynamicTanh(dim) for _ in range(rho_layers))
        self.rho = nn.ModuleList(MLP(dim, 2 * dim, dim) for _ in range(rho_layers))
        self.out = nn.Linear(dim, 2)

    def forward(self, z):
        for norm, rho in zip(self.rho_norms, self.rho, strict=True):
            z = z + rho(norm(z))
        return self.out(z)


class DeepSets(nn.Module):
    def __init__(self, dim: int = 32, phi_layers: int = 2, rho_layers: int = 1,
                 interaction_layers: int = 0, interaction_k: int = 0):
        super().__init__()
        self.body = DeepSetsBody(dim, phi_layers, interaction_layers, interaction_k)
        self.classifier = DeepSetsHead(dim, rho_layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.body(x))


def omnilearned_features(p4: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(d eta, d phi, ln pT, ln E) per particle, angles relative to the jet axis, zero on padding."""
    p = p4.double()
    m = mask.unsqueeze(-1).double()
    e, px, py, pz = p.unbind(-1)
    pt = torch.hypot(px, py).clamp(min=1e-12)
    jet = (p * m).sum(1)
    jet_pt = torch.hypot(jet[:, 1], jet[:, 2]).clamp(min=1e-12)
    deta = torch.asinh(pz / pt) - torch.asinh(jet[:, 3] / jet_pt)[:, None]
    dphi = torch.atan2(py, px) - torch.atan2(jet[:, 2], jet[:, 1])[:, None]
    dphi = (dphi + math.pi) % (2 * math.pi) - math.pi
    return (torch.stack([deta, dphi, pt.log(), e.clamp(min=1e-12).log()], -1) * m).float()


class DeepSetsTagger(nn.Module):
    net_class = DeepSets

    def __init__(self, message_passing: bool = False):
        super().__init__()
        # their "distillnet" size; message passing over the 64 leading particles
        mp = dict(interaction_layers=1, interaction_k=64) if message_passing else {}
        self.net = DeepSets(**mp)

    def forward(self, batch: JetBatch) -> torch.Tensor:
        n = int(batch.mask[:, :150].sum(1).max())   # their files keep 150 particles
        z = self.net(omnilearned_features(batch.p4[:, :n], batch.mask[:, :n]))
        return z[:, 1] - z[:, 0]
