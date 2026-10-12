"""Particle features.

Per particle: log pT, log E, log pT/pT_jet, log E/E_jet, dphi, deta and dR, standardised
as for the L-GATr, L-GATr-slim, ParT and MIParT teachers. Per pair: ln dR, ln kT, ln z and ln m.
"""

from __future__ import annotations

import torch
from lloca.utils.lorentz import lorentz_squarednorm

EPS = 1e-10
MASS_REG = 0.005
# (mean, scale) of each per-particle feature, in the order above
STANDARDISATION = ((1.7, 0.7), (2.0, 0.7), (-4.7, 0.7), (-4.7, 0.7), (0, 1), (0, 1), (0.2, 4))


def _nonzero(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x.abs() < EPS, EPS, x)


def pt(p: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((p[..., 1] ** 2 + p[..., 2] ** 2).clamp(min=EPS))


def phi(p: torch.Tensor) -> torch.Tensor:
    return torch.arctan2(_nonzero(p[..., 2]), _nonzero(p[..., 1]))


def eta(p: torch.Tensor) -> torch.Tensor:
    x = p[..., 3] / torch.sqrt(torch.sum(p[..., 1:] ** 2, dim=-1).clamp(min=EPS))
    return 0.5 * (torch.log((1 + x).clamp(min=EPS)) - torch.log((1 - x).clamp(min=EPS)))


def apply_mass_regulator(p4: torch.Tensor, mask: torch.Tensor, mass_reg: float = MASS_REG):
    """Raise the energy of near-massless particles so that m >= mass_reg."""
    out = p4.clone()
    below = (lorentz_squarednorm(out) < mass_reg**2) & mask
    out[..., 0][below] = (out[..., 1:][below] ** 2).sum(dim=-1).add(mass_reg**2).sqrt()
    return out


def tagging_features(p: torch.Tensor, jet: torch.Tensor) -> torch.Tensor:
    """(..., 4) particle and jet momenta to (..., 7) standardised features."""
    # same operations in the same order as lloca-experiments' get_tagging_features,
    # so the values agree to the bit (verified)
    log_pt = pt(p).unsqueeze(-1).log()
    log_e = p[..., 0].unsqueeze(-1).clamp(min=EPS).log()
    log_pt_rel = (pt(p).log() - pt(jet).log()).unsqueeze(-1)
    log_e_rel = (p[..., 0].clamp(min=EPS).log() - jet[..., 0].clamp(min=EPS).log()).unsqueeze(-1)
    dphi = ((phi(p) - phi(jet) + torch.pi) % (2 * torch.pi) - torch.pi).unsqueeze(-1)
    deta = -(eta(p) - eta(jet)).unsqueeze(-1)
    dr = torch.sqrt((dphi**2 + deta**2).clamp(min=EPS))
    feats = (log_pt, log_e, log_pt_rel, log_e_rel, dphi, deta, dr)
    return torch.cat([(f - m) * s for f, (m, s) in zip(feats, STANDARDISATION, strict=True)], dim=-1)


def dense_tagging_features(p4: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B, N, 4) to (B, N, 7), zero on padding."""
    p4 = apply_mass_regulator(p4, mask)
    m = mask.unsqueeze(-1).to(p4.dtype)
    jet = (p4 * m).sum(dim=1, keepdim=True).expand_as(p4)
    return tagging_features(p4, jet) * m


def pairwise_features(p4: torch.Tensor, mask: torch.Tensor):
    """ln dR, ln kT, ln z and ln m of every particle pair: (B, N, N, 4) and the pair mask."""
    p4 = apply_mass_regulator(p4, mask)
    # pseudorapidity and ln m, where ParT uses rapidity and ln m^2: part_200k was trained so
    pt_, eta_, phi_ = pt(p4), eta(p4), phi(p4)
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)

    deta = eta_.unsqueeze(2) - eta_.unsqueeze(1)
    dphi = (phi_.unsqueeze(2) - phi_.unsqueeze(1) + torch.pi) % (2 * torch.pi) - torch.pi
    dr = (deta**2 + dphi**2).clamp(min=EPS).sqrt()
    pt_min = torch.minimum(pt_.unsqueeze(2), pt_.unsqueeze(1))
    kt = pt_min * dr
    z = pt_min / (pt_.unsqueeze(2) + pt_.unsqueeze(1)).clamp(min=EPS)
    mass = lorentz_squarednorm(p4.unsqueeze(2) + p4.unsqueeze(1)).clamp(min=EPS).sqrt()

    pair = torch.stack([x.clamp(min=EPS).log() for x in (dr, kt, z, mass)], dim=-1)
    return pair * pair_mask.unsqueeze(-1).to(pair.dtype), pair_mask
