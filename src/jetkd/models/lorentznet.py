"""LorentzNet (Gong et al., arXiv:2201.08187).

Adapted from the copy in lloca-experiments (Spinner et al.) of the official code
(github.com/sdogsq/LorentzNet-release). Particles and spurions form a complete graph
per jet; the seven particle features are the node scalars and the particle momenta
are divided by 20.
"""

from __future__ import annotations

import torch
from lloca.utils.utils import get_edge_index_from_ptr
from torch import nn
from torch_geometric.nn.aggr import MeanAggregation

from jetkd.data import JetBatch
from jetkd.models.inputs import embed_sparse


def segment_sum(data, ids, n):
    return data.new_zeros((n, data.size(1))).index_add_(0, ids, data)


def segment_mean(data, ids, n):
    count = data.new_zeros((n, data.size(1))).index_add_(0, ids, torch.ones_like(data))
    return segment_sum(data, ids, n) / count.clamp(min=1)


# torch.pow here and p * q below, exactly as the original, so results agree to the bit
def normsq(p):
    p2 = torch.pow(p, 2)
    return 2 * p2[..., 0] - p2.sum(dim=-1)


def dot(p, q):
    pq = p * q
    return 2 * pq[..., 0] - pq.sum(dim=-1)


def psi(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


class LGEB(nn.Module):
    """Lorentz group equivariant block: messages from Minkowski norms and products."""

    def __init__(self, n_hidden, n_scalar, c_weight, last_layer=False):
        super().__init__()
        self.c_weight = c_weight
        self.phi_e = nn.Sequential(
            nn.Linear(2 * n_hidden + 2, n_hidden, bias=False), nn.BatchNorm1d(n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), nn.ReLU())
        self.phi_h = nn.Sequential(
            nn.Linear(2 * n_hidden + n_scalar, n_hidden), nn.BatchNorm1d(n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden))
        layer = nn.Linear(n_hidden, 1, bias=False)
        nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi_x = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU(), layer)
        self.phi_m = nn.Sequential(nn.Linear(n_hidden, 1), nn.Sigmoid())
        self.last_layer = last_layer
        if last_layer:
            del self.phi_x

    def forward(self, h, x, edges, scalars):
        i, j = edges
        x_diff = x[i] - x[j]
        norms = psi(normsq(x_diff).unsqueeze(1))
        dots = psi(dot(x[i], x[j]).unsqueeze(1))
        m = self.phi_e(torch.cat([h[i], h[j], norms, dots], dim=1))
        m = m * self.phi_m(m)
        if not self.last_layer:
            trans = torch.clamp(x_diff * self.phi_x(m), min=-100, max=100)
            x = x + segment_mean(trans, i, x.size(0)) * self.c_weight
        h = h + self.phi_h(torch.cat([h, segment_sum(m, i, h.size(0)), scalars], dim=1))
        return h, x


# same initial weights, logits and checkpoints as the original (verified bit for bit)
class LorentzNet(nn.Module):
    def __init__(self, n_hidden, n_layers, c_weight, dropout, n_scalar=7):
        super().__init__()
        self.embedding = nn.Linear(n_scalar, n_hidden)
        self.LGEBs = nn.ModuleList([LGEB(n_hidden, n_scalar, c_weight, last_layer=(i == n_layers - 1))
                                    for i in range(n_layers)])
        self.aggregator = MeanAggregation()
        self.graph_dec = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Dropout(dropout),
                                       nn.Linear(n_hidden, 1))

    def forward(self, scalars, x, edges, jet):
        h = self.embedding(scalars)
        for layer in self.LGEBs:
            h, x = layer(h, x, edges, scalars)
        return self.graph_dec(self.aggregator(h, index=jet))


class LorentzNetTagger(nn.Module):
    def __init__(self, n_hidden=68, n_layers=6, c_weight=5e-3, dropout=0.2):
        super().__init__()
        self.net = LorentzNet(n_hidden, n_layers, c_weight, dropout)

    def forward(self, batch: JetBatch) -> torch.Tensor:
        n = int(batch.mask.sum(dim=1).max())
        p4, feats, spurion, jet, ptr = embed_sparse(batch.p4[:, :n], batch.mask[:, :n],
                                                   torch.float32)
        p4[~spurion] = p4[~spurion] / 20
        edges = get_edge_index_from_ptr(ptr, p4.shape, remove_self_loops=True)
        return self.net(feats, p4.to(feats.dtype), edges, jet)[:, 0]
