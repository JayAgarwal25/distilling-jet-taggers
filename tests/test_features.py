import pytest
import torch
from jets import random_jets
from lloca.utils.lorentz import lorentz_squarednorm

from jetkd.data import particle_mask
from jetkd.features import (
    MASS_REG,
    apply_mass_regulator,
    dense_tagging_features,
    eta,
    pairwise_features,
    tagging_features,
)


@pytest.fixture(scope="module")
def jets():
    p4 = random_jets(64).double()
    return p4, particle_mask(p4)


def test_dense_features_equal_per_jet_features(jets):
    p4, mask = jets
    dense = dense_tagging_features(p4, mask)
    assert dense.shape == (*p4.shape[:2], 7)
    for j in range(0, len(p4), 9):
        p = apply_mass_regulator(p4[j], mask[j])[mask[j]]
        ref = tagging_features(p, p.sum(dim=0).expand_as(p))
        assert torch.allclose(dense[j][mask[j]], ref, atol=1e-12, rtol=0)
    assert not dense[~mask].any()


def test_mass_regulator(jets):
    p4, mask = jets
    out = apply_mass_regulator(p4, mask)
    assert torch.equal(out[..., 1:], p4[..., 1:])
    assert float(lorentz_squarednorm(out)[mask].min()) >= MASS_REG**2 - 1e-9


def test_pairwise_features(jets):
    p4, mask = jets
    pair, pair_mask = pairwise_features(p4, mask)
    assert pair.shape == (*p4.shape[:2], p4.shape[1], 4)
    assert torch.allclose(pair, pair.transpose(1, 2), atol=1e-12, rtol=0)
    assert not pair[~pair_mask].any() and torch.isfinite(pair).all()
    assert pair[0, 0, 0, 2].exp().item() == pytest.approx(0.5)   # z of a particle with itself


def test_eta_is_not_capped():
    assert float(eta(torch.tensor([1e4, 1e-3, 1e-3, 1e4], dtype=torch.float64)).abs()) > 5
