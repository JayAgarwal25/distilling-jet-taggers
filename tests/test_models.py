import pytest
import torch
from jets import random_jets

from jetkd.data import JetBatch, particle_mask
from jetkd.models import MODELS, build

PARAMETERS = {
    "lgatr": 1_077_474,
    "lgatr_slim": 2_030_817,
    "lgatr_slim_200k": 202_213,
    "lgatr_slim_200k_L1": 40_837,
    "lgatr_slim_200k_L10": 403_933,
    "lgatr_slim_200k_L2": 81_181,
    "lgatr_slim_200k_L3": 121_525,
    "lgatr_slim_200k_L4": 161_869,
    "lgatr_slim_200k_L6": 242_557,
    "lgatr_slim_200k_L8": 323_245,
    "lgatr_slim_deep_200k": 180_969,
    "lgatr_slim_deep_20k": 20_405,
    "lgatr_slim_deep_2k": 1_763,
    "lorentznet_200k": 202_511,
    "mipart": 656_949,
    "mipart_200k": 199_837,
    "part_200k": 205_093,
    "transformer_200k": 188_113,
    "transformer_200k_L0": 433,
    "transformer_200k_L1": 19_201,
    "transformer_200k_L2": 37_969,
    "transformer_200k_L3": 56_737,
    "transformer_200k_L4": 75_505,
    "transformer_200k_L6": 113_041,
    "transformer_200k_L8": 150_577,
    "transformer_deep_200k": 84_449,
    "transformer_deep_20k": 16_465,
    "transformer_deep_2k": 1_237,
}


@pytest.fixture(scope="module")
def batch():
    p4 = random_jets(6, max_particles=80)
    return JetBatch(p4, particle_mask(p4), torch.zeros(6), torch.arange(6))


def test_parameter_counts():
    assert set(MODELS) == set(PARAMETERS)
    for name, n in PARAMETERS.items():
        assert sum(p.numel() for p in build(name).parameters()) == n, name


@pytest.mark.parametrize("name", ["transformer_200k", "part_200k", "mipart_200k", "lgatr_slim_200k",
                                  "lorentznet_200k", "lgatr"])
def test_one_logit_per_jet_independent_of_the_batch(name, batch):
    model = build(name).eval()
    with torch.no_grad():
        z = model(batch)
        alone = torch.cat([model(JetBatch(batch.p4[i:i + 1], batch.mask[i:i + 1], None, None))
                           for i in range(len(batch.p4))])
    assert z.shape == (6,) and torch.isfinite(z).all()
    assert torch.allclose(z, alone, atol=1e-4, rtol=1e-4)
