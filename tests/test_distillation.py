import pytest
import torch
import torch.nn.functional as F

from jetkd.distillation import KDConfig, binary_kl, kd_loss


@pytest.fixture
def sample():
    g = torch.Generator().manual_seed(0)
    return (torch.randn(4096, generator=g), torch.randn(4096, generator=g) * 3,
            (torch.rand(4096, generator=g) < 0.5).float())


def test_kl_is_zero_with_zero_gradient_at_the_teacher(sample):
    _, t, _ = sample
    s = t.clone().requires_grad_(True)
    kl = binary_kl(s, t, 2.0)
    kl.backward()
    assert kl.item() == pytest.approx(0.0, abs=1e-6)
    assert s.grad.abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_temperature_changes_the_loss(sample):
    s, t, _ = sample
    losses = [T**2 * binary_kl(s, t, T).item() for T in (1.0, 2.0, 4.0)]
    assert len({round(x, 3) for x in losses}) == 3


def test_alpha_mixes_hard_and_soft(sample):
    s, t, y = sample
    total, parts = kd_loss(s, y, t, KDConfig(alpha=0.0))
    assert total.item() == pytest.approx(parts["hard"].item())
    total, parts = kd_loss(s, y, t, KDConfig(alpha=1.0))
    assert total.item() == pytest.approx(parts["soft"].item())


def test_scratch_is_plain_bce(sample):
    s, _, y = sample
    total, parts = kd_loss(s, y)
    assert parts["soft"] is None
    assert total.item() == pytest.approx(F.binary_cross_entropy_with_logits(s, y).item())


def test_teacher_gets_no_gradient_and_extremes_stay_finite():
    s = torch.tensor([300.0, -300.0, 0.0], requires_grad=True)
    t = torch.tensor([-300.0, 300.0, 0.0], requires_grad=True)
    total, _ = kd_loss(s, torch.tensor([1.0, 0.0, 1.0]), t)
    total.backward()
    assert t.grad is None and torch.isfinite(total)
