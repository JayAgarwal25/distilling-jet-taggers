import numpy as np

from jetkd.analysis import accepted_background, confidence_groups, kl, shared_background


def jets(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % 2).astype(np.float64)
    return rng.normal(size=n) + 2 * y, y


def test_kl_is_zero_for_the_teacher_itself():
    z, _ = jets()
    assert kl(z, z) == 0 and kl(z, z + 1) > 0


def test_shared_background():
    z, y = jets()
    assert shared_background(z, z, y, 0.5) == 1
    assert shared_background(z, -z, y, 0.5) < 0.1
    # each model takes its own threshold: a monotonic change of the logits shares everything
    assert shared_background(z, 3 * z + 1, y, 0.3) == 1


def test_accepted_background_at_the_signal_efficiency():
    z, y = jets()
    cut = np.quantile(z[y == 1], 0.5)
    assert np.array_equal(accepted_background(z, y, 0.5), (z > cut) & (y == 0))


def test_confidence_groups_are_equal_per_class():
    z, y = jets()
    g = confidence_groups(z, y)
    for c in (0, 1):
        assert set(np.bincount(g[y == c])) == {200}
    # higher group, more confident in the true class
    sign = 2 * y - 1
    assert (sign * z)[(y == 1) & (g == 9)].min() >= (sign * z)[(y == 1) & (g == 0)].max()
