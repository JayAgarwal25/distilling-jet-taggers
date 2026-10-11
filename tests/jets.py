import numpy as np
import torch


def random_jets(n: int, max_particles: int = 200, seed: int = 0) -> torch.Tensor:
    """(n, max_particles, 4) massive particles, zero padded, in the dataset's layout."""
    rng = np.random.default_rng(seed)
    p4 = np.zeros((n, max_particles, 4), dtype=np.float32)
    for i, c in enumerate(rng.integers(2, min(60, max_particles), size=n)):
        p = rng.normal(size=(c, 3)) * [30, 30, 100]
        p4[i, :c, 1:] = p
        p4[i, :c, 0] = np.sqrt((p**2).sum(axis=1) + 0.01)
    return torch.from_numpy(p4)
