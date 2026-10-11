import numpy as np
import torch
from jets import random_jets
from torch.utils.data import DataLoader

from jetkd.data import TopTaggingSplit, collate, particle_mask


def test_split_reads_and_batches_in_row_order(tmp_path):
    p4 = random_jets(50).numpy()
    labels = np.arange(50) % 2
    path = tmp_path / "jets.npz"
    np.savez_compressed(path, kinematics_test=p4, labels_test=labels)

    ds = TopTaggingSplit("test", path=path)
    batches = list(DataLoader(ds, batch_size=7, collate_fn=collate))
    assert torch.equal(torch.cat([b.sid for b in batches]), torch.arange(50))
    for b in batches:
        assert torch.equal(b.p4, torch.from_numpy(p4[b.sid.numpy()]))
        assert torch.equal(b.label, torch.from_numpy(labels[b.sid.numpy()].astype(np.float32)))
        assert torch.equal(b.mask, particle_mask(b.p4))


def test_mask_drops_padding_and_near_zero_components():
    p4 = torch.tensor([[[5.0, 1.0, 2.0, 3.0], [5.0, 1.0, 1e-6, 3.0], [0.0, 0.0, 0.0, 0.0]]])
    assert particle_mask(p4).tolist() == [[True, False, False]]
