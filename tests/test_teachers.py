import numpy as np
import pytest
import torch

from jetkd.teachers import check_rows, ensemble_logits, from_probabilities, logits_path, save_logits
from jetkd.train import load_teacher_logits


@pytest.fixture
def data(tmp_path):
    rng = np.random.default_rng(0)
    k = np.zeros((40, 5, 4), dtype=np.float32)
    k[:, :2, 1:] = rng.normal(size=(40, 2, 3)) * 50
    k[:, :2, 0] = 200
    np.savez(tmp_path / "jets.npz", kinematics_val=k, labels_val=np.arange(40) % 2)
    return tmp_path / "jets.npz"


def test_saved_logits_load_for_training(tmp_path):
    labels = (np.arange(40) % 3 == 0).astype(np.float32)
    save_logits(tmp_path / "t.npz", np.linspace(-2, 2, 40), labels)
    z = load_teacher_logits(tmp_path / "t.npz", torch.from_numpy(labels[:25]))
    assert z.shape == (25,) and z.dtype == np.float32
    with pytest.raises(ValueError):
        load_teacher_logits(tmp_path / "t.npz", torch.from_numpy(1 - labels))


def test_ensemble_is_the_mean_member_logit(tmp_path):
    labels = (np.arange(10) % 2).astype(np.float32)
    zs = np.random.default_rng(0).normal(size=(3, 10)).astype(np.float32)
    for s, z in zip((1001, 1002, 1003), zs, strict=True):
        save_logits(logits_path(tmp_path, "lgatr_slim", s, "val"), z, labels)
    z, y = ensemble_logits(tmp_path, "lgatr_slim_ens", "val")
    assert z.dtype == np.float32 and np.allclose(z, zs.mean(0)) and np.array_equal(y, labels)
    save_logits(logits_path(tmp_path, "lgatr_slim", 1003, "val"), zs[2], 1 - labels)
    with pytest.raises(ValueError):
        ensemble_logits(tmp_path, "lgatr_slim_ens", "val")


def test_logits_from_saturated_probabilities():
    p_top = np.array([1 - 1e-12, 0.5, 1e-30])
    z = from_probabilities(p_top, 1 - p_top + np.array([0, 0, 0]))
    assert np.isfinite(z).all() and z[0] > 20 and z[1] == 0 and z[2] < -60


def test_rows_must_match_the_dataset(data):
    labels = np.arange(40) % 2
    check_rows(labels, "val", data)
    with pytest.raises(ValueError):
        check_rows(labels[::-1], "val", data)
    with np.load(data) as f:
        k = f["kinematics_val"]
    pt = np.hypot(k[:, :, 1].sum(1), k[:, :, 2].sum(1))
    check_rows(labels, "val", data, jet_pt=pt)
    with pytest.raises(ValueError):
        check_rows(labels, "val", data, jet_pt=pt[::-1], pt_outliers=5)
