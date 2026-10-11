import numpy as np
import pytest

from jetkd.metrics import binary_metrics, rejection


def test_rejection_interpolates_between_roc_points():
    fpr, tpr = np.array([0.0, 0.1, 0.2]), np.array([0.0, 0.4, 0.8])
    assert rejection(fpr, tpr, 0.4) == pytest.approx(10.0)
    assert rejection(fpr, tpr, 0.6) == pytest.approx(1 / 0.15)


def test_rejection_takes_the_smallest_fpr_at_a_repeated_tpr():
    fpr, tpr = np.array([0.0, 0.2, 0.3, 0.4, 1.0]), np.array([0.0, 0.5, 0.5, 0.5, 1.0])
    assert rejection(fpr, tpr, 0.5) == pytest.approx(5.0)


def test_metrics_on_scores_with_a_known_roc():
    # signal: 200 at score 2, 200 at score 1; background: 10 at 2, 40 at 1, 950 at 0
    z = np.r_[np.full(200, 2.0), np.full(200, 1.0), np.full(10, 2.0), np.full(40, 1.0),
              np.zeros(950)]
    y = np.r_[np.ones(400), np.zeros(1000)]
    m = binary_metrics(z, y, signal_efficiencies=(0.5, 0.75))
    assert m["rej50"] == pytest.approx(100.0)
    assert m["rej75"] == pytest.approx(100 / 3)


def test_auc_loss_and_accuracy():
    y = np.r_[np.ones(500), np.zeros(500)]
    assert binary_metrics(np.r_[np.full(500, 5.0), np.full(500, -5.0)], y)["auc"] == 1.0
    m = binary_metrics(np.zeros(1000), y)
    assert m["auc"] == 0.5 and m["loss"] == pytest.approx(np.log(2))
    assert binary_metrics(np.array([800.0, -800.0]), np.array([1.0, 0.0]))["loss"] == 0.0
