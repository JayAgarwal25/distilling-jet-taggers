import numpy as np
import pytest

from jetkd.study import Run
from jetkd.summary import summarize

RUNS = [Run("part_200k", t, ts, s, "part", ())
        for s in (1001, 1002) for t, ts in ((None, None), ("lgatr", s))]


def write(path, logits, labels):
    path.mkdir(parents=True)
    np.savez(path / "predictions_test.npz", logits=logits, labels=labels)


def test_cells_and_paired_gains(tmp_path):
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.5).astype(np.float32)
    for r in RUNS:
        strength = 2.0 + (0.5 if r.teacher else 0.0) + (0.1 if r.seed == 1002 else 0.0)
        write(tmp_path / r.id, (rng.normal(size=4000) + strength * y).astype(np.float32), y)
    teachers = tmp_path / "teachers"
    teachers.mkdir()
    np.savez(teachers / "lgatr_s1001_test.npz", logits=rng.normal(size=4000) + 4 * y, labels=y)

    out = summarize(tmp_path, teachers, runs=RUNS)
    scratch, kd = out["cells"]["part_200k|scratch"], out["cells"]["part_200k|lgatr"]
    assert scratch["seeds"] == kd["seeds"] == [1001, 1002] and kd["gain_seeds"] == 2
    runs = out["runs"]
    gain = np.mean([runs[f"part_200k__kd_lgatr_t{s}__s{s}"]["auc"]
                    - runs[f"part_200k__scratch__s{s}"]["auc"] for s in (1001, 1002)])
    assert kd["gain_auc"] == pytest.approx(gain) and kd["gain_auc"] > 0
    assert out["teachers"]["lgatr"]["auc_sd"] is None and out["teachers"]["lgatr"]["auc"] > kd["auc"]
