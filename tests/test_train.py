import dataclasses
import json

import numpy as np
import pytest
import torch
from jets import random_jets

from jetkd.models import MODELS
from jetkd.recipes import RECIPES, recipe_for
from jetkd.study import paper_runs
from jetkd.train import RunConfig, Trainer, run


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    arrays = {}
    for split, n in (("train", 96), ("val", 32), ("test", 32)):
        arrays[f"kinematics_{split}"] = random_jets(n, max_particles=40, seed=n).numpy()
        arrays[f"labels_{split}"] = np.arange(n) % 2
    np.savez_compressed(d / "jets.npz", **arrays)
    np.savez(d / "teacher.npz", logits=np.linspace(-3, 3, 96, dtype=np.float32),
             sids=np.arange(96), labels=arrays["labels_train"].astype(np.float32))
    return d


def config(data, out, **kw):
    recipe = dataclasses.replace(recipe_for("transformer_200k_L1"), batch_size=16, iterations=6,
                                 validate_every=3)
    return RunConfig(model="transformer_200k_L1", recipe=recipe, data=str(data / "jets.npz"),
                     out=str(out), device="cpu", workers=0, **kw)


def test_every_model_and_run_has_a_recipe():
    assert all(recipe_for(m).name in RECIPES for m in MODELS)
    assert all(recipe_for(r.student).name == r.recipe for r in paper_runs())


def test_distilled_run_writes_its_results(data, tmp_path):
    cfg = config(data, tmp_path, teacher="lgatr", teacher_seed=1001,
                 teacher_logits=str(data / "teacher.npz"))
    results = run(cfg)
    out = tmp_path / cfg.id
    assert cfg.id == "transformer_200k_L1__kd_lgatr_t1001__s1001"
    assert {"best.pt", "last.pt", "config.json", "history.json", "results.json",
            "predictions_val.npz", "predictions_test.npz"} <= {p.name for p in out.iterdir()}
    assert results["test"]["n"] == 32
    assert json.loads((out / "history.json").read_text())["history"][0]["train"]["soft"] > 0


def test_resume_continues_exactly(data, tmp_path):
    whole = Trainer(config(data, tmp_path / "a"))
    whole.fit()
    Trainer(config(data, tmp_path / "b")).fit(stop_after=3)
    resumed = Trainer(config(data, tmp_path / "b"))
    assert resumed.start_step == 3
    resumed.fit()
    def rows(t):
        return [(r["train"], r["val"], r["lr"]) for r in t.history]

    assert rows(whole) == rows(resumed)
    for a, b in zip(whole.model.parameters(), resumed.model.parameters(), strict=True):
        assert torch.equal(a, b)
