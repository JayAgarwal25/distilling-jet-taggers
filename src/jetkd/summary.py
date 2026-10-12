"""The paper's numbers from saved predictions: test metrics per run, then per cell over seeds.

A cell is one student trained from scratch or distilled from one teacher. Its value is
the mean over seeds, with the seed standard deviation where there are several seeds.
The gain of a distilled cell is paired: each run minus the scratch run of the same
student and seed, averaged over the seeds that have both.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from jetkd.metrics import binary_metrics
from jetkd.study import Run, paper_runs

METRICS = ("auc", "rej50", "rej30")


def metrics_of(path: Path) -> dict:
    with np.load(path) as f:
        m = binary_metrics(f["logits"], f["labels"])
    return {k: m[k] for k in METRICS}


def mean_sd(rows: list[dict]) -> dict:
    out = {}
    for m in METRICS:
        v = np.array([r[m] for r in rows])
        out[m] = float(v.mean())
        out[f"{m}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else None
    return out


def summarize(runs_dir: str | Path = "runs", teachers_dir: str | Path = "data/teachers",
              runs: list[Run] | None = None) -> dict:
    """Metrics of every paper run found in `runs_dir`, every cell and every teacher."""
    runs = runs or paper_runs()
    per_run = {r.id: metrics_of(p) for r in runs
               if (p := Path(runs_dir) / r.id / "predictions_test.npz").is_file()}

    groups = defaultdict(list)
    for r in runs:
        if r.id in per_run:
            groups[(r.student, r.teacher)].append(r)
    cells = {}
    for (student, teacher), members in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        cell = {"seeds": sorted(r.seed for r in members),
                **mean_sd([per_run[r.id] for r in members])}
        if teacher is not None:
            pairs = [(per_run[r.id], per_run.get(f"{student}__scratch__s{r.seed}")) for r in members]
            pairs = [(kd, sc) for kd, sc in pairs if sc is not None]
            for m in METRICS:
                cell[f"gain_{m}"] = float(np.mean([kd[m] - sc[m] for kd, sc in pairs])) if pairs else None
            cell["gain_seeds"] = len(pairs)
        cells[f"{student}|{teacher or 'scratch'}"] = cell

    teachers = defaultdict(list)
    for p in sorted(Path(teachers_dir).glob("*_s*_test.npz")):
        teacher, seed = p.name.removesuffix("_test.npz").rsplit("_s", 1)
        teachers[teacher].append({"seed": int(seed), **metrics_of(p)})
    teachers = {t: {"seeds": [r["seed"] for r in rows], **mean_sd(rows)}
                for t, rows in teachers.items()}
    return {"runs": per_run, "cells": cells, "teachers": teachers}
