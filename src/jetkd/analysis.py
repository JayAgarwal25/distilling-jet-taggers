"""What a student takes from its teacher (Section 6 of the paper), from the test logits of every run.

Each distilled run is compared with the teacher run it distilled from and with the run of the same
student and seed trained without a teacher. A model's mistakes at signal efficiency eps_S are the
background jets it accepts at its own threshold. Values are means over seeds.
"""

from __future__ import annotations

from collections import defaultdict
from functools import cache
from pathlib import Path

import numpy as np

from jetkd.study import MATRIX_STUDENTS, TEACHERS, Run, paper_runs
from jetkd.teachers import logits_path

EPS = (0.5, 0.3)
PERJET_TEACHERS = ("part_pretrained", "omnilearned_s", "lgatr")


def _logsig(z: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -z)


def kl(z_teacher: np.ndarray, z_student: np.ndarray) -> float:
    """Mean KL divergence from the teacher's predicted probabilities to the student's."""
    p = np.exp(_logsig(z_teacher))
    return float(np.mean(p * (_logsig(z_teacher) - _logsig(z_student))
                         + (1 - p) * (_logsig(-z_teacher) - _logsig(-z_student))))


def accepted_background(z: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray:
    cut = np.quantile(z[y == 1], 1 - eps)
    return (z > cut) & (y == 0)


def shared_background(z_teacher: np.ndarray, z_student: np.ndarray, y: np.ndarray, eps: float) -> float:
    """Of the background jets the teacher accepts, the fraction the student accepts too."""
    t = accepted_background(z_teacher, y, eps)
    return float(np.sum(t & accepted_background(z_student, y, eps)) / np.sum(t))


def confidence_groups(z_teacher: np.ndarray, y: np.ndarray, n: int = 10) -> np.ndarray:
    """Per class, n groups of equal size by the teacher's probability for the true class."""
    idx = np.empty(len(y), dtype=int)
    for cls in (0, 1):
        m = y == cls
        c = (2 * y - 1)[m] * z_teacher[m]
        edges = np.quantile(c, np.linspace(0, 1, n + 1))
        idx[m] = np.clip(np.searchsorted(edges, c, side="right") - 1, 0, n - 1)
    return idx


def _mean(rows: list):
    """Mean over seeds of equally shaped dicts and lists of numbers."""
    if isinstance(rows[0], dict):
        return {k: _mean([r[k] for r in rows]) for k in rows[0]}
    if isinstance(rows[0], list):
        return [_mean(list(v)) for v in zip(*rows, strict=True)]
    return float(np.mean(rows))


def analyze(runs_dir: str | Path = "runs", teachers_dir: str | Path = "data/teachers") -> dict:
    """Every analysis of the paper, keyed `student|teacher` (+ `|other teacher`, `|eps_S x 100`)."""

    @cache
    def student(run_id: str) -> np.ndarray:
        with np.load(Path(runs_dir) / run_id / "predictions_test.npz") as f:
            return f["logits"].astype(np.float64)

    @cache
    def teacher(t: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
        with np.load(logits_path(teachers_dir, t, seed, "test")) as f:
            return f["logits"].astype(np.float64), f["labels"].astype(np.float64)

    def scratch(r: Run) -> str:
        return f"{r.student}__scratch__s{r.seed}"

    cells: dict[tuple, list[Run]] = defaultdict(list)
    for r in paper_runs():
        if r.teacher in TEACHERS and (Path(runs_dir) / r.id / "predictions_test.npz").is_file():
            cells[(r.student, r.teacher)].append(r)

    out = {"fidelity": {}, "perjet": {}, "distinctive": {}, "accepted": {}}
    for (s, t), runs in cells.items():
        rows = {"distilled": [], "scratch": []}
        for r in runs:
            zt, y = teacher(t, r.teacher_seed)
            for arm, z in (("distilled", student(r.id)), ("scratch", student(scratch(r)))):
                rows[arm].append({"kl": kl(zt, z), **{f"shared{round(100 * e)}": shared_background(zt, z, y, e)
                                                      for e in EPS}})
        out["fidelity"][f"{s}|{t}"] = {arm: _mean(v) for arm, v in rows.items()}

    for s in MATRIX_STUDENTS:
        for t in PERJET_TEACHERS:   # probability of the true class in ten groups of teacher confidence
            per = []
            for r in cells[(s, t)]:
                zt, y = teacher(t, r.teacher_seed)
                g, sign = confidence_groups(zt, y), 2 * y - 1
                zs = {"teacher_p": zt, "distilled_p": student(r.id), "scratch_p": student(scratch(r))}
                row = {}
                for cls, c in (("signal", 1), ("background", 0)):
                    m = y == c
                    p = {k: np.exp(_logsig((sign * z)[m])) for k, z in zs.items()}
                    row[cls] = [{k: float(v[g[m] == b].mean()) for k, v in p.items()} for b in range(10)]
                per.append(row)
            out["perjet"][f"{s}|{t}"] = _mean(per)

        for eps in EPS:
            e = round(100 * eps)
            for a in TEACHERS:   # background jets teacher a accepts and teacher b rejects
                for b in TEACHERS:
                    if a == b:
                        continue
                    ra = {r.seed: r for r in cells[(s, a)]}
                    rb = {r.seed: r for r in cells[(s, b)]}
                    per = []
                    for seed in sorted(set(ra) & set(rb)):
                        za, y = teacher(a, ra[seed].teacher_seed)
                        only_a = accepted_background(za, y, eps) & ~accepted_background(
                            teacher(b, rb[seed].teacher_seed)[0], y, eps)
                        per.append({k: float(np.mean(accepted_background(student(run), y, eps)[only_a]))
                                    for k, run in (("student_A", ra[seed].id), ("student_B", rb[seed].id),
                                                   ("scratch", scratch(ra[seed])))})
                    out["distinctive"][f"{s}|{a}|{b}|{e}"] = _mean(per)
            for t in TEACHERS:   # background jets the student accepts, by the teacher's decision
                per = []
                for r in cells[(s, t)]:
                    zt, y = teacher(t, r.teacher_seed)
                    acc = accepted_background(zt, y, eps)
                    per.append({f"{arm}_teacher_{dec}": int(np.sum(accepted_background(student(run), y, eps) & m))
                                for arm, run in (("without", scratch(r)), ("with", r.id))
                                for dec, m in (("accepts", acc), ("rejects", (y == 0) & ~acc))})
                out["accepted"][f"{s}|{t}|{e}"] = _mean(per)
    return out
