import sys
from pathlib import Path

from jetkd.study import paper_runs
from jetkd.teachers import ENSEMBLES

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from reproduce import jobs  # noqa: E402

TABLES = ("teachers", "matrix_auc", "size_short", "size_other", "depth_main", "depth_other", "ens_students")


def ready(lines: list[str], i: int, teacher: str, seed: int) -> bool:
    """The teacher's logits exist before line i: imported by hand, or written after training."""
    if any(line.startswith(f"# {teacher} run {seed}:") for line in lines):
        return True
    k = next((j for j, line in enumerate(lines[:i])
              if f"teacher_logits.py --model {teacher} --seed {seed}" in line), None)
    if k is None:
        return False
    if teacher in ENSEMBLES:
        member, seeds = ENSEMBLES[teacher]
        return all(ready(lines, k, member, s) for s in seeds)
    return any(f"train.py --run {teacher}__scratch__s{seed}" in line for line in lines[:k])


def test_jobs_cover_every_run_and_train_teachers_first():
    trained = set()
    for table in TABLES:
        lines = jobs(table)
        assert lines[-1] == f"python scripts/reproduce.py {table}"
        for i, line in enumerate(lines):
            if "train.py --run" in line:
                rid = line.split("--run ")[1]
                if "__kd_" in rid:
                    teacher, seed = rid.split("__kd_")[1].split("__")[0].rsplit("_t", 1)
                    assert ready(lines, i, teacher, int(seed)), (table, rid)
                trained.add(rid)
    assert {r.id for r in paper_runs()} <= trained
