import sys
from pathlib import Path

from jetkd.study import paper_runs

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from reproduce import jobs  # noqa: E402

TABLES = ("teachers", "students", "matrix_auc", "size_full", "depth_transformer", "depth_slim")


def test_jobs_cover_every_run_and_train_teachers_first():
    trained = set()
    for table in TABLES:
        lines = jobs(table)
        assert lines[-1] == f"python scripts/reproduce.py {table}"
        runs = [line.split("--run ")[1] for line in lines if "train.py --run" in line]
        for i, rid in enumerate(runs):
            if "__kd_" in rid:
                teacher, seed = rid.split("__kd_")[1].split("__")[0].rsplit("_t", 1)
                assert (f"{teacher}__scratch__s{seed}" in runs[:i]
                        or any(f"# {teacher} run {seed}:" in line for line in lines))
        trained.update(runs)
    assert {r.id for r in paper_runs()} <= trained
