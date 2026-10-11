from collections import Counter
from pathlib import Path

from jetkd import study

RUNS_YAML = Path(__file__).resolve().parents[1] / "configs" / "runs.yaml"


def test_runs_yaml_matches_the_rules():
    assert RUNS_YAML.read_text() == study.dump(study.paper_runs())
    assert study.load(RUNS_YAML) == study.paper_runs()


def test_run_count_and_unique_ids():
    runs = study.paper_runs()
    # matrix 5 x 3 seeds x 7 arms, width 6 x 3 x 4, depth 14 new students x 4
    assert len(runs) == 105 + 72 + 56
    assert len({r.id for r in runs}) == len(runs)


def test_teacher_seed_pairing():
    for r in study.paper_runs():
        if r.teacher in study.PAIRED_TEACHERS:
            assert r.teacher_seed == r.seed
        elif r.teacher is not None:
            assert r.teacher_seed == 1001


def test_depth_studies_use_seed_1001_only():
    depth = [r for r in study.paper_runs() if "_L" in r.student]
    assert depth and {r.seed for r in depth} == {1001}


def test_every_matrix_cell_has_three_seeds():
    cells = Counter((r.student, r.teacher) for r in study.paper_runs()
                    if "complete" in r.tables)
    assert len(cells) == len(study.MATRIX_STUDENTS) * (len(study.TEACHERS) + 1)
    assert set(cells.values()) == {3}
