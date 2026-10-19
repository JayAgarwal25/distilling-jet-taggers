from collections import Counter
from pathlib import Path

from jetkd import study

RUNS_YAML = Path(__file__).resolve().parents[1] / "configs" / "runs.yaml"


def test_runs_yaml_is_current():
    assert RUNS_YAML.read_text() == study.dump(study.paper_runs())
    assert study.load(RUNS_YAML) == study.paper_runs()


def test_run_counts():
    runs = study.paper_runs()
    # matrix, students of the ensemble, width, depth, trained teachers
    assert len({r.id for r in runs}) == len(runs) == 105 + 15 + 72 + 56 + 12


def test_teacher_seeds():
    for r in study.paper_runs():
        if r.teacher:
            assert r.teacher_seed == (r.seed if r.teacher in study.PAIRED_TEACHERS else 1001)


def test_every_matrix_cell_has_three_seeds():
    cells = Counter((r.student, r.teacher) for r in study.paper_runs() if "matrix_auc" in r.tables)
    assert len(cells) == 5 * 7 and set(cells.values()) == {3}


def test_reference_files_are_current():
    from jetkd.reference import files

    root = RUNS_YAML.parents[1]
    for path, text in files().items():
        if path != "configs/RUNS.md" or "| AUC |" not in (root / path).read_text():
            assert (root / path).read_text() == text, path
