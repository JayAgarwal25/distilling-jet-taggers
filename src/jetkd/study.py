"""The paper's run design: every training run behind a number in the paper.

Three studies share one naming scheme. A run id is
`<student>__<arm>__s<seed>`, where the arm is `scratch` or
`kd_<teacher>_t<teacher seed>`.

- Matrix: five ~200k-parameter students, each trained from scratch and
  distilled from each of six teachers, three seeds per cell.
- Width: the deep Transformer and L-GATr-slim ladders of the Economical Jet
  Taggers (arXiv:2512.17011, Table 7) at 200k, 20k and 2k parameters, from
  scratch and from three teachers, three seeds per cell.
- Depth: the matrix Transformer and L-GATr-slim with only the block count
  changed, from scratch and from the same three teachers, seed 1001 only.

`python -m jetkd.study configs/runs.yaml` writes the run list;
`--check` compares the file on disk with the rules here.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

SEEDS = (1001, 1002, 1003)

TEACHERS = ("part", "mipart", "lgatr", "lgatr_slim", "omnilearned_s", "part_pretrained")
# We trained these three teachers once per seed, and the student of seed s
# distills from the teacher run of seed s. The others are used through their
# seed 1001 run.
PAIRED_TEACHERS = ("part", "lgatr", "lgatr_slim")
SCALING_TEACHERS = ("lgatr", "part_pretrained", "omnilearned_s")

MATRIX_STUDENTS = (
    "transformer_200k", "part_200k", "mipart_200k", "lgatr_slim_200k", "lorentznet_200k",
)
WIDTH_STUDENTS = tuple(
    f"{family}_deep_{size}"
    for family in ("transformer", "lgatr_slim")
    for size in ("200k", "20k", "2k")
)
# block counts per depth study; the matrix student itself is the entry with
# its own block count (10 for the Transformer, 5 for L-GATr-slim)
DEPTHS = {
    "transformer_200k": ((0, 1, 2, 3, 4, 6, 8, 10), 10),
    "lgatr_slim_200k": ((1, 2, 3, 4, 5, 6, 8, 10), 5),
}

# Each family trains under the recipe it was published with. The Transformer
# and the width ladders share one recipe per size class, taken from the
# Economical Jet Taggers; changing the depth never changes the recipe.
RECIPES = {
    "transformer_200k": "size_200k",
    "part_200k": "part",
    "mipart_200k": "mipart",
    "lgatr_slim_200k": "lgatr_slim",
    "lorentznet_200k": "lorentznet",
    **{s: f"size_{s.rsplit('_', 1)[1]}" for s in WIDTH_STUDENTS},
}


@dataclass(frozen=True)
class Run:
    student: str
    teacher: str | None
    teacher_seed: int | None
    seed: int
    recipe: str
    tables: tuple[str, ...]

    @property
    def id(self) -> str:
        return run_id(self.student, self.seed, self.teacher, self.teacher_seed)

    def as_dict(self) -> dict:
        return {
            "id": self.id, "student": self.student, "teacher": self.teacher,
            "teacher_seed": self.teacher_seed, "seed": self.seed, "recipe": self.recipe,
            "tables": list(self.tables),
        }


def run_id(student: str, seed: int, teacher: str | None = None,
           teacher_seed: int | None = None) -> str:
    arm = f"kd_{teacher}_t{teacher_seed}" if teacher else "scratch"
    return f"{student}__{arm}__s{seed}"


def teacher_seed(teacher: str, seed: int) -> int:
    return seed if teacher in PAIRED_TEACHERS else 1001


def depth_student(base: str, blocks: int) -> str:
    return base if blocks == DEPTHS[base][1] else f"{base}_L{blocks}"


def recipe(student: str) -> str:
    base = student.split("_L")[0]
    return RECIPES[base]


def paper_runs() -> list[Run]:
    """Every run in the paper, each with the tables that read it."""
    tables: dict[tuple, set[str]] = {}

    def use(student, seed, teacher, names):
        ts = teacher_seed(teacher, seed) if teacher else None
        tables.setdefault((student, teacher, ts, seed), set()).update(names)

    matrix = {"matrix_auc", "matrix_rej", "gains", "complete"}
    for student in MATRIX_STUDENTS:
        for seed in SEEDS:
            use(student, seed, None, matrix | {"students", "cost_provisional"})
            for teacher in TEACHERS:
                extra = set()
                if teacher == "part_pretrained":
                    extra.add("cost_provisional")
                    if student in ("transformer_200k", "lgatr_slim_200k"):
                        extra.add("benyas")
                use(student, seed, teacher, matrix | extra)

    for student in WIDTH_STUDENTS:
        for seed in SEEDS:
            use(student, seed, None, {"size_full", "size", "cost_provisional"})
            for teacher in SCALING_TEACHERS:
                extra = {"size", "cost_provisional", "benyas"} if teacher == "part_pretrained" else set()
                use(student, seed, teacher, {"size_full"} | extra)

    for base, (depths, _) in DEPTHS.items():
        table = "depth_transformer" if base == "transformer_200k" else "depth_slim"
        for blocks in depths:
            student = depth_student(base, blocks)
            use(student, 1001, None, {table, "depth"})
            for teacher in SCALING_TEACHERS:
                extra = {"depth"} if teacher == "part_pretrained" else set()
                use(student, 1001, teacher, {table} | extra)

    return [Run(student, teacher, ts, seed, recipe(student), tuple(sorted(names)))
            for (student, teacher, ts, seed), names in tables.items()]


HEADER = "# Every training run in the paper. Generated by `python -m jetkd.study`; do not edit.\n"


def dump(runs: list[Run]) -> str:
    body = yaml.safe_dump({"runs": [r.as_dict() for r in runs]},
                          sort_keys=False, default_flow_style=None, width=100)
    return HEADER + body


def load(path: str | Path) -> list[Run]:
    data = yaml.safe_load(Path(path).read_text())
    runs = []
    for d in data["runs"]:
        run = Run(d["student"], d["teacher"], d["teacher_seed"], d["seed"], d["recipe"],
                  tuple(d["tables"]))
        if run.id != d["id"]:
            raise ValueError(f"run id {d['id']!r} does not match its fields ({run.id!r})")
        runs.append(run)
    return runs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path", nargs="?", default="configs/runs.yaml")
    ap.add_argument("--check", action="store_true",
                    help="fail if the file differs from the rules instead of writing it")
    args = ap.parse_args(argv)
    text = dump(paper_runs())
    path = Path(args.path)
    if args.check:
        if not path.is_file() or path.read_text() != text:
            print(f"{path} is out of date; run `python -m jetkd.study {path}`", file=sys.stderr)
            return 1
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"wrote {len(paper_runs())} runs to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
