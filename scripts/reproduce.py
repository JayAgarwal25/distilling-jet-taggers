#!/usr/bin/env python
"""Reproduce a table of the paper, or all tables and figures.

    python scripts/reproduce.py matrix_auc --print-jobs   # every training command behind it
    python scripts/reproduce.py matrix_auc                # rebuild it from the trained models
    python scripts/reproduce.py all                       # every table and figure

The first form lists the commands that train the table's teachers, write their logits and
train its students. The second evaluates each model's `runs/<id>/best.pt` on the test set
(the released checkpoints, or your own after retraining), builds the table from these
predictions and compares it with the paper's table in paper/tables/. Released checkpoints
on a GPU give the paper's digits; retrained models agree within the seed spread.
Pretrained ParT and OmniLearned-S enter through their test logits in data/teachers/.
"""

import argparse
import difflib
import json
from pathlib import Path

import numpy as np
import torch
from count_flops import MODELS as COUNTED
from count_flops import count_all
from make_figures import FIGURES, figures
from make_figures import NEEDS_TIMINGS as FIGURES_NEED_TIMINGS
from make_tables import NEEDS_ANALYSIS, NEEDS_TIMINGS, TABLES, Results

from jetkd.analysis import analyze
from jetkd.data import TopTaggingSplit
from jetkd.models import build
from jetkd.recipes import recipe_for
from jetkd.study import SEEDS, TEACHERS, TRAINED_TEACHERS, paper_runs
from jetkd.summary import summarize
from jetkd.teachers import ENSEMBLES, logits_path, model_logits, save_logits

# tables that print teacher metrics, from all three runs of every teacher
WITH_TEACHERS = ("teachers", "teacher_ensembles", *NEEDS_TIMINGS)


def table_runs(table: str):
    return [r for r in paper_runs() if table == "all" or table in r.tables]


def teachers_needed(table: str) -> list[tuple[str, int]]:
    if table in (*WITH_TEACHERS, "all"):
        return [(t, s) for t in TEACHERS for s in SEEDS]
    return sorted({(r.teacher, r.teacher_seed) for r in table_runs(table) if r.teacher})


def jobs(table: str) -> list[str]:
    distilled_from = {(r.teacher, r.teacher_seed) for r in paper_runs() if r.teacher}
    distilled_from |= {(m, s) for t in ENSEMBLES for m, seeds in [ENSEMBLES[t]] for s in seeds}
    needed = set(teachers_needed(table))
    needed |= {(m, s) for t, _ in needed if t in ENSEMBLES for m, seeds in [ENSEMBLES[t]] for s in seeds}
    lines = []
    for t, s in sorted(needed, key=lambda ts: (ts[0] in ENSEMBLES, ts)):   # ensembles after their members
        if t in ENSEMBLES:
            lines.append(f"python scripts/teacher_logits.py --model {t} --seed {s}")
        elif t in TRAINED_TEACHERS:
            splits = "" if (t, s) in distilled_from else " --splits test"   # teacher table only
            lines += [f"python scripts/train.py --run {t}__scratch__s{s}",
                      f"python scripts/teacher_logits.py --model {t} --seed {s} "
                      f"--checkpoint runs/{t}__scratch__s{s}/best.pt{splits}"]
        else:
            lines.append(f"# {t} run {s}: train it and import its logits as in teachers/README.md")
    lines += [f"python scripts/train.py --run {r.id}" for r in table_runs(table) if r.student not in TEACHERS]
    return lines + [f"python scripts/reproduce.py {table}"]


def test_logits(model: str, checkpoint: Path, test, args) -> tuple[np.ndarray, np.ndarray]:
    net = build(model)
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
    # the batch size of evaluation during training
    batch_size = args.batch_size or max(recipe_for(model).batch_size, 512)
    return model_logits(net, "test", test, args.device, batch_size, args.workers)


def evaluate(table: str, args) -> tuple[Path, Path]:
    """Test logits of every model behind `table`, under args.out."""
    out = Path(args.out)
    runs_out, teachers_out = out / "runs", out / "teachers"
    teachers_out.mkdir(parents=True, exist_ok=True)
    test = None
    for r in table_runs(table):
        path = runs_out / r.id / "predictions_test.npz"
        if r.student in TEACHERS or path.is_file():
            continue
        test = test or TopTaggingSplit("test", args.data, args.limit)
        z, y = test_logits(r.student, Path(args.runs) / r.id / "best.pt", test, args)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, logits=z, labels=y)
        print(f"{r.id}: done", flush=True)
    for t, s in teachers_needed(table) if table in (*WITH_TEACHERS, *NEEDS_ANALYSIS, "all") else []:
        path = logits_path(teachers_out, t, s, "test")
        if path.is_file():
            continue
        if t in TRAINED_TEACHERS:
            test = test or TopTaggingSplit("test", args.data, args.limit)
            z, y = test_logits(t, Path(args.runs) / f"{t}__scratch__s{s}" / "best.pt", test, args)
        else:
            with np.load(logits_path(args.teachers, t, s, "test")) as f:
                z, y = f["logits"][: args.limit], f["labels"][: args.limit]
        save_logits(path, z, y)
        print(f"{t} run {s}: done", flush=True)
    return runs_out, teachers_out


def results(table: str, args) -> tuple[dict, dict | None]:
    """Summary and, where needed, analysis of the models behind `table`, written as JSON."""
    runs_out, teachers_out = evaluate(table, args)
    summary = summarize(runs_out, teachers_out, table_runs(table))
    analysis = analyze(runs_out, teachers_out) if table in (*NEEDS_ANALYSIS, "all") else None
    out = Path(args.out)
    (out / f"summary_{table}.json").write_text(json.dumps(summary, indent=1))
    if analysis:
        (out / f"analysis_{table}.json").write_text(json.dumps(analysis, indent=1))
    return summary, analysis


def compare(table: str, tex: str, args) -> bool:
    """Write the reproduced table and print its difference to the paper's."""
    path = Path(args.out) / "tables" / f"{table}.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex)
    diff = list(difflib.unified_diff(body((Path(args.paper) / f"{table}.tex").read_text()),
                                     body(tex), "paper", "reproduced", lineterm=""))
    print("\n".join(diff) if diff else f"{path}: identical to the paper's table")
    return not diff


def body(tex: str) -> list[str]:
    return [line for line in tex.splitlines() if not line.startswith("%")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("table", choices=[*sorted(TABLES), "all"])
    ap.add_argument("--print-jobs", action="store_true", help="print the training commands and stop")
    ap.add_argument("--runs", default="runs", help="run directories with best.pt")
    ap.add_argument("--teachers", default="data/teachers", help="released teacher logits")
    ap.add_argument("--data", default="data/toptagging_full.npz")
    ap.add_argument("--paper", default="paper/tables", help="the paper's tables")
    ap.add_argument("--benchmark", default="results/benchmark.json",
                    help="timings for the cost tables: yours from scripts/benchmark.py, or ours, "
                         "paper/benchmark.json")
    ap.add_argument("--out", default="reproduced")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int,
                    help="default: as in training; on CPU use about 100, since packed attention "
                         "without xformers builds a dense mask over the batch")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="first test jets only, for a quick check")
    args = ap.parse_args()

    if args.print_jobs:
        print("\n".join(jobs(args.table)))
        return
    torch.set_float32_matmul_precision("highest")   # as in training
    names = sorted(TABLES) if args.table == "all" else [args.table]
    timed = Path(args.benchmark).is_file()
    if not timed and set(names) & set(NEEDS_TIMINGS):
        if args.table != "all":
            ap.error(f"{args.table} needs {args.benchmark}: run scripts/benchmark.py or pass --benchmark")
        names = [t for t in names if t not in NEEDS_TIMINGS]
        print(f"no {args.benchmark}: the cost tables and figures are left out")
    summary, analysis = results(args.table, args)
    benchmark = flops = None
    if timed and set(names) & set(NEEDS_TIMINGS):
        benchmark = json.loads(Path(args.benchmark).read_text())
        path = Path(args.out) / "flops.json"
        if not path.is_file():
            path.write_text(json.dumps(count_all(COUNTED, args.data), indent=1))
        flops = json.loads(path.read_text())
    r = Results(summary, analysis, benchmark, flops)
    same = [compare(t, TABLES[t](r), args) for t in names]
    if args.table == "all":
        paths = figures(summary, analysis, Path(args.out) / "figures",
                        [f for f in FIGURES if benchmark or f not in FIGURES_NEED_TIMINGS], benchmark, flops)
        print(f"{len(paths)} figures in {Path(args.out) / 'figures'}")
    raise SystemExit(0 if all(same) else 1)


if __name__ == "__main__":
    main()
