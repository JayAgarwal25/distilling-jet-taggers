#!/usr/bin/env python
"""Reproduce a table of the paper.

    python scripts/reproduce.py matrix_auc --print-jobs   # every training command behind it
    python scripts/reproduce.py matrix_auc                # rebuild it from the trained models

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
from make_tables import NEEDS_FLOPS, TABLES, Results

from jetkd.data import TopTaggingSplit
from jetkd.models import build
from jetkd.recipes import recipe_for
from jetkd.study import SEEDS, TEACHERS, TRAINED_TEACHERS, paper_runs
from jetkd.summary import summarize
from jetkd.teachers import logits_path, model_logits, save_logits

# tables that print teacher metrics, from all three runs of every teacher
WITH_TEACHERS = ("teachers", "matrix_auc", "operations")


def table_runs(table: str):
    return [r for r in paper_runs() if table in r.tables]


def teachers_needed(table: str) -> list[tuple[str, int]]:
    if table in WITH_TEACHERS:
        return [(t, s) for t in TEACHERS for s in SEEDS]
    return sorted({(r.teacher, r.teacher_seed) for r in table_runs(table) if r.teacher})


def jobs(table: str) -> list[str]:
    distilled_from = {(r.teacher, r.teacher_seed) for r in paper_runs() if r.teacher}
    lines = []
    for t, s in teachers_needed(table):
        if t in TRAINED_TEACHERS:
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


def rebuild(table: str, args) -> str:
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
    for t, s in teachers_needed(table) if table in WITH_TEACHERS else []:
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

    flops = None
    if table in NEEDS_FLOPS:
        path = out / "flops.json"
        if not path.is_file():
            path.write_text(json.dumps(count_all(COUNTED, args.data), indent=1))
        flops = json.loads(path.read_text())
    summary = summarize(runs_out, teachers_out, table_runs(table))
    (out / f"summary_{table}.json").write_text(json.dumps(summary, indent=1))
    return TABLES[table](Results(summary, flops))


def body(tex: str) -> list[str]:
    return [line for line in tex.splitlines() if not line.startswith("%")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("table", choices=sorted(TABLES))
    ap.add_argument("--print-jobs", action="store_true", help="print the training commands and stop")
    ap.add_argument("--runs", default="runs", help="run directories with best.pt")
    ap.add_argument("--teachers", default="data/teachers", help="released teacher logits")
    ap.add_argument("--data", default="data/toptagging_full.npz")
    ap.add_argument("--paper", default="paper/tables", help="the paper's tables")
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
    tex = rebuild(args.table, args)
    path = Path(args.out) / "tables" / f"{args.table}.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex)
    diff = list(difflib.unified_diff(body((Path(args.paper) / f"{args.table}.tex").read_text()),
                                     body(tex), "paper", "reproduced", lineterm=""))
    print("\n".join(diff) if diff else f"{path}: identical to the paper's table")
    raise SystemExit(1 if diff else 0)


if __name__ == "__main__":
    main()
