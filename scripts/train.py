#!/usr/bin/env python
"""Train one model, from scratch or distilled from a teacher's logits.

    python scripts/train.py --run part_200k__kd_lgatr_t1002__s1002
    python scripts/train.py --model part_200k --seed 1001 --teacher lgatr --teacher-seed 1001

`--run` takes any id from configs/runs.yaml. Teacher logits are read from
<teachers>/<teacher>_s<teacher seed>_train.npz unless --teacher-logits is given.
"""

from __future__ import annotations

import argparse
import dataclasses
import re

from jetkd.models import MODELS
from jetkd.recipes import recipe_for
from jetkd.train import RunConfig, run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", help="a run id from configs/runs.yaml")
    ap.add_argument("--model", choices=sorted(MODELS))
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--teacher")
    ap.add_argument("--teacher-seed", type=int, default=1001)
    ap.add_argument("--teacher-logits", help="train-split teacher logits (.npz)")
    ap.add_argument("--teachers", default="data/teachers", help="directory of teacher logits")
    ap.add_argument("--data", default="data/toptagging_full.npz")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="use only the first rows of each split")
    ap.add_argument("--iterations", type=int, help="shorten the recipe, for a quick test")
    args = ap.parse_args()

    if args.run:
        m = re.fullmatch(r"(.+)__(scratch|kd_(.+)_t(\d+))__s(\d+)", args.run)
        if not m:
            ap.error(f"not a run id: {args.run}")
        args.model, args.seed = m[1], int(m[5])
        args.teacher, args.teacher_seed = (m[3], int(m[4])) if m[3] else (None, 1001)
    if args.model is None:
        ap.error("give --run or --model")

    recipe = recipe_for(args.model)
    if args.iterations:
        recipe = dataclasses.replace(recipe, iterations=args.iterations,
                                     validate_every=min(recipe.validate_every, args.iterations))
    logits = args.teacher_logits
    if args.teacher and not logits:
        logits = f"{args.teachers}/{args.teacher}_s{args.teacher_seed}_train.npz"
    run(RunConfig(
        model=args.model, seed=args.seed, recipe=recipe, data=args.data, out=args.out,
        teacher=args.teacher, teacher_seed=args.teacher_seed if args.teacher else None,
        teacher_logits=logits, device=args.device, workers=args.workers, limit=args.limit))


if __name__ == "__main__":
    main()
