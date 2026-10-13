#!/usr/bin/env python
"""Distill your own teacher into your own student.

First save your teacher's logits for each split as .npz with keys `logits` and `labels`,
in the row order of data/toptagging_full.npz, and import them (the import checks the rows):

    python scripts/import_logits.py --name my_teacher --format logits --file my_logits_{split}.npz

Then train any module that maps a `JetBatch` to one logit per jet, here a small Deep Sets:

    python examples/distill_your_own.py --teacher my_teacher
    python examples/distill_your_own.py                       # the same student without a teacher

For a quick check on CPU add `--device cpu --workers 0 --limit 20000 --iterations 200`.
"""

import argparse
import dataclasses

import torch
from torch import nn

from jetkd.data import JetBatch
from jetkd.features import dense_tagging_features
from jetkd.models import MODELS
from jetkd.recipes import Recipe
from jetkd.train import RunConfig, run


class DeepSets(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.particle = nn.Sequential(nn.Linear(7, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.jet = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, batch: JetBatch) -> torch.Tensor:
        x = self.particle(dense_tagging_features(batch.p4, batch.mask))
        x = (x * batch.mask.unsqueeze(-1)).sum(1)
        return self.jet(x).squeeze(-1)


RECIPE = Recipe(name="deepsets", source="examples/distill_your_own.py", optimizer="AdamW", lr=1e-3,
                weight_decay=0.01, scheduler="cosine", batch_size=256, iterations=50_000,
                validate_every=2_500)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--teacher", help="name given to scripts/import_logits.py")
    ap.add_argument("--teacher-seed", type=int, default=1001)
    ap.add_argument("--teachers", default="data/teachers")
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--data", default="data/toptagging_full.npz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--iterations", type=int)
    args = ap.parse_args()

    MODELS["deepsets"] = DeepSets
    recipe = RECIPE
    if args.iterations:
        recipe = dataclasses.replace(recipe, iterations=args.iterations,
                                     validate_every=min(recipe.validate_every, args.iterations))
    logits = f"{args.teachers}/{args.teacher}_s{args.teacher_seed}_train.npz" if args.teacher else None
    run(RunConfig(model="deepsets", seed=args.seed, recipe=recipe, data=args.data,
                  teacher=args.teacher, teacher_seed=args.teacher_seed if args.teacher else None,
                  teacher_logits=logits, device=args.device, workers=args.workers, limit=args.limit))


if __name__ == "__main__":
    main()
