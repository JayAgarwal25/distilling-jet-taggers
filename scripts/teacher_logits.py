#!/usr/bin/env python
"""Write a trained jetkd teacher's logits on every split, for distillation.

    python scripts/teacher_logits.py --model lgatr --seed 1001 \
        --checkpoint runs/lgatr__scratch__s1001/best.pt

An ensemble teacher is the mean of its members' logits, which must be written first:

    python scripts/teacher_logits.py --model lgatr_slim_ens --seed 1001
"""

import argparse

import torch

from jetkd.metrics import binary_metrics
from jetkd.models import MODELS, build
from jetkd.teachers import ENSEMBLES, ensemble_logits, logits_path, model_logits, save_logits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted({*MODELS, *ENSEMBLES}))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--checkpoint", help="not for an ensemble")
    ap.add_argument("--data", default="data/toptagging_full.npz")
    ap.add_argument("--out", default="data/teachers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--batch-size", type=int, default=512,
                    help="on CPU use about 100: without xformers attention builds a dense mask")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if args.model not in ENSEMBLES:
        if not args.checkpoint:
            ap.error("--checkpoint is required")
        model = build(args.model)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model", state))
    for split in args.splits.split(","):
        if args.model in ENSEMBLES:
            z, y = ensemble_logits(args.out, args.model, split)
        else:
            z, y = model_logits(model, split, args.data, args.device, args.batch_size, args.workers)
        path = logits_path(args.out, args.model, args.seed, split)
        save_logits(path, z, y)
        m = binary_metrics(z, y)
        print(f"{split}: AUC {m['auc']:.5f} rej50 {m['rej50']:.0f} rej30 {m['rej30']:.0f} -> {path}")


if __name__ == "__main__":
    main()
