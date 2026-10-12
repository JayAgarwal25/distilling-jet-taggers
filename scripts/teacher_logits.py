#!/usr/bin/env python
"""Write a trained jetkd teacher's logits on every split, for distillation.

    python scripts/teacher_logits.py --model lgatr --seed 1001 \
        --checkpoint runs/lgatr__scratch__s1001/best.pt
"""

import argparse

import torch

from jetkd.metrics import binary_metrics
from jetkd.models import MODELS, build
from jetkd.teachers import logits_path, model_logits, save_logits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data/toptagging_full.npz")
    ap.add_argument("--out", default="data/teachers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--splits", default="train,val,test")
    args = ap.parse_args()

    model = build(args.model)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model", state))
    for split in args.splits.split(","):
        z, y = model_logits(model, split, args.data, args.device)
        path = logits_path(args.out, args.model, args.seed, split)
        save_logits(path, z, y)
        m = binary_metrics(z, y)
        print(f"{split}: AUC {m['auc']:.5f} rej50 {m['rej50']:.0f} rej30 {m['rej30']:.0f} -> {path}")


if __name__ == "__main__":
    main()
