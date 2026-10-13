#!/usr/bin/env python
"""Tag jets with a trained model: build it, load its checkpoint and call it on a batch.

    python examples/tag_jets.py --model transformer_200k \
        --checkpoint runs/transformer_200k__kd_part_pretrained_t1001__s1001/best.pt

A model takes a `JetBatch` of zero-padded four-momenta (E, px, py, pz) and returns one
logit per jet; the sigmoid of the logit is the top-quark probability. On CPU, keep batches
small: without xformers, attention over a batch of packed jets builds a dense mask.
"""

import argparse

import numpy as np
import torch

from jetkd.data import JetBatch, particle_mask
from jetkd.metrics import binary_metrics
from jetkd.models import build


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data/toptagging_full.npz")
    ap.add_argument("--jets", type=int, default=2000, help="the first test jets")
    ap.add_argument("--batch-size", type=int, default=100)
    args = ap.parse_args()

    model = build(args.model)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"])
    model.eval()

    with np.load(args.data) as f:
        p4 = torch.from_numpy(f["kinematics_test"][: args.jets].astype(np.float32))
        labels = f["labels_test"][: args.jets]
    logits = []
    with torch.no_grad():
        for x in p4.split(args.batch_size):
            batch = JetBatch(x, particle_mask(x), torch.zeros(len(x)), torch.arange(len(x)))
            logits.append(model(batch).float())
    logits = torch.cat(logits)

    print("top probability of the first jets:", torch.sigmoid(logits[:5]).numpy().round(4))
    print(f"AUC on {len(p4)} test jets: {binary_metrics(logits.numpy(), labels)['auc']:.4f}")


if __name__ == "__main__":
    main()
