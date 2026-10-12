#!/usr/bin/env python
"""Import a teacher's predictions from another pipeline, after checking the row order.

Formats (`{split}` in --file is replaced by train, val and test):
  logits       .npz with `logits` and `labels`, one row per jet: any model of your own
  weaver       weaver's prediction ROOT file (score_jet_isTop, score_jet_isQCD, jet_isTop, jet_pt)
  omnilearned  OmniLearned's evaluation .npz (`prediction` as [QCD, top], `pid`)

    python scripts/import_logits.py --name part_pretrained --seed 1001 --format weaver \
        --file predict_output/pred_{split}.root
"""

import argparse

import numpy as np

from jetkd.metrics import binary_metrics
from jetkd.teachers import check_rows, from_probabilities, logits_path, save_logits


def read(fmt: str, path: str):
    """Logits, labels and (where the file has it) jet pT."""
    if fmt == "logits":
        with np.load(path) as f:
            return f["logits"].astype(np.float32), f["labels"], None
    if fmt == "weaver":
        import uproot

        with uproot.open(path) as f:
            a = f[f.keys()[0]].arrays(library="np")
        return (from_probabilities(a["score_jet_isTop"], a["score_jet_isQCD"]),
                a["jet_isTop"], a["jet_pt"])
    with np.load(path) as f:   # omnilearned
        p, pid = f["prediction"], f["pid"]
    return from_probabilities(p[:, 1], p[:, 0]), pid, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--name", required=True, help="teacher name, e.g. part_pretrained")
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--format", required=True, choices=["logits", "weaver", "omnilearned"])
    ap.add_argument("--file", required=True, help="path with {split} in it")
    ap.add_argument("--data", default="data/toptagging_full.npz")
    ap.add_argument("--out", default="data/teachers")
    ap.add_argument("--splits", default="train,val,test")
    args = ap.parse_args()

    for split in args.splits.split(","):
        z, y, jet_pt = read(args.format, args.file.format(split=split))
        check_rows(y, split, args.data, jet_pt)
        path = logits_path(args.out, args.name, args.seed, split)
        save_logits(path, z, y)
        m = binary_metrics(z, y)
        print(f"{split}: AUC {m['auc']:.5f} rej50 {m['rej50']:.0f} rej30 {m['rej30']:.0f} -> {path}")


if __name__ == "__main__":
    main()
