#!/usr/bin/env python
"""Floating-point operations per jet on real test jets, as in the paper's comparison with Benyas et al.

Each of the first 300 test jets is passed alone, cut to its own particles, in float32 on the
CPU. torch's FlopCounterMode counts matrix products and attention; elementwise operations
are not counted. A model that calls a matrix operation the counter cannot see is refused.

    python scripts/count_flops.py --out results/flops.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.utils.flop_counter as fc
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode, register_flop_formula, sdpa_flop_count

from jetkd.data import JetBatch, TopTaggingSplit, particle_mask
from jetkd.models import build

MODELS = ["benyas_deepsets", "benyas_deepsets_mp",
          *(f"{f}_deep_{s}" for s in ("2k", "20k", "200k") for f in ("transformer", "lgatr_slim")),
          "transformer_200k", "lgatr_slim_200k", "part", "mipart", "lgatr", "lgatr_slim"]


# torch has no formula for the CPU flash-attention kernel that L-GATr-slim calls
@register_flop_formula(torch.ops.aten._scaled_dot_product_flash_attention_for_cpu, get_raw=True)
def _cpu_flash(query, key, value, *args, out_val=None, **kwargs):
    return sdpa_flop_count(query.shape, key.shape, value.shape)


class SeenOps(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.add(func.overloadpacket)
        return func(*args, **(kwargs or {}))


def uncounted(model, batch) -> list[str]:
    with SeenOps() as seen:
        model(batch)
    words = ("attention", "bmm", "mm", "matmul", "conv", "einsum", "linear")
    return sorted(str(op) for op in seen.ops if any(w in str(op) for w in words)
                  and op not in fc.flop_registry)


@torch.no_grad()
def count(name: str, jets: list[JetBatch]) -> dict:
    model = build(name).eval()
    for p in model.parameters():   # the counter's module tracker trips on ParT otherwise
        p.requires_grad_(False)
    if missing := uncounted(model, jets[0]):
        raise SystemExit(f"{name}: the counter cannot see {missing}")
    flops = []
    for jet in jets:
        counter = FlopCounterMode(display=False)
        with counter:
            model(jet)
        flops.append(counter.get_total_flops())
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "flops_mean": float(np.mean(flops)), "flops_median": float(np.median(flops))}


def count_all(models: list[str], data: str, n_jets: int = 300) -> dict:
    p4 = TopTaggingSplit("test", data, n_jets).p4
    n = particle_mask(p4).sum(1)
    jets = [JetBatch(p4[i:i + 1, :k], particle_mask(p4[i:i + 1, :k]), None, None)
            for i, k in enumerate(n.tolist())]
    out = {"n_jets": n_jets, "particles_mean": float(n.float().mean()),
           "particles_median": float(n.float().median()), "models": {}}
    for name in models:
        out["models"][name] = m = count(name, jets)
        print(f"{name:24s} {m['parameters']:>9,d} parameters  {m['flops_median'] / 1e6:8.2f} MFLOP "
              "at the median jet", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--jets", type=int, default=300)
    ap.add_argument("--data", default="data/toptagging_full.npz")
    ap.add_argument("--out", default="results/flops.json")
    args = ap.parse_args()
    out = count_all(args.models, args.data, args.jets)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
