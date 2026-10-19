#!/usr/bin/env python
"""Time the paper's models on your hardware, as in the paper's inference cost tables.

    python scripts/benchmark.py --device cuda                 # batch of 512 jets on the GPU
    python scripts/benchmark.py --device cpu --core 2         # one jet on one pinned CPU core

Every model runs in every setting of `jetkd.benchmark.settings`, on test jets and on random
50-particle jets, three times, each in a fresh process; repeats alternate the order. Measurements
go to results/benchmark/<device>/, a finished one is skipped, and results/benchmark.json collects
them for scripts/make_tables.py and scripts/make_figures.py. Timings depend on the hardware; orderings
and rough ratios carry over to other machines. The paper's own measurements are in paper/benchmark.json.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from jetkd.benchmark import REGIMES, environment, measure, settings, summarize

# the models of the paper's cost tables
MODELS = ["part", "mipart", "lgatr", "lgatr_slim", "omnilearned_s", "part_pretrained",
          "transformer_200k", "lorentznet_200k", "part_200k", "mipart_200k", "lgatr_slim_200k",
          *(f"{f}_deep_{s}" for f in ("transformer", "lgatr_slim") for s in ("200k", "20k", "2k")), "deepsets_mp"]


def path(out: Path, device: str, model: str, regime: str, setting: str, repeat: int) -> Path:
    return out / device / model / f"{regime}__{setting.replace(',', '_')}__r{repeat}.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--device", choices=("cuda", "cpu"), required=True)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=REGIMES)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--core", type=int, help="CPU core to pin every measurement to")
    ap.add_argument("--data", default="data/toptagging_full.npz")
    ap.add_argument("--out", default="results/benchmark")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per measurement")
    ap.add_argument("--one", help=argparse.SUPPRESS)   # model,regime,setting: one measurement, here
    args = ap.parse_args()
    out = Path(args.out)

    if args.one:
        model, regime, setting = args.one.split(":")
        print(json.dumps({**measure(model, args.device, regime, setting, args.data), "environment": environment()}))
        return

    env = dict(os.environ)
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        if args.core is not None:
            os.sched_setaffinity(0, {args.core})   # inherited by every measurement
    jobs = [(m, r, s) for m in args.models for r in args.regimes for s in settings(m)]
    for k in range(args.repeats):
        for model, regime, setting in jobs if k % 2 == 0 else jobs[::-1]:
            dst = path(out, args.device, model, regime, setting, k)
            if dst.is_file():
                continue
            cmd = [sys.executable, __file__, "--device", args.device, "--data", args.data,
                   "--one", f"{model}:{regime}:{setting}"]
            try:
                p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.timeout)
                result = json.loads(p.stdout.splitlines()[-1]) if p.returncode == 0 else None
                error = None if result else (p.stderr.strip().splitlines() or ["failed"])[-1]
            except subprocess.TimeoutExpired:
                result, error = None, f"no result within {args.timeout} s"
            if result is None:
                result = {"model": model, "device": args.device, "regime": regime, "setting": setting, "error": error}
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps({**result, "repeat": k}, indent=1))
            print(f"{model} {regime} {setting} run {k}: "
                  + (f"{result['median_ms']:.3f} ms" if "median_ms" in result else error), flush=True)

    measurements = [json.loads(f.read_text()) for f in sorted(out.glob("*/*/*.json"))]
    record = Path(f"{out}.json")
    first = next((m for m in measurements if "environment" in m), {})
    record.write_text(json.dumps({"environment": first.get("environment"), "rows": summarize(measurements)},
                                 indent=1))
    print(f"{len(measurements)} measurements -> {record}")


if __name__ == "__main__":
    main()
