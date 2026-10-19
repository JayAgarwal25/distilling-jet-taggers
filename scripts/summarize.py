#!/usr/bin/env python
"""Summarize finished runs as JSON: test metrics per run, per cell and per teacher, and the
analyses of what each student takes from its teacher (jetkd.analysis).

    python scripts/summarize.py --runs runs --teachers data/teachers --out results/summary.json
"""

import argparse
import json
from pathlib import Path

from jetkd.analysis import analyze
from jetkd.summary import summarize


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--teachers", default="data/teachers")
    ap.add_argument("--out", default="results/summary.json")
    ap.add_argument("--analysis", default="results/analysis.json")
    args = ap.parse_args()
    out = summarize(args.runs, args.teachers)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"{len(out['runs'])} runs, {len(out['cells'])} cells, {len(out['teachers'])} teachers "
          f"-> {args.out}")
    Path(args.analysis).write_text(json.dumps(analyze(args.runs, args.teachers), indent=1))
    print(f"analysis -> {args.analysis}")


if __name__ == "__main__":
    main()
