#!/usr/bin/env python
"""Download the top tagging dataset as one file.

The file is the reference sample of Kasieczka et al. (Zenodo 2603256) in the layout
used by L-GATr and LLoCa, served by the Heidelberg ITP.

    python scripts/get_data.py [--out data/toptagging_full.npz]
"""

import argparse
import sys
import urllib.request
from pathlib import Path

URL = "https://www.thphys.uni-heidelberg.de/~plehn/data/toptagging_full.npz"
SIZE = 1_524_658_601


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default="data/toptagging_full.npz")
    args = ap.parse_args()
    out = Path(args.out)
    if out.is_file():
        print(f"{out} is already there")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(".part")
    with urllib.request.urlopen(URL) as r, open(part, "wb") as f:
        done = 0
        while chunk := r.read(1 << 22):
            f.write(chunk)
            done += len(chunk)
            print(f"\r{done / 1e9:.2f} / {SIZE / 1e9:.2f} GB", end="", file=sys.stderr)
    print(file=sys.stderr)
    part.replace(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
