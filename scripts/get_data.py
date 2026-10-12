#!/usr/bin/env python
"""Download the top tagging dataset as one file and check it.

The file is the reference sample of Kasieczka et al. (Zenodo 2603256) in the layout
used by L-GATr and LLoCa, served by the Heidelberg ITP. Its sha256 is the one every
run in the paper recorded.

    python scripts/get_data.py [--out data/toptagging_full.npz]
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://www.thphys.uni-heidelberg.de/~plehn/data/toptagging_full.npz"
SIZE = 1_524_658_601
SHA256 = "5f1470fd2e1f2342cd45618e7e6b14958e3e9d6d50b2e1835759fb3347d16de4"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default="data/toptagging_full.npz")
    args = ap.parse_args()
    out = Path(args.out)
    if out.is_file() and sha256(out) == SHA256:
        print(f"{out} is already there and checks out")
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
    if part.stat().st_size != SIZE or sha256(part) != SHA256:
        sys.exit(f"{part} does not match the expected size and sha256; not using it")
    part.replace(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
