#!/usr/bin/env python
"""Download the released checkpoints and teacher logits and unpack them.

    python scripts/get_checkpoints.py                  # students, teachers and teacher logits
    python scripts/get_checkpoints.py teacher_logits   # only what distillation needs

students.tar unpacks to runs/<id>/best.pt, teachers.tar to runs/ and data/external_teachers/ (the
fine-tuned pretrained ParT and OmniLearned-S checkpoints as their code bases saved them), and
teacher_logits.tar to data/teachers/.
"""

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://github.com/JayAgarwal25/distilling-jet-taggers/releases/download/v1.0.0"   # the release assets
ARCHIVES = ("students", "teachers", "teacher_logits")


def download(url: str, path: Path) -> None:
    part = path.with_suffix(".part")
    with urllib.request.urlopen(url) as r, open(part, "wb") as f:
        size, done = int(r.headers.get("Content-Length", 0)), 0
        while chunk := r.read(1 << 22):
            f.write(chunk)
            done += len(chunk)
            print(f"\r{path.name}: {done / 1e6:.0f} / {size / 1e6:.0f} MB", end="", file=sys.stderr)
    print(file=sys.stderr)
    part.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("archives", nargs="*", help=f"any of {', '.join(ARCHIVES)}; all by default")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out", default=".")
    ap.add_argument("--keep", action="store_true", help="keep the downloaded .tar files")
    args = ap.parse_args()
    if set(args.archives) - set(ARCHIVES):
        ap.error(f"archives are {', '.join(ARCHIVES)}")

    for name in args.archives or ARCHIVES:
        tar = Path(args.out) / f"{name}.tar"
        if not tar.is_file():
            download(f"{args.url}/{tar.name}", tar)
        print(f"{tar.name}: unpacking", flush=True)
        with tarfile.open(tar) as tf:
            tf.extractall(args.out, filter="data")
        if not args.keep:
            tar.unlink()
        print(f"{tar.name}: unpacked", flush=True)


if __name__ == "__main__":
    main()
