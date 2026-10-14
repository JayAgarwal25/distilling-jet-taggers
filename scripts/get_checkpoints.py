#!/usr/bin/env python
"""Download the released checkpoints and teacher logits, check them and unpack them.

    python scripts/get_checkpoints.py                  # students, teachers and teacher logits
    python scripts/get_checkpoints.py teacher_logits   # only what distillation needs

students.tar unpacks to runs/<id>/best.pt, teachers.tar to runs/ and data/external_teachers/ (the
fine-tuned pretrained ParT and OmniLearned-S checkpoints as their code bases saved them), and
teacher_logits.tar to data/teachers/. Every archive is checked against configs/SHA256SUMS.
"""

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path

URL = "https://zenodo.org/records/RECORD/files"   # the release record
ARCHIVES = ("students", "teachers", "teacher_logits")
SUMS = Path(__file__).resolve().parents[1] / "configs" / "SHA256SUMS"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("archives", nargs="*", choices=ARCHIVES, default=list(ARCHIVES))
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out", default=".")
    ap.add_argument("--keep", action="store_true", help="keep the downloaded .tar files")
    args = ap.parse_args()

    sums = dict(line.split()[::-1] for line in SUMS.read_text().splitlines() if line.strip())
    for name in args.archives:
        tar = Path(args.out) / f"{name}.tar"
        if not tar.is_file():
            print(f"downloading {tar.name}", flush=True)
            urllib.request.urlretrieve(f"{args.url}/{tar.name}", tar)
        if sha256(tar) != sums[tar.name]:
            raise SystemExit(f"{tar}: checksum differs from configs/SHA256SUMS")
        with tarfile.open(tar) as tf:
            tf.extractall(args.out, filter="data")
        if not args.keep:
            tar.unlink()
        print(f"{tar.name}: checked and unpacked", flush=True)


if __name__ == "__main__":
    main()
