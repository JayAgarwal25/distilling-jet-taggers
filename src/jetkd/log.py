"""Timestamped progress lines, the same for training and the scripts."""

from __future__ import annotations

import time
from pathlib import Path


def log(message: str, file: str | Path | None = None) -> None:
    """Print a timestamped line and, given a file, append it there."""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    if file is not None:
        with open(file, "a") as f:
            f.write(line + "\n")


def duration(seconds: float) -> str:
    """Seconds as `2h 05m`, `7m` or `12s`."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m = round(seconds / 60)
    return f"{m // 60}h {m % 60:02d}m" if m >= 60 else f"{m}m"
