"""Inference cost of a model: time per batch on the GPU and per jet on one CPU core.

A setting names the particle lists (packed or padded), compilation (eager or compiled) and the
attention (built-in or xformers). Compiled means the network is compiled and the inputs are built eager,
as tagging-guide's `compile` option does. One measurement times a model in one setting on one kind of jets (regime):
100 warm-up and 100 timed calls of the whole forward, from four-momenta in CPU memory to logits.
`summarize` turns repeated measurements into one row per model, device and regime: every setting with
its runs, and the fastest setting by the median of its runs.
"""

from __future__ import annotations

import platform
import time
from collections import defaultdict
from functools import partial
from importlib import metadata

import numpy as np
import torch

from jetkd.data import JetBatch, TopTaggingSplit, particle_mask
from jetkd.models import MODELS, REFERENCE_MODELS, build

REGIMES = ("real", "rnd50")   # test jets, or 50 particles with random momenta (arXiv:2608.02735)
BATCH_SIZE = {"cuda": 512, "cpu": 1}


def lists(model: str) -> tuple[str, ...]:
    entry = {**MODELS, **REFERENCE_MODELS}[model]
    cls = entry.func if isinstance(entry, partial) else entry
    if hasattr(cls, "padded"):
        return ("packed", "padded")
    return ("packed",) if model.startswith("lorentznet") else ("padded",)


def settings(model: str) -> list[str]:
    """Every setting of a model: its particle lists, eager and compiled."""
    attention = "xformers" if len(lists(model)) == 2 else "builtin"
    return [f"{p},{c},{attention}" for p in lists(model) for c in ("eager", "compiled")]


def compile_network(model: torch.nn.Module) -> None:
    """Compile the network of a model in place; the inputs stay eager."""
    if hasattr(model, "net"):
        model.net.forward = torch.compile(model.net.forward, dynamic=True)
    else:   # ParT 205k
        model.body = torch.compile(model.body, dynamic=True)


def random_momenta(n: int, particles: int = 50, seed: int = 42) -> torch.Tensor:
    """Massive particles with random momenta, as tagging-guide's cost estimate draws them."""
    g = torch.Generator().manual_seed(seed)
    mass = torch.randn(n, particles, 1, generator=g, dtype=torch.float64).exp()
    p3 = torch.randn(n, particles, 3, generator=g, dtype=torch.float64)
    return torch.cat([(mass**2 + p3.square().sum(-1, keepdim=True)).sqrt(), p3], dim=-1)


def batches(regime: str, batch_size: int, data: str) -> list[JetBatch]:
    """Four batches of test jets, or one batch of random jets, padded to the dataset's 200 particles."""
    if regime == "real":
        p4 = TopTaggingSplit("test", data, 4 * batch_size).p4
    else:
        p4 = torch.zeros(batch_size, 200, 4)
        p4[:, :50] = random_momenta(batch_size).float()
    out = []
    for i in range(0, len(p4), batch_size):
        p = p4[i:i + batch_size]
        out.append(JetBatch(p, particle_mask(p), torch.zeros(len(p)), torch.arange(len(p))))
    return out


def environment() -> dict:
    env = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(), "cpu": _cpu_name()}
    for package in ("lgatr", "lloca", "xformers", "omnilearned"):
        try:
            env[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env.update(gpu=p.name, gpu_memory_gb=round(p.total_memory / 2**30, 1))
    return env


def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            return next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        return platform.processor()


@torch.inference_mode()
def measure(model: str, device: str, regime: str, setting: str, data: str,
            warmup: int = 100, steps: int = 100) -> dict:
    """One measurement; run it in a fresh process, so compilation and memory never carry over."""
    padded, compiled, _ = setting.split(",")
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")   # 32-bit as in training, no TF32
    if device == "cpu":
        torch.set_num_threads(1)
    else:
        torch.backends.cuda.enable_cudnn_sdp(False)   # as tagging-guide's benchmark
    net = build(model).eval().to(device)
    if hasattr(net, "padded"):
        net.padded = padded == "padded"
    if compiled == "compiled":
        compile_network(net)
    bs = BATCH_SIZE[device]
    pool = batches(regime, bs, data)

    def timed(batch: JetBatch) -> float:
        if device == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            net(batch.to(device))
            end.record()
            end.synchronize()
            return start.elapsed_time(end)
        t0 = time.perf_counter_ns()
        net(batch.to(device))
        return (time.perf_counter_ns() - t0) * 1e-6

    times = []
    for i in range(warmup + steps):
        if i == 1 and device == "cuda":   # the first call compiles
            torch.cuda.reset_peak_memory_stats()
        t = timed(pool[i % len(pool)])
        if i >= warmup:
            times.append(t)
    q20, median, q80 = np.quantile(times, [0.2, 0.5, 0.8])
    out = {"model": model, "device": device, "regime": regime, "batch_size": bs, "setting": setting,
           "median_ms": float(median), "q20_ms": float(q20), "q80_ms": float(q80),
           "parameters": sum(p.numel() for p in net.parameters())}
    if device == "cuda":
        out["peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return out


def summarize(measurements: list[dict]) -> list[dict]:
    """One row per model, device and regime: every setting with the medians of its runs, and the
    fastest setting by the median over its runs. Failed settings keep their error."""
    groups = defaultdict(lambda: defaultdict(list))
    for m in measurements:
        groups[(m["model"], m["device"], m["regime"])][m["setting"]].append(m)
    rows = []
    for (model, device, regime), by_setting in groups.items():
        entries = {}
        for setting, runs in by_setting.items():
            ok = [r for r in runs if "error" not in r]
            if not ok:
                entries[setting] = {"error": runs[0]["error"]}
                continue
            e = {"runs_ms": [r["median_ms"] for r in ok], "median_ms": float(np.median([r["median_ms"] for r in ok]))}
            if device == "cuda":
                e["peak_mb"] = float(np.median([r["peak_mb"] for r in ok]))
            entries[setting] = e
        ran = {k: v for k, v in entries.items() if "median_ms" in v}
        if not ran:
            continue
        best = min(ran, key=lambda k: ran[k]["median_ms"])
        rows.append({"model": model, "device": device, "regime": regime, "batch_size": BATCH_SIZE[device],
                     **({"threads": 1} if device == "cpu" else {}), "settings": entries, "selected": best,
                     **ran[best]})
    return rows
