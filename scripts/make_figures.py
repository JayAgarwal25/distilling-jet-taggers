#!/usr/bin/env python
"""The paper's figures from results/summary.json and results/analysis.json (scripts/summarize.py),
results/benchmark.json (scripts/benchmark.py) and results/flops.json (scripts/count_flops.py).

    python scripts/make_figures.py --out figures            # all of them
    python scripts/make_figures.py strength analysis       # some of them

Each figure is written as PDF and PNG.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.legend_handler  # noqa: E402
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.transforms  # noqa: E402
import numpy as np  # noqa: E402
from make_tables import Results, depth_models, params, width_models  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from jetkd.study import TEACHERS  # noqa: E402

BASE = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 13,
    "axes.labelsize": 14,
    "legend.fontsize": 11.5,
    "axes.linewidth": 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
}
SMALL = {"font.size": 9, "axes.labelsize": 9.5, "legend.fontsize": 8}
# the matrix students in the tables' order, one colour and marker each; * Lorentz-equivariant
STUDENTS = [
    ("transformer_200k", "Transformer", "#2a78d6", "o"),
    ("lorentznet_200k", "LorentzNet*", "#eb6834", "v"),
    ("part_200k", "ParT", "#1baf7a", "s"),
    ("mipart_200k", "MIParT", "#eda100", "^"),
    ("lgatr_slim_200k", "L-GATr-slim*", "#e87ba4", "D"),
]
# dagger: pretrained
NAMES = {"part": "ParT", "mipart": "MIParT", "lgatr": "L-GATr*", "lgatr_slim": "L-GATr-slim*",
         "omnilearned_s": "OmniLearned-S†", "part_pretrained": "ParT-f.t.†"}
# the scaling studies: label, colour, marker, base model of the depth study, family of the width study
FAMILIES = [("Transformer", "#2a78d6", "o", "transformer_200k", "transformer"),
            ("L-GATr-slim*", "#e87ba4", "D", "lgatr_slim_200k", "lgatr_slim")]
# teachers of the scaling studies, weakest first; curves take the student's colour, lighter for a
# weaker teacher, one marker per teacher
STUDY_TEACHERS = [("lgatr", "P"), ("omnilearned_s", "X"), ("part_pretrained", "s")]
STUDY_SHADES = {"#2a78d6": ("#8ab8ee", "#2a78d6", "#123e75"), "#e87ba4": ("#f0a3c3", "#e87ba4", "#9e2f5d")}
DISTILLED = dict(ls="-", lw=1.2)
NO_TEACHER = dict(ls=":", lw=1.2, mfc="white")


def save(fig, path: Path, tight: bool = False) -> list[Path]:
    kw = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(path, **kw)
    fig.savefig(path.with_suffix(".png"), dpi=200, **kw)
    plt.close(fig)
    return [path]


def by_auc(r: Results) -> list[str]:
    return sorted(TEACHERS, key=lambda t: r.teacher(t)["auc"])


def mode_handles() -> list:
    return [Line2D([], [], color="#0b0b0b", marker="o", ms=4.5, label="distilled", **DISTILLED),
            Line2D([], [], color="#0b0b0b", marker="o", ms=4.5, label="no teacher", **NO_TEACHER)]


def legends(fig, models: list, y: float = 1.0) -> None:
    """Models in their colours on the first row, distilled and no teacher in black on the second."""
    if models:
        fig.legend(handles=[Line2D([], [], color=c, marker=mk, ms=5, lw=0, label=lb) for lb, c, mk in models],
                   loc="upper center", ncol=len(models), frameon=False, fontsize=8, handlelength=1.0,
                   columnspacing=1.5, bbox_to_anchor=(0.5, y))
        y -= 0.19 / fig.get_figheight()   # one legend row, whatever the figure height
    fig.legend(handles=mode_handles(), loc="upper center", ncol=2, frameon=False, fontsize=8,
               handlelength=2.4, columnspacing=2.0, bbox_to_anchor=(0.5, y))


# ---------------------------------------------------------------- Sections 4 and 5

def rej_by_teacher(r: Results, a: dict, out: Path) -> list[Path]:
    """Background rejection of every student without a teacher and from each teacher, with the
    teacher's own rejection as a black bar on a grey band of its standard deviation over runs."""
    groups = [None, *TEACHERS]
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 7.0), sharex=True)
    for ax, m, eps in ((axes[0], "rej50", "0.5"), (axes[1], "rej30", "0.3")):
        for g, t in enumerate(groups):
            if g > 0:
                ax.axvline(g - 0.5, color="#DDDDDD", lw=0.8, zorder=0)
            if t is not None:
                tv, ts = r.teacher(t)[m], r.teacher(t)[f"{m}_sd"]
                ax.fill_between([g - 0.42, g + 0.42], tv - ts, tv + ts, color="#9a9a9a", alpha=0.22, lw=0, zorder=1)
                ax.plot([g - 0.42, g + 0.42], [tv, tv], color="#0b0b0b", lw=1.8, zorder=2, solid_capstyle="butt")
            for i, (s, _, color, marker) in enumerate(STUDENTS):
                c, x = r.cell(s, t), g + (i - 2) * 0.15
                ax.plot(x, c[m], color=color, marker=marker, ms=5.6, mec="white", mew=0.6, lw=0, zorder=3)
                # bars above the markers, so that no marker hides half of a bar
                ax.errorbar(x, c[m], yerr=c[f"{m}_sd"], color=color, lw=0, elinewidth=1.1, capsize=0, zorder=4)
        ax.set_ylabel(rf"$1/\epsilon_B$ at $\epsilon_S={eps}$")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-0.5, len(groups) - 0.5)
        ax.yaxis.grid(True, color="#e4e4e4", lw=0.6)
        ax.set_axisbelow(True)
    axes[1].set_xticks(range(len(groups)))
    axes[1].set_xticklabels(["No teacher", *NAMES.values()], rotation=25, ha="right", rotation_mode="anchor")
    axes[1].set_xlabel("Teacher")
    handles = [Line2D([], [], color=c, marker=mk, ms=5.6, lw=0) for _, _, c, mk in STUDENTS]
    handles.append((matplotlib.patches.Patch(facecolor="#9a9a9a", alpha=0.22, lw=0),
                    Line2D([], [], color="#0b0b0b", lw=1.8)))
    fig.legend(handles, [lb for _, lb, _, _ in STUDENTS] + ["teacher"],
               handler_map={tuple: matplotlib.legend_handler.HandlerTuple(ndivide=1, pad=0)},
               loc="upper center", ncol=6, frameon=False, fontsize=10.5, handlelength=1.4, columnspacing=1.2,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, out / "rej_by_teacher.pdf")


def strength(r: Results, a: dict, out: Path) -> list[Path]:
    """Gain over the student without a teacher in AUC and in rejection at eps_S = 0.5, every teacher at
    its AUC; a black tick at the mean over the five students."""
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 8})
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.6))
    order = by_auc(r)
    tx = [r.teacher(t)["auc"] for t in order]
    span = max(tx) - min(tx)
    for ax, m, ylabel in ((axes[0], "auc", r"AUC gain ($10^{-4}$)"),
                          (axes[1], "rej50", r"Gain in $1/\epsilon_B$ at $\epsilon_S=0.5$ (%)")):
        for t, x in zip(order, tx, strict=True):
            g = np.array([1e4 * (r.cell(s, t)[m] - r.cell(s, None)[m]) if m == "auc" else
                          100 * (r.cell(s, t)[m] / r.cell(s, None)[m] - 1) for s, *_ in STUDENTS])
            ax.plot([x, x], [g.min(), g.max()], color="#C8C8C8", lw=1.0, solid_capstyle="round", zorder=1)
            for gi, (_, _, color, marker) in zip(g, STUDENTS, strict=True):
                ax.plot(x, gi, marker=marker, ms=5, color=color, mec="white", mew=0.4, lw=0, zorder=2)
            ax.plot(x, g.mean(), marker="_", ms=8, mew=1.8, color="#0b0b0b", zorder=3)
        ax.set_xlim(min(tx) - 0.08 * span, max(tx) + 0.08 * span)
        ax.set_xticks(tx, [NAMES[t] for t in order], rotation=55, ha="right", rotation_mode="anchor", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(ylabel)
        ax.tick_params(labelsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], color=c, marker=mk, ms=5, lw=0, label=lb) for _, lb, c, mk in STUDENTS]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=8, handlelength=1.0,
               columnspacing=1.2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0.06, 1, 0.92), w_pad=1.2)
    x0, x1, y = axes[0].get_position().x0, axes[1].get_position().x1, 0.035   # teacher AUC grows to the right
    fig.add_artist(matplotlib.patches.FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
                                                      arrowstyle="-|>", mutation_scale=10, lw=1.0, color="#0b0b0b"))
    fig.text((x0 + x1) / 2, y, "Teacher AUC", ha="center", va="center", fontsize=9.5,
             bbox=dict(fc="white", ec="none", pad=2.0))
    return save(fig, out / "strength.pdf")


def scaling(r: Results, study: str, m: str, out: Path) -> list[Path]:
    """Rejection against parameters, Transformer (left) and L-GATr-slim (right), from each teacher of the
    scaling studies and without a teacher, with the teachers as grey dashed lines. Width study: three runs
    with standard deviations; depth study: one run, zero blocks left out."""
    eps = {"rej30": "0.3", "rej50": "0.5"}[m]
    plt.rcParams.update(SMALL)
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.2), sharey=True)
    keys = [None] + [k for k, *_ in STUDY_TEACHERS]

    def cells(base: str, family: str) -> list[dict]:
        if study == "depth":
            return [{"params": params(md), **{k: r.run(md, k) for k in keys}}
                    for n, md in depth_models(base) if n]
        return sorted(({"params": params(md), **{k: r.cell(md, k) for k in keys}} for md in width_models(family)),
                      key=lambda v: v["params"])

    allv = [r.teacher(k)[m] for k, *_ in STUDY_TEACHERS]   # one y range for both panels
    for *_, base, family in FAMILIES:
        for v in cells(base, family):
            for k in keys:
                sd = v[k].get(f"{m}_sd", 0) if study == "width" else 0
                allv += [v[k][m] + sd, v[k][m] - sd]
    glo, ghi = min(allv), max(allv)
    for col, (label, scolor, smarker, base, family) in enumerate(FAMILIES):
        ax = axes[col]
        vs = cells(base, family)
        x = np.array([v["params"] for v in vs], dtype=float)
        series = [(None, scolor, smarker, NO_TEACHER)] + [
            (k, shade, mk, DISTILLED) for (k, mk), shade in zip(STUDY_TEACHERS, STUDY_SHADES[scolor], strict=True)]
        for k, color, marker, style in series:
            y = np.array([v[k][m] for v in vs])
            ax.plot(x, y, color=color, marker=marker, ms=4.5, mec=color, zorder=3, **style)
            if study == "width":
                ax.errorbar(x, y, yerr=np.array([v[k][f"{m}_sd"] for v in vs]), color=color, lw=0,
                            elinewidth=0.8, capsize=0, zorder=2)
        lines = []
        for k, *_ in STUDY_TEACHERS:
            tv = r.teacher(k)[m]
            ax.axhline(tv, color="#8a8a8a", lw=0.7, ls=(0, (4, 3)), alpha=0.7, zorder=0)
            lines.append((tv, NAMES[k], "#555555"))
        ax.set_xscale("log")
        ax.set_xlim(x.min() / 1.2, x.max() * 1.2)
        span = ghi - glo
        ax.set_ylim(glo - 0.04 * span, ghi + 0.12 * span)
        # teacher names in the left panel, from the top down: above the line if there is room
        h = 0.085 * (span * 1.16)
        floor = np.inf
        for tv, name, color in (sorted(lines, reverse=True) if col == 0 else []):
            above = tv + 0.012 * span + h < floor
            y = tv + (0.012 if above else -0.012) * span
            floor = y if above else y - h
            ax.text(0.02, y, name, color=color, alpha=0.75, fontsize=7.5, va="bottom" if above else "top",
                    ha="left", transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData))
        ticks = [2e3, 1e4, 1e5] if study == "width" else [2e4, 5e4, 1e5, 2e5, 4e5]
        ticks = [t for t in ticks if x.min() / 1.2 <= t <= x.max() * 1.2]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t / 1e3:.0f}k" for t in ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(True, which="major", color="#e4e4e4", lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Number of parameters")
        if col == 0:
            ax.set_ylabel(rf"$1/\epsilon_B$ at $\epsilon_S={eps}$")
        ax.legend(handles=[Line2D([], [], color=shade, marker=mk, ms=5, lw=1.2, label=NAMES[k])
                           for (k, mk), shade in zip(STUDY_TEACHERS, STUDY_SHADES[scolor], strict=True)],
                  loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=7,
                  handlelength=1.3, handletextpad=0.3, columnspacing=0.7, borderpad=0.1, borderaxespad=0.2)
        ax.set_title(label, fontsize=9.5, pad=16)
    legends(fig, [])
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=0.2)
    return save(fig, out / (f"scaling_{study}.pdf" if m == "rej30" else f"scaling_{study}_{m}.pdf"))


# ---------------------------------------------------------------- Section 6

def kl_vs_size(ax, fid: dict, teacher: str, study: str) -> None:
    """KL divergence to the teacher against student parameters, distilled and without a teacher."""
    for _, color, marker, base, family in FAMILIES:
        models = list(reversed(width_models(family))) if study == "width" else [md for n, md in depth_models(base) if n]
        pts = [(params(md), fid[f"{md}|{teacher}"]) for md in models if f"{md}|{teacher}" in fid]
        x = [p for p, _ in pts]
        for arm, style in (("distilled", DISTILLED), ("scratch", NO_TEACHER)):
            ax.plot(x, [v[arm]["kl"] for _, v in pts], color=color, marker=marker, ms=4.5, mec=color, **style)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks([0.003, 0.01, 0.03, 0.1], ["0.003", "0.01", "0.03", "0.1"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(True, which="major", color="#e4e4e4", lw=0.6)
    ax.set_axisbelow(True)


def confidence(ax, jet: dict, student: str, teacher: str, cls: str, color: str, marker: str) -> None:
    """Student minus teacher probability for the true class, per group of teacher confidence."""
    rows = jet[f"{student}|{teacher}"][cls]
    tp = np.array([b["teacher_p"] for b in rows])
    x = np.arange(1, len(rows) + 1)
    ax.axhline(0, color="#0b0b0b", lw=1.2, zorder=1)
    for k, style in (("distilled_p", DISTILLED), ("scratch_p", NO_TEACHER)):
        ax.plot(x, np.array([b[k] for b in rows]) - tp, color=color, marker=marker, ms=3.8, mec=color, zorder=2,
                **style)
    ax.set_xticks(x, ["low"] + [""] * (len(rows) - 2) + ["high"])
    ax.annotate("teacher", (5.5, 0), xytext=(0, 2.5), textcoords="offset points", ha="center", va="bottom",
                fontsize=7.5, color="#0b0b0b")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, max(hi, 0) + 0.12 * (hi - lo))
    ax.grid(True, which="major", color="#e4e4e4", lw=0.6)
    ax.set_axisbelow(True)


def shared(ax, fid: dict, r: Results, students: list, key: str) -> None:
    """Of the background jets each teacher accepts, the share its students accept too."""
    order = by_auc(r)
    step = 0.16 if len(students) > 2 else 0.22
    for i, t in enumerate(order):
        if i:
            ax.axvline(i - 0.5, color="#E4E4E4", lw=0.8, zorder=0)
        for k, (s, _, color, marker) in enumerate(students):
            x = i + (k - (len(students) - 1) / 2) * step
            ax.plot(x, 100 * fid[f"{s}|{t}"]["scratch"][key], marker=marker, ms=4.5, mfc="white", mec=color,
                    mew=1.0, lw=0, zorder=2)
            ax.plot(x, 100 * fid[f"{s}|{t}"]["distilled"][key], marker=marker, ms=4.5, color=color, lw=0, zorder=3)
    ax.set_xticks(range(len(order)), [NAMES[t] for t in order], rotation=35, ha="right", rotation_mode="anchor",
                  fontsize=8)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.tick_params(axis="x", length=0)
    ax.grid(True, axis="y", color="#e4e4e4", lw=0.6)
    ax.set_axisbelow(True)


def analysis(r: Results, a: dict, out: Path) -> list[Path]:
    """(a) KL divergence to ParT-f.t. in the width study, (b) shared background jets at eps_S = 0.5,
    (c, d) probability of the true class against ParT-f.t.'s, for Transformer and L-GATr-slim."""
    plt.rcParams.update(SMALL)
    fig = plt.figure(figsize=(5.6, 4.9))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.38)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    gs_c = gs[1, :].subgridspec(1, 2, wspace=0.3)
    ax_s, ax_g = fig.add_subplot(gs_c[0]), fig.add_subplot(gs_c[1], sharey=None)
    pair = [STUDENTS[0], STUDENTS[4]]   # Transformer, L-GATr-slim

    kl_vs_size(ax_a, a["fidelity"], "part_pretrained", "width")
    ax_a.set_xlabel("Student parameters")
    ax_a.set_ylabel("KL divergence")
    ax_a.set_title("(a) Width study", fontsize=9.5, loc="left")
    shared(ax_b, a["fidelity"], r, pair, "shared50")
    ax_b.set_ylabel("Shared background jets (%)")
    ax_b.set_title("(b) Mistakes shared with the teacher", fontsize=9.5, loc="left")
    for ax, cls, where in ((ax_s, "signal", 0.88), (ax_g, "background", 0.1)):
        for s, _, color, marker in pair:
            confidence(ax, a["perjet"], s, "part_pretrained", cls, color, marker)
        ax.set_xlabel("Teacher confidence")
        ax.set_title("(c) Signal jets" if cls == "signal" else "(d) Background jets", fontsize=9.5, loc="left")
        ax.text(0.97, where, "student − ParT-f.t.†", transform=ax.transAxes, ha="right", va="center",
                fontsize=8.5, color="#555555")
    ax_s.set_ylabel("Difference in probability\nof the true class")
    for ax in (ax_a, ax_b, ax_s, ax_g):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    legends(fig, [(lb, c, mk) for lb, c, mk, *_ in FAMILIES], y=1.02)
    return save(fig, out / "analysis.pdf", tight=True)


def analysis_size(r: Results, a: dict, out: Path) -> list[Path]:
    """KL divergence to the teacher against student size, width and depth study, three teachers."""
    plt.rcParams.update(SMALL)
    fig, axes = plt.subplots(2, 3, figsize=(5.6, 4.0), sharey=True)
    for row, study in enumerate(("width", "depth")):
        for col, t in enumerate(("lgatr", "omnilearned_s", "part_pretrained")):
            ax = axes[row, col]
            kl_vs_size(ax, a["fidelity"], t, study)
            ax.set_title(f"{study.capitalize()} study, {NAMES[t]}", fontsize=8.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7.5)
            if row == 1:
                ax.set_xlabel("Student parameters")
        axes[row, 0].set_ylabel("KL divergence to the teacher")
    legends(fig, [(lb, c, mk) for lb, c, mk, *_ in FAMILIES])
    fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=0.8, w_pad=0.6)
    return save(fig, out / "analysis_size.pdf")


def analysis_perjet(r: Results, a: dict, out: Path) -> list[Path]:
    """Every matrix student's probability for the true class against its teacher's, one figure per
    teacher of the scaling studies."""
    plt.rcParams.update(SMALL)
    paths = []
    for teacher in ("part_pretrained", "omnilearned_s", "lgatr"):
        fig, axes = plt.subplots(len(STUDENTS), 2, figsize=(5.6, 7.4), sharex=True)
        for row, (s, label, color, marker) in enumerate(STUDENTS):
            for col, cls in enumerate(("signal", "background")):
                ax = axes[row, col]
                confidence(ax, a["perjet"], s, teacher, cls, color, marker)
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(labelsize=7.5)
                if row == 0:
                    ax.set_title(cls.capitalize(), fontsize=9.5)
                if row == len(STUDENTS) - 1:
                    ax.set_xlabel("Teacher confidence")
            axes[row, 0].set_ylabel(label, fontsize=9)
        fig.supylabel(f"Student − {NAMES[teacher]}, difference in probability of the true class", fontsize=9.5)
        legends(fig, [])
        fig.tight_layout(rect=(0.02, 0, 1, 0.975), h_pad=0.5, w_pad=0.8)
        paths += save(fig, out / f"analysis_perjet_{teacher}.pdf")
    return paths


def analysis_shared(r: Results, a: dict, out: Path) -> list[Path]:
    """Shared background jets for all five matrix students, at eps_S = 0.5 and 0.3."""
    plt.rcParams.update(SMALL)
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.1), sharey=True)
    for ax, key, eps in ((axes[0], "shared50", "0.5"), (axes[1], "shared30", "0.3")):
        shared(ax, a["fidelity"], r, STUDENTS, key)
        ax.set_title(rf"$\epsilon_S={eps}$", fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("Shared background jets (%)")
    legends(fig, [(lb, c, mk) for _, lb, c, mk in STUDENTS])
    fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=0.8)
    return save(fig, out / "analysis_shared.pdf")


def analysis_distinctive(r: Results, a: dict, out: Path) -> list[Path]:
    """Of the background jets one teacher accepts and another rejects, the share the student accepts,
    distilled from the first, from the second or without a teacher; means over students and the
    other teachers."""
    order = by_auc(r)
    styles = [("student_A", "distilled from this teacher", dict(color="#2a78d6", mfc="#2a78d6")),
              ("scratch", "no teacher", dict(color="#7a7a7a", mfc="white", mew=1.3)),
              ("student_B", "distilled from another teacher", dict(color="#eb6834", mfc="#eb6834"))]
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9.5})
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.0), sharey=True)
    for ax, e in zip(axes, ("50", "30"), strict=True):
        for row, t in enumerate(order[::-1]):
            means = {k: 100 * np.mean([a["distinctive"][f"{s}|{t}|{b}|{e}"][k] for s, *_ in STUDENTS
                                       for b in order if b != t]) for k, *_ in styles}
            ax.plot([min(means.values()), max(means.values())], [row, row], color="#d0d0d0", lw=1.0, zorder=1)
            for k, _, st in styles:   # no teacher underneath, so an equal value stays visible
                ax.plot(means[k], row, marker="o", ms=6, lw=0, zorder=2 if k == "scratch" else 3, **st)
        ax.set_yticks(range(len(order)), [NAMES[t].rstrip("*†") for t in order[::-1]], fontsize=8)
        ax.set_xlim(0, 70)
        ax.set_title(rf"$\epsilon_S={int(e) / 100}$", fontsize=9.5)
        ax.set_xlabel("Teacher's mistakes repeated (%)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="x", color="#e4e4e4", lw=0.6)
        ax.set_axisbelow(True)
    fig.legend(handles=[Line2D([], [], marker="o", ms=6, lw=0, label=lb, **st) for _, lb, st in styles],
               loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return save(fig, out / "analysis_distinctive.pdf")


def analysis_accepted(r: Results, a: dict, out: Path) -> list[Path]:
    """Background jets Transformer and L-GATr-slim accept at eps_S = 0.5, with and without distillation,
    among the jets each teacher rejects and among those it accepts."""
    pair = [STUDENTS[0], STUDENTS[4]]
    order = by_auc(r)
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9.5})
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.2), sharey=True)
    for ax, (dec, title) in zip(axes, (("rejects", "Jets the teacher rejects"),
                                       ("accepts", "Jets the teacher accepts")), strict=True):
        for s, _, color, marker in pair:
            for arm, style in (("without", NO_TEACHER), ("with", DISTILLED)):
                ax.plot(range(len(order)), [a["accepted"][f"{s}|{t}|50"][f"{arm}_teacher_{dec}"] for t in order],
                        marker=marker, color=color, ms=4.5, **style)
        ax.set_title(title, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color="#e4e4e4", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xticks(range(len(order)), [NAMES[t].rstrip("*†") for t in order], rotation=55, ha="right",
                      rotation_mode="anchor", fontsize=8)
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("Background jets accepted")
    fig.legend(handles=[Line2D([], [], color=c, marker=mk, ms=5, lw=0, label=lb) for _, lb, c, mk in pair],
               loc="upper center", ncol=2, frameon=False, fontsize=8, handlelength=1.0, columnspacing=1.5,
               bbox_to_anchor=(0.5, 1.0))
    fig.legend(handles=[Line2D([], [], color="#0b0b0b", marker="o", ms=4.5, label="with distillation", **DISTILLED),
                        Line2D([], [], color="#0b0b0b", marker="o", ms=4.5, label="without distillation",
                               **NO_TEACHER)],
               loc="upper center", ncol=2, frameon=False, fontsize=8, handlelength=2.4, columnspacing=2.0,
               bbox_to_anchor=(0.5, 1.0 - 0.19 / fig.get_figheight()))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    return save(fig, out / "analysis_accepted.pdf")


# ---------------------------------------------------------------- Section 7

# model families of the cost figures, Deep Sets + GNN of arXiv:2609.26899 in grey
COST_FAMILIES = [
    (["transformer_deep_2k", "transformer_deep_20k", "transformer_deep_200k", "transformer_200k"],
     "Transformer", "#2a78d6", "o"),
    (["lorentznet_200k"], "LorentzNet*", "#eb6834", "v"),
    (["part_200k", "part", "part_pretrained"], "ParT", "#1baf7a", "s"),
    (["mipart_200k", "mipart"], "MIParT", "#eda100", "^"),
    (["lgatr_slim_deep_2k", "lgatr_slim_deep_20k", "lgatr_slim_deep_200k", "lgatr_slim_200k", "lgatr_slim"],
     "L-GATr-slim*", "#e87ba4", "D"),
    (["lgatr"], "L-GATr*", "#008300", "P"),
    (["omnilearned_s"], "OmniLearned-S\u2020", "#6250d6", "X"),
    (["deepsets_mp"], "Deep Sets + GNN", "#7A7A7A", "h"),
]
DEEPSETS_GNN_REJ = {"rej50": 450, "rej30": 1980}   # as published
COST_PERF_SKIP = {"transformer_deep_2k"}   # far below the others in performance
COST_SIZES = {"transformer_deep_20k": "16k", "transformer_deep_200k": "84k", "transformer_200k": "188k",
              "lgatr_slim_deep_2k": "1.8k", "lgatr_slim_deep_20k": "20k", "lgatr_slim_deep_200k": "181k",
              "lgatr_slim_200k": "202k"}


def cost_points(r: Results) -> dict[str, dict]:
    """Parameters, operations per median jet and the timings on real jets of every cost-figure model."""
    out = {}
    for names, *_ in COST_FAMILIES:
        for m in names:
            g, c = r.timings.get((m, "cuda", "real")), r.timings.get((m, "cpu", "real"))
            out[m] = dict(params=r.flops[m]["parameters"] if m == "deepsets_mp" else params(m),
                          flops=r.flops[m]["flops_median"] if m in r.flops else None,
                          gpu=g["batch_size"] / (g["median_ms"] / 1e3) if g else None,
                          mem=g.get("peak_mb") if g else None,
                          cpu=c["median_ms"] if c else None)
    return out


def cost_params(r: Results, a: dict, out: Path) -> list[Path]:
    """Operations per jet, GPU throughput and peak memory at a batch of 512 and CPU latency for one jet
    against parameters, real test jets, fastest 32-bit setting."""
    pts = cost_points(r)
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10.5, "legend.fontsize": 8})
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 5.0), sharex=True)
    panels = [("flops", "Operations per jet (FLOP)"), ("gpu", "GPU throughput (jets/s)"),
              ("mem", "GPU peak memory (MB)"), ("cpu", "CPU latency (ms per jet)")]
    for ax, (key, ylabel) in zip(axes.flat, panels, strict=True):
        for names, _, color, marker in COST_FAMILIES:
            xy = sorted((pts[m]["params"], pts[m][key]) for m in names if pts[m][key] is not None)
            if xy:
                ax.plot(*zip(*xy, strict=True), color=color, marker=marker, ms=5.5, mec="white", mew=0.5, lw=0,
                        zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, which="major", color="#e4e4e4", lw=0.6)
        ax.set_axisbelow(True)
        if key in ("gpu", "cpu"):   # under two decades: label 2 and 5 as well
            ticks = [5e3, 1e4, 2e4, 5e4, 1e5] if key == "gpu" else [2, 5, 10, 20, 50]
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{t / 1e3:.0f}k" for t in ticks] if key == "gpu" else [f"{t:g}" for t in ticks])
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for ax in axes[1]:
        ax.set_xlabel("Number of parameters")
    fig.legend(handles=[Line2D([], [], color=c, marker=mk, ms=5.5, mec="white", mew=0.5, lw=0, label=lb)
                        for _, lb, c, mk in COST_FAMILIES],
               loc="upper center", ncol=4, frameon=False, fontsize=8, handlelength=1.6, columnspacing=1.0,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=0.6, w_pad=1.0)
    return save(fig, out / "cost_params.pdf")


def cost_perf(r: Results, a: dict, out: Path) -> list[Path]:
    """Rejection at eps_S = 0.5 (left) and 0.3 (right) against GPU time per jet, CPU latency and
    operations per jet: students distilled from ParT-f.t. filled, teacher models hollow (ParT-f.t. a
    star), a faint dotted line through the width-study students of each architecture."""
    pts = cost_points(r)
    chains = [["transformer_deep_20k", "transformer_deep_200k", "transformer_200k"],
              ["lgatr_slim_deep_2k", "lgatr_slim_deep_20k", "lgatr_slim_deep_200k"]]
    marks = []   # (model, role, cell, colour, marker)
    for names, _, color, marker in COST_FAMILIES:
        for m in names:
            if m in COST_PERF_SKIP:
                continue
            if m in TEACHERS:
                cell, role = r.teacher(m), "teacher"
            elif m == "deepsets_mp":
                cell, role = DEEPSETS_GNN_REJ, "other"
            else:
                cell, role = r.cell(m, "part_pretrained"), "distilled"
            marks.append((m, role, cell, color, "*" if m == "part_pretrained" else marker))

    def look(role: str, color: str, mk: str) -> dict:
        big = 1.6 if mk == "*" else 1.0
        if role == "teacher":   # hollow, with a thick edge in the model's colour
            return dict(color=color, marker=mk, ms=6.2 * big, mfc="white", mec=color, mew=1.5)
        return dict(color=color, marker=mk, ms=4.5, mec=color, mew=0.6)

    plt.rcParams.update({"font.size": 9.5, "axes.labelsize": 10, "legend.fontsize": 8})
    fig, axes = plt.subplots(3, 2, figsize=(5.6, 6.2), sharex="row", sharey="col")
    allpts, sizes = {}, {}
    panels = [("gpu", r"GPU time per jet ($\mu$s)", lambda v: 1e6 / v, [10, 20, 50, 100, 200]),
              ("cpu", "CPU latency (ms per jet)", lambda v: v, [2, 5, 10, 20, 50]),
              ("flops", "Operations per jet (FLOP)", lambda v: v, None)]
    for row, (key, xlabel, f, ticks) in enumerate(panels):
        for col, (metric, eps) in enumerate((("rej50", "0.5"), ("rej30", "0.3"))):
            ax = axes[row, col]
            for chain in chains:
                xy = sorted((f(pts[m][key]), cell[metric]) for m, _, cell, _, _ in marks if m in chain)
                color = [c for m, _, _, c, _ in marks if m == chain[0]][0]
                ax.plot(*zip(*xy, strict=True), color=color, lw=0.9, ls=(0, (1, 2)), alpha=0.6, zorder=1)
            for m, role, cell, color, mk in marks:
                if pts[m][key] is None:
                    continue
                x, y = f(pts[m][key]), cell[metric]
                ax.errorbar(x, y, yerr=cell.get(f"{metric}_sd", 0), color=color, lw=0, elinewidth=0.7, capsize=0,
                            zorder=2)
                ax.plot(x, y, lw=0, zorder=4 if role == "teacher" else 3, **look(role, color, mk))
                if key == "flops":   # student sizes, in this row only
                    allpts.setdefault(ax, []).append((x, y))
                    if m in COST_SIZES:
                        sizes.setdefault(ax, []).append((x, y, COST_SIZES[m], color))
            ax.set_xscale("log")
            xs = [f(pts[m][key]) for m, *_ in marks if pts[m][key] is not None]
            ax.set_xlim(min(xs) / 1.35, max(xs) * 1.35)
            if ticks:
                ax.set_xticks(ticks)
                ax.set_xticklabels([f"{t:g}" for t in ticks])
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.tick_params(labelbottom=True, labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(True, which="major", color="#e4e4e4", lw=0.6)
            ax.set_axisbelow(True)
            ax.set_xlabel(xlabel)   # reserves the space; one label per row below
            if row == 0:
                ax.set_title(rf"$\epsilon_S={eps}$", fontsize=10)
            if col == 0:
                ax.set_ylabel(r"$1/\epsilon_B$")
    models = [Line2D([], [], lw=0, label=lb, **look("distilled", c, mk)) for _, lb, c, mk in COST_FAMILIES]
    models.append(Line2D([], [], lw=0, label=NAMES["part_pretrained"], **look("teacher", "#1baf7a", "*")))
    fig.legend(handles=models, loc="upper center", ncol=5, frameon=False, fontsize=8, handlelength=1.2,
               columnspacing=1.0, bbox_to_anchor=(0.5, 1.0))
    key = [Line2D([], [], lw=0, label=f"distilled from {NAMES['part_pretrained']}",
                  **look("distilled", "#0b0b0b", "o")),
           Line2D([], [], lw=0, label="teacher models", **look("teacher", "#0b0b0b", "o"))]
    fig.legend(handles=key, loc="upper center", ncol=2, frameon=False, fontsize=8, handlelength=1.2,
               columnspacing=2.0, bbox_to_anchor=(0.5, 1.0 - 0.40 / fig.get_figheight()))
    fig.tight_layout(rect=(0, 0, 1, 0.905), h_pad=0.6, w_pad=0.8)
    fig.canvas.draw()   # one x label per row, centred under both panels
    inv = fig.transFigure.inverted()
    for row, (_, xlabel, *_) in enumerate(panels):
        y = inv.transform(axes[row, 0].xaxis.label.get_window_extent().get_points().mean(0))[1]
        for ax in axes[row]:
            ax.xaxis.label.set_visible(False)
        x = (axes[row, 0].get_position().x0 + axes[row, 1].get_position().x1) / 2
        fig.text(x, y, xlabel, ha="center", va="center", fontsize=10)
    rend = fig.canvas.get_renderer()
    pad = 3.2 * fig.dpi / 72
    for ax, items in sizes.items():   # each size next to its own point, at the first free spot
        box = ax.get_window_extent(rend)
        taken = [matplotlib.transforms.Bbox.from_bounds(px - pad, py - pad, 2 * pad, 2 * pad)
                 for px, py in (ax.transData.transform(p) for p in allpts[ax])]
        for x, y, text, color in items:
            for dx, dy, ha, va in ((4, 0, "left", "center"), (-4, 0, "right", "center"),
                                   (0, 4, "center", "bottom"), (0, -4, "center", "top"),
                                   (3, 3, "left", "bottom"), (-3, 3, "right", "bottom"),
                                   (3, -3, "left", "top"), (-3, -3, "right", "top")):
                t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                                fontsize=5.5, color=color, zorder=5)
                bb = t.get_window_extent(rend)
                if (box.x0 <= bb.x0 and bb.x1 <= box.x1 and box.y0 <= bb.y0 and bb.y1 <= box.y1
                        and not any(bb.overlaps(o) for o in taken)):
                    taken.append(bb)
                    break
                t.remove()
            else:
                raise AssertionError(f"no free spot for the size label {text}")
    return save(fig, out / "cost_perf.pdf")


FIGURES = {
    "rej_by_teacher": rej_by_teacher,
    "strength": strength,
    "scaling_width": lambda r, a, out: scaling(r, "width", "rej30", out),
    "scaling_depth": lambda r, a, out: scaling(r, "depth", "rej30", out),
    "scaling_width_rej50": lambda r, a, out: scaling(r, "width", "rej50", out),
    "scaling_depth_rej50": lambda r, a, out: scaling(r, "depth", "rej50", out),
    "analysis": analysis,
    "analysis_size": analysis_size,
    "analysis_shared": analysis_shared,
    "analysis_distinctive": analysis_distinctive,
    "analysis_accepted": analysis_accepted,
    "analysis_perjet": analysis_perjet,
    "cost_perf": cost_perf,
    "cost_params": cost_params,
}
NEEDS_TIMINGS = ("cost_perf", "cost_params")


def figures(summary: dict, analysis_: dict, out: Path, names=FIGURES, benchmark: dict | None = None,
            flops: dict | None = None) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    r, paths = Results(summary, analysis_, benchmark, flops), []
    for name in names:
        with plt.rc_context(BASE):   # every figure from the same settings
            paths += FIGURES[name](r, analysis_, out)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("figures", nargs="*", help=f"default: all of {', '.join(FIGURES)}")
    ap.add_argument("--summary", default="results/summary.json")
    ap.add_argument("--analysis", default="results/analysis.json")
    ap.add_argument("--benchmark", default="results/benchmark.json")
    ap.add_argument("--flops", default="results/flops.json")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    if unknown := set(args.figures) - set(FIGURES):
        ap.error(f"unknown figures: {', '.join(sorted(unknown))}")
    timings = Path(args.benchmark).is_file() and Path(args.flops).is_file()
    names = args.figures or [f for f in FIGURES if timings or f not in NEEDS_TIMINGS]
    if not timings and set(names) & set(NEEDS_TIMINGS):
        ap.error(f"{', '.join(NEEDS_TIMINGS)} need {args.benchmark} and {args.flops}")
    paths = figures(*(json.loads(Path(f).read_text()) for f in (args.summary, args.analysis)), Path(args.out),
                    names, *(json.loads(Path(f).read_text()) if timings else None for f in (args.benchmark, args.flops)))
    print(f"wrote {len(paths)} figures to {args.out}")


if __name__ == "__main__":
    main()
