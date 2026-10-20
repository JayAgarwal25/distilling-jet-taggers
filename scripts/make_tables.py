#!/usr/bin/env python
"""The paper's LaTeX tables from results/summary.json, results/analysis.json (scripts/summarize.py),
results/benchmark.json (scripts/benchmark.py), results/flops.json (scripts/count_flops.py) and the code.

The paper defines the macros \\std (seed spread, in grey), \\eqv (Lorentz-equivariant)
and \\pre (pretrained).

    python scripts/make_tables.py --out tables
"""

import argparse
import json
from functools import cache
from pathlib import Path

from jetkd.recipes import recipe_for
from jetkd.reference import model_settings
from jetkd.study import DEPTHS, ENSEMBLE, MATRIX_STUDENTS, TEACHERS

EQV = r"\eqv{}"
PRE = r"\pre{}"
ONE_RUN = r"$^\ddagger$"
REJ = r"$1/\epsilon_B$ ($\epsilon_S{=}0.5$) & $1/\epsilon_B$ ($\epsilon_S{=}0.3$)"
NAMES = {
    "transformer": "Transformer", "part": "ParT", "mipart": "MIParT", "lgatr": "L-GATr",
    "lgatr_slim": "L-GATr-slim", "lorentznet": "LorentzNet", "omnilearned_s": "OmniLearned-S",
    "part_pretrained": "ParT-f.t.",
}
EQUIVARIANT = ("lgatr", "lgatr_slim", "lorentznet")
PRETRAINED = ("omnilearned_s", "part_pretrained")
# counted in the teachers' own code bases
EXTERNAL_PARAMS = {"omnilearned_s": 2_708_149, "part_pretrained": 2_141_562}
SIZES = ("200k", "20k", "2k")
WIDTH_FAMILIES = ("transformer", "lgatr_slim")
SCHEDULES = {"cosine": "cosine", "flat+decay": "flat, then decay"}


def family(model: str) -> str:
    return model.split("_deep")[0].split("_200k")[0]


def name(model: str) -> str:
    f = family(model)
    return NAMES[f] + (EQV if f in EQUIVARIANT else "") + (PRE if f in PRETRAINED else "")


@cache
def settings(model: str) -> dict:
    return model_settings(model)


def params(model: str) -> int:
    return EXTERNAL_PARAMS.get(model) or settings(model)["parameters"]


def millions(model: str) -> str:
    return f"{params(model) / 1e6:.1f}M"


def short(p: int) -> str:
    """Parameters in full below 1k, else in thousands."""
    return num(p) if p < 1e3 else f"{p / 1e3:.1f}k" if p < 1e4 else f"{p / 1e3:.0f}k"


def num(x: float) -> str:
    return f"{x:,.0f}".replace(",", "{,}")


def sci(x: float) -> str:
    if x == 0 or 0.1 <= x < 100:
        return f"{x:g}"
    m, e = f"{x:.0e}".split("e")
    return "$" + ("" if m == "1" else m + r"\times") + "10^{" + str(int(e)) + "}$"


def bold(s: str, on: bool) -> str:
    return r"\textbf{" + s + "}" if on else s


def runs(c: dict) -> int:
    return len(c.get("seeds") or [0])


def auc(c: dict | None, best: bool = False) -> str:
    """AUC with its seed spread, or a double dagger for a single run."""
    if c is None:
        return "--"
    s = bold(f"{c['auc']:.5f}", best)
    return s + (rf"\std{{{c['auc_sd']:.5f}}}" if runs(c) > 1 and c.get("auc_sd") else ONE_RUN)


def auc_runs(c: dict) -> str:
    """As `auc`, without the dagger: these tables state their run counts in the caption."""
    return auc(c) if runs(c) > 1 else f"{c['auc']:.5f}"


def rej(c: dict | None, m: str, best: bool = False) -> str:
    if c is None:
        return "--"
    sd = c.get(f"{m}_sd")
    if sd is None or runs(c) < 2:
        return bold(num(c[m]), best)
    return bold(num(c[m]), best) + (rf"\std{{{sd:.1f}}}" if sd < 1 else rf"\std{{{sd:.0f}}}")


def pm(c: dict, m: str) -> str:
    return auc(c) if m == "auc" else rej(c, m)


def wrap(colspec: str, header: str, rows: list[str]) -> str:
    body = "\n".join(r if r.endswith("\\midrule") or r.startswith("\\cmidrule") else r + r" \\"
                     for r in rows)
    return ("\\begin{tabular}{" + colspec + "}\n\\toprule\n" + header + r" \\" + "\n\\midrule\n"
            + body + "\n\\bottomrule\n\\end{tabular}\n")


def sized(size: str, tex: str, sep: str | None = None) -> str:
    """`tex` in a font size, optionally with another column separation."""
    return "{\\" + size + "\n" + (f"\\setlength{{\\tabcolsep}}{{{sep}}}\n" if sep else "") + tex + "}\n"


def student_header(lead: int) -> str:
    """A spanning "Student" row over the student columns, after `lead` other columns."""
    n = len(MATRIX_STUDENTS)
    return ("&" * lead + r" \multicolumn{" + str(n) + r"}{c}{Student} \\"
            + r" \cmidrule(lr){" + f"{lead + 1}-{lead + n}" + "}\n")


class Results:
    """The summary, read by student and teacher (None for no teacher), the analyses, the timings
    and the operation counts."""

    def __init__(self, summary: dict, analysis: dict | None = None, benchmark: dict | None = None,
                 flops: dict | None = None):
        self.s = summary
        self.analysis = analysis
        self.timings = {(t["model"], t["device"], t["regime"]): t for t in (benchmark or {}).get("rows", [])}
        self.flops = (flops or {}).get("models", {})

    def cell(self, student: str, teacher: str | None) -> dict | None:
        return self.s["cells"].get(f"{student}|{teacher or 'scratch'}")

    def run(self, student: str, teacher: str | None) -> dict | None:
        """The seed-1001 run, as the depth study has only that seed."""
        arm = f"kd_{teacher}_t1001" if teacher else "scratch"
        return self.s["runs"].get(f"{student}__{arm}__s1001")

    def teacher(self, t: str) -> dict:
        return self.s["teachers"][t]

    def teacher_ensemble(self, t: str) -> dict:
        return self.s["teacher_ensembles"][t]


# ---------------------------------------------------------------- tables

def teachers(r: Results) -> str:
    rows = []
    for pretrained in (False, True):
        if rows:
            rows.append(r"\midrule")
        for t in TEACHERS:
            if (t in PRETRAINED) == pretrained:
                c = r.teacher(t)
                rows.append(f"{name(t)} & {millions(t)} & {auc(c)} & {rej(c, 'rej50')} & {rej(c, 'rej30')}")
    return wrap("lrlrr", "Teacher & Params & AUC & " + REJ, rows)


# published performance of each teacher and of LorentzNet, from the papers that introduced them:
# parameters, AUC and 1/eps_B at eps_S = 0.5 and 0.3
PUBLISHED = {
    "part": ("2.1M", "0.9858", r"413\std{16}", r"1{,}602\std{81}"),
    "mipart": ("0.7M", "0.9868", r"505\std{8}", r"2{,}010\std{97}"),
    "lgatr": ("1.1M", r"0.9870\std{0.0001}", r"540\std{20}", r"2{,}240\std{70}"),
    "lgatr_slim": ("2.0M", r"0.9869\std{0.0001}", r"546\std{7}", r"2{,}264\std{93}"),
    "omnilearned_s": ("3M", "0.9875", r"565\std{12}", r"2{,}637\std{128}"),
    "part_pretrained": ("2.1M", "0.9877", r"691\std{15}", r"2{,}766\std{130}"),
    "lorentznet_200k": ("224k", "0.9868", r"498\std{18}", r"2{,}195\std{173}"),
}


def published(r: Results) -> str:
    """Our teachers and LorentzNet without a teacher next to their published performance."""
    rows = []
    for pretrained in (False, True):
        if rows:
            rows.append(r"\midrule")
        for t in TEACHERS:
            if (t in PRETRAINED) == pretrained:
                c = r.teacher(t)
                rows.append(f"{name(t)} & {millions(t)} & {auc(c)} & {rej(c, 'rej50')} & {rej(c, 'rej30')} & "
                            + " & ".join(PUBLISHED[t]))
    s = "lorentznet_200k"
    c = r.cell(s, None)
    rows += [r"\midrule", f"{name(s)} & {num(params(s) / 1e3)}k & {auc(c)} & {rej(c, 'rej50')} & "
             f"{rej(c, 'rej30')} & " + " & ".join(PUBLISHED[s])]
    head = (r"& \multicolumn{4}{c}{This work} & \multicolumn{4}{c}{Published} \\"
            "\n\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}\n"
            r"& & & \multicolumn{2}{c}{$1/\epsilon_B$} & & & \multicolumn{2}{c}{$1/\epsilon_B$} \\"
            "\n\\cmidrule(lr){4-5} \\cmidrule(lr){8-9}\n"
            r"Model & Params & AUC & $\epsilon_S{=}0.5$ & $\epsilon_S{=}0.3$ & Params & AUC & $\epsilon_S{=}0.5$ & $\epsilon_S{=}0.3$")
    return wrap("lrlrrrlrr", head, rows)


def teacher_ensembles(r: Results) -> str:
    """Every teacher as a single run (mean and spread over its runs) and as their ensemble."""
    rows = []
    for i, t in enumerate(TEACHERS):
        if t in PRETRAINED and TEACHERS[i - 1] not in PRETRAINED:
            rows.append(r"\midrule")
        c, x = r.teacher(t), r.teacher_ensemble(t)
        rows.append(f"{name(t)} & {auc(c)} & {rej(c, 'rej50')} & {rej(c, 'rej30')} & "
                    f"{x['auc']:.5f} & {num(x['rej50'])} & {num(x['rej30'])}")
    head = (r"& \multicolumn{3}{c}{Single run} & \multicolumn{3}{c}{Ensemble of three runs} \\"
            "\n\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n"
            r"& & \multicolumn{2}{c}{$1/\epsilon_B$} & & \multicolumn{2}{c}{$1/\epsilon_B$} \\"
            "\n\\cmidrule(lr){3-4} \\cmidrule(lr){6-7}\n"
            r"Teacher & AUC & $\epsilon_S{=}0.5$ & $\epsilon_S{=}0.3$ & AUC & $\epsilon_S{=}0.5$ & $\epsilon_S{=}0.3$")
    return wrap("llrrlrr", head, rows)


def students(r: Results) -> str:
    rows = []
    for s in MATRIX_STUDENTS:
        c = r.cell(s, None)
        rows.append(f"{name(s)} & {num(params(s) / 1e3)}k & {auc(c)} & {rej(c, 'rej50')} & {rej(c, 'rej30')}")
    return wrap("lrlrr", "Student & Params & AUC & " + REJ, rows)


def best_teacher(r: Results, s: str, m: str) -> str:
    return max((t for t in TEACHERS if r.cell(s, t)), key=lambda t: r.cell(s, t)[m])


def matrix_auc(r: Results) -> str:
    best = {s: best_teacher(r, s, "auc") for s in MATRIX_STUDENTS}
    rows = ["No teacher & " + " & ".join(auc(r.cell(s, None)) for s in MATRIX_STUDENTS), r"\midrule"]
    for i, t in enumerate(TEACHERS):
        if t in PRETRAINED and TEACHERS[i - 1] not in PRETRAINED:
            rows.append(r"\midrule")
        rows.append(f"{name(t)} & " + " & ".join(auc(r.cell(s, t), t == best[s]) for s in MATRIX_STUDENTS))
    header = student_header(1) + "Teacher & " + " & ".join(map(name, MATRIX_STUDENTS))
    return sized("footnotesize", wrap("l" + "l" * len(MATRIX_STUDENTS), header, rows), "3.4pt")


def matrix_rej(r: Results) -> str:
    rows = []
    for m, eff in (("rej50", "0.5"), ("rej30", "0.3")):
        if rows:
            rows.append(r"\midrule")
        rows.append(r"\multicolumn{" + str(len(MATRIX_STUDENTS) + 1)
                    + r"}{l}{\textit{$1/\epsilon_B$ at $\epsilon_S=" + eff + "$}}")
        best = {s: best_teacher(r, s, m) for s in MATRIX_STUDENTS}
        rows.append("No teacher & " + " & ".join(rej(r.cell(s, None), m) for s in MATRIX_STUDENTS))
        for t in TEACHERS:
            rows.append(f"{name(t)} & " + " & ".join(rej(r.cell(s, t), m, t == best[s]) for s in MATRIX_STUDENTS))
    header = student_header(1) + "Teacher & " + " & ".join(map(name, MATRIX_STUDENTS))
    return sized("small", wrap("l" + "r" * len(MATRIX_STUDENTS), header, rows))


def depth_models(base: str) -> list[tuple[int, str]]:
    blocks, own = DEPTHS[base]
    return [(n, base if n == own else f"{base}_L{n}") for n in blocks]


def width_models(f: str) -> list[str]:
    return [f"{f}_deep_{sz}" for sz in SIZES]


def scaling(r: Results, study: str, pair: tuple[str | None, str]) -> str:
    """A scaling study for two teachers side by side (None: no teacher), smallest student first:
    AUC and both rejections. Width: three runs with spreads; depth: one run per number of blocks."""
    rows = []
    for base, f in zip(DEPTHS, WIDTH_FAMILIES, strict=True):
        if rows:
            rows.append(r"\midrule")
        if study == "width":
            entries = [([], m, lambda t, m=m: r.cell(m, t)) for m in reversed(width_models(f))]
        else:
            entries = [([str(n)], m, lambda t, m=m: r.run(m, t)) for n, m in depth_models(base)]
        for i, (blocks, m, cell) in enumerate(entries):
            lead = [name(base) if i == 0 else "", *blocks, short(params(m))]
            rows.append(" & ".join(lead + [x for t in pair for c in [cell(t)]
                                           for x in (auc_runs(c), rej(c, "rej50"), rej(c, "rej30"))]))
    k = 3 if study == "depth" else 2
    title = ["No teacher" if t is None else name(t) for t in pair]
    header = ("& " * k + rf"\multicolumn{{3}}{{c}}{{{title[0]}}} & \multicolumn{{3}}{{c}}{{{title[1]}}} \\"
              rf" \cmidrule(lr){{{k + 1}-{k + 3}}} \cmidrule(lr){{{k + 4}-{k + 6}}}" "\n"
              + "& " * (k + 1) + r"\multicolumn{2}{c}{$1/\epsilon_B$} & & \multicolumn{2}{c}{$1/\epsilon_B$} \\"
              + rf" \cmidrule(lr){{{k + 2}-{k + 3}}} \cmidrule(lr){{{k + 5}-{k + 6}}}" "\n"
              + ("Student & Blocks & Params" if study == "depth" else "Student & Params")
              + r" & AUC & $\epsilon_S{=}0.5$ & $\epsilon_S{=}0.3$" * 2)
    return sized("footnotesize", wrap("l" + "r" * (k - 1) + "lrr" * 2, header, rows), "3.5pt")


def recipes() -> str:
    entries = [(name(s), recipe_for(s)) for s in MATRIX_STUDENTS]
    entries.append(None)   # the width study, one row per model
    entries += [(f"{name(m)} {short(params(m))}", recipe_for(m)) for f in WIDTH_FAMILIES for m in width_models(f)]
    rows = [r"\midrule" if e is None else
            f"{e[0]} & {e[1].optimizer} & {sci(e[1].lr)} & {sci(e[1].weight_decay)} & {num(e[1].batch_size)} & "
            f"{num(e[1].iterations)} & {SCHEDULES[e[1].scheduler]}" for e in entries]
    header = "Model & Optimizer & Learning rate & Weight decay & Batch & Steps & Schedule"
    return sized("small", wrap("lllllrl", header, rows))


# feature channels per head of the Transformer's attention representations
CHANNELS_PER_HEAD = {"4x0n+1x1n": 8, "1x1n": 4}


def architectures() -> str:
    def a(m: str) -> dict:
        return settings(m)["arguments"]

    def row(label, width, heads, blocks, readout, m):
        return f"{label} & {width} & {heads} & {blocks} & {readout} & {num(params(m))}"

    def transformer(m):
        return row(name(m), CHANNELS_PER_HEAD[a(m)["attn_reps"]] * a(m)["num_heads"], a(m)["num_heads"],
                   a(m)["num_blocks"], "mean", m)

    def slim(m):
        return row(name(m), f"{a(m)['hidden_v_channels']} v, {a(m)['hidden_s_channels']} s",
                   a(m)["num_heads"], a(m)["num_blocks"], "global token", m)

    def mipart(m):
        return row(name(m), a(m)["embed_dims"][-1], a(m)["num_heads"],
                   f"{a(m)['num_MIlayers']} + {a(m)['num_layers']} + {a(m)['num_cls_layers']}", "class token", m)

    p, lg = a("part"), a("lgatr")
    part_blocks = f"{p['num_layers']} + {p['num_cls_layers']}"
    rows = [r"\multicolumn{6}{l}{\textit{Teachers}}",
            row("ParT", p["embed_dims"][-1], p["num_heads"], part_blocks, "class token", "part"),
            mipart("mipart"),
            row(name("lgatr"), f"{lg['hidden_mv_channels']} mv, {lg['hidden_s_channels']} s", lg["num_heads"],
                lg["num_blocks"], "global token", "lgatr"),
            slim("lgatr_slim"),
            row(name("omnilearned_s"), 128, 8, "8 + 2", "4 class tokens", "omnilearned_s"),
            # the ParT authors' fine-tuning network: ParT's blocks, other inputs
            row(name("part_pretrained"), p["embed_dims"][-1], p["num_heads"], part_blocks, "class token",
                "part_pretrained"),
            r"\midrule", r"\multicolumn{6}{l}{\textit{Students}}",
            transformer("transformer_200k"),
            row("ParT", a("part_200k")["d_model"], a("part_200k")["num_heads"], a("part_200k")["num_blocks"],
                "mean", "part_200k"),
            mipart("mipart_200k"),
            slim("lgatr_slim_200k"),
            row(name("lorentznet_200k"), a("lorentznet_200k")["n_hidden"], "--", a("lorentznet_200k")["n_layers"],
                "mean", "lorentznet_200k"),
            r"\midrule", r"\multicolumn{6}{l}{\textit{Width study}}"]
    rows += [transformer(m) for m in width_models("transformer")]
    rows += [slim(m) for m in width_models("lgatr_slim")]
    return sized("small", wrap("llllll", "Model & Width & Heads & Blocks & Read-out & Params", rows))


def ens_students(r: Results) -> str:
    """Students of L-GATr-slim, of the ensemble of its three runs and of OmniLearned-S."""
    arms = [(name("lgatr_slim"), "lgatr_slim"), (name("lgatr_slim") + " ens.", ENSEMBLE),
            (name("omnilearned_s"), "omnilearned_s")]
    n = len(MATRIX_STUDENTS) + 1
    rows = []
    for m, title in (("auc", "AUC"), ("rej50", r"$1/\epsilon_B$ at $\epsilon_S=0.5$"),
                     ("rej30", r"$1/\epsilon_B$ at $\epsilon_S=0.3$")):
        if rows:
            rows.append(r"\midrule")
        rows.append(r"\multicolumn{" + str(n) + r"}{l}{\textit{" + title + "}}")
        for label, t in arms:
            rows.append(f"{label} & " + " & ".join(pm(r.cell(s, t), m) for s in MATRIX_STUDENTS))
    header = student_header(1) + "Teacher & " + " & ".join(map(name, MATRIX_STUDENTS))
    return sized("footnotesize", wrap("l" * n, header, rows), "3pt")


def accepted(r: Results) -> str:
    """Background jets every matrix student accepts at eps_S = 0.5, without and with distillation,
    among the jets each teacher rejects and among those it accepts."""
    a = r.analysis["accepted"]
    n = 2 * len(MATRIX_STUDENTS) + 1
    rows = []
    for dec, title in (("rejects", "Background jets the teacher rejects"),
                       ("accepts", "Background jets the teacher accepts")):
        if rows:
            rows.append(r"\midrule")
        rows.append(r"\multicolumn{" + str(n) + r"}{l}{\textit{" + title + "}}")
        for t in TEACHERS:
            rows.append(f"{name(t)} & " + " & ".join(num(a[f"{s}|{t}|50"][f"{arm}_teacher_{dec}"])
                                                     for s in MATRIX_STUDENTS for arm in ("without", "with")))
    head = (" & " + " & ".join(r"\multicolumn{2}{c}{" + name(s) + "}" for s in MATRIX_STUDENTS) + r" \\" + "\n"
            + " ".join(rf"\cmidrule(lr){{{2 * k + 2}-{2 * k + 3}}}" for k in range(len(MATRIX_STUDENTS))) + "\n"
            + "Teacher & " + " & ".join("without & with" for _ in MATRIX_STUDENTS))
    return sized("footnotesize", wrap("l" + "r" * (n - 1), head, rows), "3.4pt")


# Deep Sets + GNN of arXiv:2609.26899 (Table 1), distilled from OmniLearned-L: performance as published,
# cost measured by us
DEEPSETS_GNN = {"model": "deepsets_mp", "label": "Deep Sets + GNN", "auc": 0.9859, "rej50": 450, "rej30": 1980}
COST_HEAD = (r"\multicolumn{2}{c}{$1/\epsilon_B$}", r"\multicolumn{2}{c}{GPU} & CPU",
             r"$\epsilon_S{=}0.5$ & $\epsilon_S{=}0.3$", r"jets/s & MB & ms/jet")
ATTENTION = {"builtin": "built-in", "xformers": "xFormers", "flex": "Flex"}


def cost_cells(r: Results, m: str, regime: str) -> str:
    """GPU jets per second and peak memory at the batch of the measurement, CPU ms per jet."""
    g, c = r.timings.get((m, "cuda", regime)), r.timings.get((m, "cpu", regime))
    gpu = "--" if g is None else num(g["batch_size"] / (g["median_ms"] / 1e3))
    mem = "--" if g is None or g.get("peak_mb") is None else num(g["peak_mb"])
    return f"{gpu} & {mem} & {'--' if c is None else format(c['median_ms'], '.1f')}"


def mflop(r: Results, m: str) -> str:
    """Operations per median jet in millions, three significant digits."""
    if m not in r.flops:
        return "--"
    x = r.flops[m]["flops_median"] / 1e6
    return f"{x:.2f}" if x < 10 else f"{x:.1f}" if x < 100 else num(x)


def perf(c: dict | None) -> str:
    if c is None:
        return "-- & -- & --"
    if c is DEEPSETS_GNN:   # as published, no spread
        return f"{c['auc']:.4f} & {num(c['rej50'])} & {num(c['rej30'])}"
    return f"{c['auc']:.5f} & {rej(c, 'rej50')} & {rej(c, 'rej30')}"


def cost_groups(r: Results, arm: str | None) -> list:
    """(title, [(label, params, performance, model)]): the arm's teacher, the five students and the
    width study where it was run."""
    groups = [] if arm is None else [("Teacher", [(name(arm), millions(arm), r.teacher(arm), arm)])]
    groups.append(("Students", [(name(s), f"{num(params(s) / 1e3)}k", r.cell(s, arm), s) for s in MATRIX_STUDENTS]))
    width = [(name(m), short(params(m)), r.cell(m, arm), m) for f in WIDTH_FAMILIES for m in width_models(f)]
    if any(c is not None for _, _, c, _ in width):
        groups.append(("Width study", width))
    return groups


def cost_rows(r: Results, groups: list) -> str:
    rows = []
    for title, items in groups:
        if rows:
            rows.append(r"\midrule")
        rows.append(r"\multicolumn{9}{l}{\textit{" + title + "}}")
        rows += [f"{label} & {ptxt} & {perf(c)} & {mflop(r, m)} & {cost_cells(r, m, 'real')}"
                 for label, ptxt, c, m in items]
    header = (rf" & & & {COST_HEAD[0]} & & {COST_HEAD[1]} \\ \cmidrule(lr){{4-5}} \cmidrule(lr){{7-8}}" "\n"
              rf"Model & Params & AUC & {COST_HEAD[2]} & MFLOP & {COST_HEAD[3]}")
    return "{\\small\\setlength{\\tabcolsep}{4pt}\n" + wrap("lrrrrrrrr", header, rows) + "}\n"


def cost(r: Results) -> str:
    """Performance, operations per median jet and inference cost on real test jets: every teacher,
    the students distilled from ParT-f.t. and Deep Sets + GNN."""
    t = "part_pretrained"
    groups = [("Teachers", [(name(x), millions(x), r.teacher(x), x) for x in TEACHERS])]
    groups += [(f"{title}, distilled from {name(t)}", items) for title, items in cost_groups(r, t)[1:]]
    groups.append((f"Distilled from OmniLearned-L{PRE}" + r"~\citep{benyas2026accelerators}",
                   [(DEEPSETS_GNN["label"], short(r.flops[DEEPSETS_GNN["model"]]["parameters"]), DEEPSETS_GNN,
                     DEEPSETS_GNN["model"])]))
    return cost_rows(r, groups)


def cost_rnd50(r: Results) -> str:
    """The cost on jets of exactly 50 particles with random momenta, the setting of arXiv:2608.02735."""
    entries = [("Teachers", [(name(t), millions(t), t) for t in TEACHERS]),
               ("Students", [(name(s), f"{num(params(s) / 1e3)}k", s) for s in MATRIX_STUDENTS]),
               ("Width study", [(name(m), short(params(m)), m) for f in WIDTH_FAMILIES for m in width_models(f)]),
               (f"Distilled from OmniLearned-L{PRE}",
                [(DEEPSETS_GNN["label"], short(r.flops[DEEPSETS_GNN["model"]]["parameters"]), DEEPSETS_GNN["model"])])]
    rows = []
    for title, items in entries:
        if rows:
            rows.append(r"\midrule")
        rows.append(r"\multicolumn{5}{l}{\textit{" + title + "}}")
        rows += [f"{label} & {ptxt} & {cost_cells(r, m, 'rnd50')}" for label, ptxt, m in items]
    header = (r" & & \multicolumn{2}{c}{GPU} & CPU \\ \cmidrule(lr){3-4}" "\n"
              r"Model & Params & jets/s & MB & ms/jet")
    return sized("small", wrap("lrrrr", header, rows))


def execution_settings(r: Results) -> str:
    """The settings timed for every model on real jets and the fastest on each device."""
    def word(setting: str) -> str:
        lists, compiled, attention = setting.split(",")
        return f"{lists}, {compiled}, {ATTENTION[attention]}"

    entries = [(name(t), t) for t in TEACHERS]
    entries += [(f"{name(s)} {num(params(s) / 1e3)}k", s) for s in MATRIX_STUDENTS]
    entries += [(f"{name(m)} {short(params(m))}", m) for f in WIDTH_FAMILIES for m in width_models(f)]
    entries.append((DEEPSETS_GNN["label"], DEEPSETS_GNN["model"]))
    rows = []
    for label, m in entries:
        g, c = r.timings[(m, "cuda", "real")], r.timings[(m, "cpu", "real")]
        tried = [x.split(",") for x in (*g["settings"], *c["settings"])]
        lists = [x for x in ("packed", "padded") if any(t[0] == x for t in tried)]
        attention = [ATTENTION[x] for x in ATTENTION if any(t.split(",")[2] == x for t in g["settings"])]
        rows.append(f"{label} & {', '.join(lists)} & {', '.join(attention)} & {word(g['selected'])} & "
                    f"{word(c['selected'])}")
    header = (r" & & & \multicolumn{2}{c}{Fastest setting} \\ \cmidrule(lr){4-5}" "\n"
              r"Model & Particle lists & Attention & GPU & CPU")
    return sized("footnotesize", wrap("lllll", header, rows))


TABLES = {
    "teachers": teachers, "published": published, "teacher_ensembles": teacher_ensembles, "students": students,
    "matrix_auc": matrix_auc, "matrix_rej": matrix_rej, "ens_students": ens_students,
    "size_short": lambda r: scaling(r, "width", (None, "part_pretrained")),
    "size_other": lambda r: scaling(r, "width", ("lgatr", "omnilearned_s")),
    "depth_main": lambda r: scaling(r, "depth", (None, "part_pretrained")),
    "depth_other": lambda r: scaling(r, "depth", ("lgatr", "omnilearned_s")),
    "recipes": lambda r: recipes(), "architectures": lambda r: architectures(),
    "accepted": accepted,
    "cost": cost, "cost_rnd50": cost_rnd50, "settings": execution_settings,
    **{f"cost_{a or 'scratch'}": (lambda r, a=a: cost_rows(r, cost_groups(r, a)))
       for a in (None, "part", "mipart", "lgatr", "lgatr_slim", "omnilearned_s")},
}
NEEDS_ANALYSIS = ("accepted",)
NEEDS_TIMINGS = ("cost", "cost_rnd50", "settings", *(f"cost_{a}" for a in
                                                     ("scratch", "part", "mipart", "lgatr", "lgatr_slim", "omnilearned_s")))


def tables(summary: dict, analysis: dict, benchmark: dict | None = None, flops: dict | None = None,
           names=TABLES) -> dict[str, str]:
    r = Results(summary, analysis, benchmark, flops)
    return {name: TABLES[name](r) for name in names}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--summary", default="results/summary.json")
    ap.add_argument("--analysis", default="results/analysis.json")
    ap.add_argument("--benchmark", default="results/benchmark.json")
    ap.add_argument("--flops", default="results/flops.json")
    ap.add_argument("--out", default="tables")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    timings = Path(args.benchmark).is_file() and Path(args.flops).is_file()
    if not timings:
        print(f"no {args.benchmark} or {args.flops}: the cost tables are left out")
    written = tables(*(json.loads(Path(f).read_text()) for f in (args.summary, args.analysis)),
                     *(json.loads(Path(f).read_text()) if timings else None for f in (args.benchmark, args.flops)),
                     names=[t for t in TABLES if timings or t not in NEEDS_TIMINGS])
    for t, tex in written.items():
        (out / f"{t}.tex").write_text("% generated by scripts/make_tables.py; do not edit\n" + tex)
    print(f"wrote {len(written)} tables to {out}")


if __name__ == "__main__":
    main()
