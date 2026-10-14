#!/usr/bin/env python
"""The paper's LaTeX tables from results/summary.json, results/flops.json and the code.

The paper defines the macros \\std (seed spread) and \\eqv (Lorentz-equivariant model).

    python scripts/make_tables.py --summary results/summary.json --out tables
"""

import argparse
import json
from functools import cache
from pathlib import Path

from jetkd.recipes import RECIPES, recipe_for
from jetkd.reference import model_settings
from jetkd.study import DEPTHS, MATRIX_STUDENTS, SCALING_TEACHERS, TEACHERS

EQV = r"\eqv"
ONE_RUN = r"$^\dagger$"
REJ = r"$1/\epsilon_B$ ($\epsilon_S{=}0.5$) & $1/\epsilon_B$ ($\epsilon_S{=}0.3$)"
NAMES = {
    "transformer": "Transformer", "part": "ParT", "mipart": "MIParT", "lgatr": "L-GATr",
    "lgatr_slim": "L-GATr-slim", "lorentznet": "LorentzNet", "omnilearned_s": "OmniLearned-S",
    "part_pretrained": "ParT (pretrained)",
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
    return NAMES[f] + (EQV if f in EQUIVARIANT else "")


@cache
def settings(model: str) -> dict:
    return model_settings(model)


def params(model: str) -> int:
    return EXTERNAL_PARAMS.get(model) or settings(model)["parameters"]


def millions(model: str) -> str:
    return f"{params(model) / 1e6:.1f}M"


def thousands(model: str) -> str:
    p = params(model)
    return f"{p / 1e3:.1f}k" if p < 1e4 else f"{p / 1e3:.0f}k"


def num(x: float) -> str:
    return f"{x:,.0f}".replace(",", "{,}")


def signed(x: float, fmt: str) -> str:
    return ("+" if x >= 0 else "$-$") + format(abs(x), fmt)


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
    """AUC with the seed spread in units of 1e-5, or a dagger for a single run."""
    if c is None:
        return "--"
    s = bold(f"{c['auc']:.5f}", best)
    return s + (rf"\std{{{c['auc_sd'] * 1e5:.0f}}}" if runs(c) > 1 and c.get("auc_sd") else ONE_RUN)


def auc_runs(c: dict) -> str:
    """As `auc`, without the dagger: these tables state their run counts in the caption."""
    return auc(c) if runs(c) > 1 else f"{c['auc']:.5f}"


def rej(c: dict | None, m: str, best: bool = False) -> str:
    if c is None:
        return "--"
    sd = c.get(f"{m}_sd")
    return bold(num(c[m]), best) + (rf"\std{{{sd:.0f}}}" if sd is not None and runs(c) > 1 else "")


def wrap(colspec: str, header: str, rows: list[str], small: bool = True) -> str:
    body = "\n".join(r if r.endswith("\\midrule") or r.startswith("\\cmidrule") else r + r" \\"
                     for r in rows)
    tex = ("\\begin{tabular}{" + colspec + "}\n\\toprule\n" + header + r" \\" + "\n\\midrule\n"
           + body + "\n\\bottomrule\n\\end{tabular}\n")
    return "{\\small\n" + tex + "}\n" if small else tex


def student_header(lead: int) -> str:
    """A spanning "Student" row over the student columns, after `lead` other columns."""
    n = len(MATRIX_STUDENTS)
    return ("&" * lead + r" \multicolumn{" + str(n) + r"}{c}{Student} \\"
            + r" \cmidrule(lr){" + f"{lead + 1}-{lead + n}" + "}\n")


class Results:
    """The summary, read by student and teacher (None for no teacher)."""

    def __init__(self, summary: dict, flops: dict | None = None):
        self.s, self.flops = summary, flops

    def cell(self, student: str, teacher: str | None) -> dict | None:
        return self.s["cells"].get(f"{student}|{teacher or 'scratch'}")

    def run(self, student: str, teacher: str | None) -> dict | None:
        """The seed-1001 run, as the depth study has only that seed."""
        arm = f"kd_{teacher}_t1001" if teacher else "scratch"
        return self.s["runs"].get(f"{student}__{arm}__s1001")

    def teacher(self, t: str) -> dict:
        return self.s["teachers"][t]


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
    return wrap("lrlrr", "Teacher & Params & AUC & " + REJ, rows, small=False)


def students(r: Results) -> str:
    rows = []
    for s in MATRIX_STUDENTS:
        c = r.cell(s, None)
        rows.append(f"{name(s)} & {num(params(s) / 1e3)}k & {auc(c)} & {rej(c, 'rej50')} & {rej(c, 'rej30')}")
    return wrap("lrlrr", "Student & Params & AUC & " + REJ, rows, small=False)


def best_teacher(r: Results, s: str, value) -> str:
    return max((t for t in TEACHERS if r.cell(s, t)), key=lambda t: value(r.cell(s, t)))


def matrix_auc(r: Results) -> str:
    best = {s: best_teacher(r, s, lambda c: c["auc"]) for s in MATRIX_STUDENTS}
    rows = ["No teacher & -- & " + " & ".join(auc(r.cell(s, None)) for s in MATRIX_STUDENTS), r"\midrule"]
    for t in TEACHERS:
        cells = [auc(r.cell(s, t), t == best[s]) for s in MATRIX_STUDENTS]
        rows.append(f"{name(t)} & {r.teacher(t)['auc']:.5f} & " + " & ".join(cells))
    header = student_header(2) + "Teacher & Teacher AUC & " + " & ".join(map(name, MATRIX_STUDENTS))
    tex = wrap("lc" + "l" * len(MATRIX_STUDENTS), header, rows, small=False)
    return "{\\small\n\\setlength{\\tabcolsep}{4pt}\n" + tex + "}\n"


def matrix_rej(r: Results) -> str:
    rows = []
    for m, eff in (("rej50", "0.5"), ("rej30", "0.3")):
        if rows:
            rows.append(r"\midrule")
        rows.append(r"\multicolumn{" + str(len(MATRIX_STUDENTS) + 1)
                    + r"}{l}{\textit{$1/\epsilon_B$ at $\epsilon_S=" + eff + "$}}")
        best = {s: best_teacher(r, s, lambda c, m=m: c[m]) for s in MATRIX_STUDENTS}
        rows.append("No teacher & " + " & ".join(rej(r.cell(s, None), m) for s in MATRIX_STUDENTS))
        for t in TEACHERS:
            rows.append(f"{name(t)} & " + " & ".join(rej(r.cell(s, t), m, t == best[s]) for s in MATRIX_STUDENTS))
    header = student_header(1) + "Teacher & " + " & ".join(map(name, MATRIX_STUDENTS))
    return wrap("l" + "r" * len(MATRIX_STUDENTS), header, rows)


def rej50_gain(c: dict) -> float:
    """Relative gain in 1/eps_B at eps_S=0.5 against the seed-paired runs without a teacher."""
    return c["gain_rej50"] / (c["rej50"] - c["gain_rej50"])


def gains(r: Results) -> str:
    blocks = [("AUC", lambda c: c["gain_auc"], lambda x: signed(x, ".5f")),
              (r"$1/\epsilon_B$ ($\epsilon_S{=}0.5$)", rej50_gain, lambda x: signed(100 * x, ".0f") + r"\%")]
    rows = []
    for label, value, fmt in blocks:
        if rows:
            rows.append("\\midrule")
        rows.append(r"\multicolumn{" + str(len(MATRIX_STUDENTS) + 1) + r"}{l}{\textit{Gain in " + label + "}}")
        best = {s: best_teacher(r, s, value) for s in MATRIX_STUDENTS}
        for t in TEACHERS:
            cells = []
            for s in MATRIX_STUDENTS:
                c = r.cell(s, t)
                if c is None or c.get("gain_auc") is None:
                    cells.append("--")
                else:
                    cells.append(bold(fmt(value(c)), t == best[s]) + (ONE_RUN if c["gain_seeds"] == 1 else ""))
            rows.append(f"{name(t)} & " + " & ".join(cells))
    header = student_header(1) + "Teacher & " + " & ".join(map(name, MATRIX_STUDENTS))
    return wrap("l" + "r" * len(MATRIX_STUDENTS), header, rows)


def depth_models(base: str) -> list[tuple[int, str]]:
    blocks, own = DEPTHS[base]
    return [(n, base if n == own else f"{base}_L{n}") for n in blocks]


def width_models(f: str) -> list[str]:
    return [f"{f}_deep_{sz}" for sz in SIZES]


def two(c: dict | None) -> str:
    return "-- & --" if c is None else f"{auc_runs(c)} & {rej(c, 'rej30')}"


SHORT_HEADER = (r"\multicolumn{2}{c}{No teacher} & \multicolumn{2}{c}{ParT (pretrained)} \\")


def depth(r: Results) -> str:
    rows = []
    for base in DEPTHS:
        if rows:
            rows.append(r"\midrule")
        for i, (n, m) in enumerate(depth_models(base)):
            rows.append(" & ".join([name(base) if i == 0 else "", str(n), num(params(m) / 1e3) + "k",
                                    two(r.run(m, None)), two(r.run(m, "part_pretrained"))]))
    header = (r" & & & " + SHORT_HEADER + r" \cmidrule(lr){4-5} \cmidrule(lr){6-7}" "\n"
              r"Student & Blocks & Params & AUC & $1/\epsilon_B$ & AUC & $1/\epsilon_B$")
    return wrap("lrrlrlr", header, rows)


def size(r: Results) -> str:
    rows = []
    for f in WIDTH_FAMILIES:
        if rows:
            rows.append(r"\midrule")
        for i, m in enumerate(width_models(f)):
            rows.append(" & ".join([name(m) if i == 0 else "", thousands(m),
                                    two(r.cell(m, None)), two(r.cell(m, "part_pretrained"))]))
    header = (r" & & " + SHORT_HEADER + r" \cmidrule(lr){3-4} \cmidrule(lr){5-6}" "\n"
              r"Student & Params & AUC & $1/\epsilon_B$ & AUC & $1/\epsilon_B$")
    return wrap("lrlrlr", header, rows)


def long_rows(entries: list[tuple[str, str, dict]]) -> list[str]:
    """Per entry: no teacher and each scaling teacher, with AUC and both rejections."""
    rows = []
    for a, b, cells in entries:
        for i, t in enumerate((None, *SCALING_TEACHERS)):
            c = cells(t)
            if c is not None:
                rows.append(" & ".join([a if i == 0 else "", b if i == 0 else "",
                                        "no teacher" if t is None else name(t), auc_runs(c),
                                        rej(c, "rej50"), rej(c, "rej30")]))
        rows.append(r"\midrule")
    return rows[:-1]


def depth_full(r: Results, base: str) -> str:
    entries = [(str(n), num(params(m)), lambda t, m=m: r.run(m, t)) for n, m in depth_models(base)]
    return wrap("lrlrrr", "Blocks & Params & Teacher & AUC & " + REJ, long_rows(entries))


def size_full(r: Results) -> str:
    entries = [(name(m), num(params(m)), lambda t, m=m: r.cell(m, t))
               for f in WIDTH_FAMILIES for m in width_models(f)]
    return wrap("lrlrrr", "Student & Params & Teacher & AUC & " + REJ, long_rows(entries))


def recipes() -> str:
    entries = [(name(s), recipe_for(s)) for s in MATRIX_STUDENTS]
    entries += [(f"Width study, {sz}", RECIPES[f"size_{sz}"]) for sz in SIZES]
    rows = [f"{label} & {rc.optimizer} & {sci(rc.lr)} & {sci(rc.weight_decay)} & {num(rc.batch_size)} & "
            f"{num(rc.iterations)} & {SCHEDULES[rc.scheduler]}" for label, rc in entries]
    header = "Model & Optimizer & Learning rate & Weight decay & Batch & Steps & Schedule"
    return wrap("lllllrl", header, rows)


def complete(r: Results) -> str:
    rows = []
    for s in MATRIX_STUDENTS:
        if rows:
            rows.append(r"\midrule")
        for i, t in enumerate((None, *TEACHERS)):
            c = r.cell(s, t)
            if c is not None:
                rows.append(f"{name(s) if i == 0 else ''} & {'no teacher' if t is None else name(t)} & "
                            f"{auc(c)} & {rej(c, 'rej50')} & {rej(c, 'rej30')} & {len(c['seeds'])}")
    return wrap("lllrrc", "Student & Teacher & AUC & " + REJ + " & Runs", rows)


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
    rows = [r"\multicolumn{6}{l}{\textit{Teachers}}",
            row("ParT", p["embed_dims"][-1], p["num_heads"], f"{p['num_layers']} + {p['num_cls_layers']}",
                "class token", "part"),
            mipart("mipart"),
            row(name("lgatr"), f"{lg['hidden_mv_channels']} mv, {lg['hidden_s_channels']} s", lg["num_heads"],
                lg["num_blocks"], "global token", "lgatr"),
            slim("lgatr_slim"),
            row("OmniLearned-S", 128, 8, "8 + 2", "4 class tokens", "omnilearned_s"),
            r"ParT (pretrained) & \multicolumn{4}{l}{as ParT, built by the ParT authors' code} & "
            + num(params("part_pretrained")),
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
    return wrap("llllll", "Model & Width & Heads & Blocks & Read-out & Params", rows)


# Deep Sets students distilled from OmniLearned-L, arXiv:2609.26899 Table 1: AUC, 1/eps_B at 0.5 and 0.3
DEEPSETS = [("Deep Sets", "deepsets", 0.9800, 199, 748),
          ("Deep Sets, message passing", "deepsets_mp", 0.9859, 450, 1980)]


def operations(r: Results) -> str:
    """Operations per jet (median test jet) against performance: the Deep Sets students and ours."""
    flops = r.flops["models"]

    def row(label, size, c, m):
        return (f"{label} & {size} & {c['auc']:.5f} & {num(c['rej50'])} & {num(c['rej30'])} & "
                f"{flops[m]['flops_median'] / 1e6:.1f}")

    rows = [r"\multicolumn{6}{l}{\textit{Benyas et al., distilled from OmniLearned-L}}"]
    rows += [f"{label} & {num(flops[m]['parameters'] / 1e3)}k & {a:.4f} & {num(r50)} & {num(r30)} & "
             f"{flops[m]['flops_median'] / 1e6:.1f}" for label, m, a, r50, r30 in DEEPSETS]
    rows += [r"\midrule", r"\multicolumn{6}{l}{\textit{Ours, distilled from pretrained ParT}}"]
    for sz in reversed(SIZES):   # smallest first
        for f in WIDTH_FAMILIES:
            m = f"{f}_deep_{sz}"
            rows.append(row(name(m), thousands(m), r.cell(m, "part_pretrained"), m))
    for s in ("transformer_200k", "lgatr_slim_200k"):
        rows.append(row(name(s), thousands(s), r.cell(s, "part_pretrained"), s))
    rows += [r"\midrule", r"\multicolumn{6}{l}{\textit{Our teachers trained from scratch}}"]
    rows += [row(name(t), millions(t), r.teacher(t), t) for t in ("part", "mipart", "lgatr", "lgatr_slim")]
    return wrap("lrlrrr", "Model & Params & AUC & " + REJ + r" & MFLOP/jet", rows)


TABLES = {
    "teachers": teachers, "students": students, "matrix_auc": matrix_auc, "matrix_rej": matrix_rej,
    "gains": gains, "depth": depth, "size": size,
    "depth_transformer": lambda r: depth_full(r, "transformer_200k"),
    "depth_slim": lambda r: depth_full(r, "lgatr_slim_200k"), "size_full": size_full,
    "recipes": lambda r: recipes(), "complete": complete, "architectures": lambda r: architectures(),
    "operations": operations,
}
NEEDS_FLOPS = ("operations",)


def tables(summary: dict, flops: dict | None = None) -> dict[str, str]:
    r = Results(summary, flops)
    return {name: make(r) for name, make in TABLES.items() if flops or name not in NEEDS_FLOPS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--summary", default="results/summary.json")
    ap.add_argument("--flops", default="results/flops.json", help="from scripts/count_flops.py")
    ap.add_argument("--out", default="tables")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    flops = json.loads(Path(args.flops).read_text()) if Path(args.flops).is_file() else None
    written = tables(json.loads(Path(args.summary).read_text()), flops)
    for t, tex in written.items():
        (out / f"{t}.tex").write_text("% generated by scripts/make_tables.py; do not edit\n" + tex)
    print(f"wrote {len(written)} tables to {out}")


if __name__ == "__main__":
    main()
