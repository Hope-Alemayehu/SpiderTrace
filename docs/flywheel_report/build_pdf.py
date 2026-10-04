"""Build the Flywheel report PDF.

    python docs/flywheel_report/build_pdf.py

1. Regenerates the figures (make_figures.py).
2. Fills the {{placeholders}} in report.md with values from report_data.py. Every
   number and every per-seed table is generated here, so adding a Kaggle Run 4
   JSON to results/ and rebuilding updates figures, tables and counts together.
3. Converts Markdown to HTML (report.html) and prints it to report.pdf with
   headless Chrome or Edge.
4. Fails if the PDF exceeds 4 pages, if any em or en dash appears in the output,
   or if a placeholder is left unfilled.
"""
from __future__ import annotations

import datetime as dt
import math
import os
import re
import subprocess
import sys

import markdown
from pypdf import PdfReader

import make_figures
from report_data import ARMS, collect, errors, precision_per_pauli, se

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_PAGES = 4
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]
LABEL = {"A": "GNN-A", "Raw": "GNN-Raw", "ZX": "GNN-ZX"}


# ---- formatting ----------------------------------------------------------------
def f4(x):
    return f"{x:.4f}"


def f3(x):
    return f"{x:.3f}"


def fp(p):
    """p-value as reported: 3 significant digits, scientific below 1e-3."""
    p = float(p)
    if p >= 0.9995:
        return "1.0"
    if p >= 0.01:
        return f"{p:.3f}"
    if p >= 0.001:
        return f"{p:.2g}"
    mant, exp = f"{p:.1e}".split("e")
    return f"{mant} &times; 10<sup>-{int(exp[1:])}</sup>"


def lam(x):
    return f"{x:g}"


def rng(xs):
    """'a' for one value, 'a to b' for several (3 decimals)."""
    lo, hi = min(xs), max(xs)
    return f3(lo) if f3(lo) == f3(hi) else f"{f3(lo)} to {f3(hi)}"


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def git_head():
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                                      text=True, stderr=subprocess.DEVNULL).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


# ---- values ----------------------------------------------------------------------
def values(D):
    V = {}
    r2, s2 = D["run2"], D["run2_summary"]
    r3, r4, s4 = D["run3"], D["run4"], D["run4_summary"]
    n = r4[0].n
    k4 = len(r4)

    V["date"] = dt.date.today().isoformat()
    V["head"] = git_head()

    # Section 1: test-set size and power.
    V["n_test"] = f"{n:,}"
    V["n_train"] = f"{50000 - 2 * n:,}"
    zx4 = s4["ZX"]["mean"]
    V["se_run4"] = f4(se(zx4, n))
    V["ci_run4"] = f4(1.96 * se(zx4, n))
    V["zx4_mean"] = f4(zx4)
    V["zx4_errors"] = str(round(zx4 * n))

    # Run 2 table: per-seed test LER with selected lambda, then mean +/- std.
    rows = []
    for a in ARMS:
        cells = [LABEL[a]]
        for r in r2:
            cells.append(f4(r.ler[a]) + ("" if a == "A" else f" <span class=lam>({lam(r.lam[a])})</span>"))
        cells.append(f"{f4(s2[a]['mean'])} &plusmn; {f4(s2[a]['std'])}")
        rows.append(cells)
    rows.append([f"MWPM (same {r2[0].n:,} shots)", "", "", "", f4(D["run2_mwpm"])])
    V["table_run2"] = table(["Arm"] + [f"Seed {r.seed}" for r in r2] + ["Mean &plusmn; std"],
                            rows, "num")
    mc_rows = [[f"Seed {r.seed}", r.mc_zx_raw["zx_only"], r.mc_zx_raw["other_only"],
                fp(r.mc_zx_raw["p"])] for r in r2]
    pool = D["run2_pooled"]
    mc_rows.append(["Pooled*", pool["zx_only"], pool["other_only"], fp(pool["p"])])
    V["table_run2_mc"] = table(["ZX vs Raw", "ZX right, Raw wrong", "Raw right, ZX wrong", "p"],
                               mc_rows, "num")
    V["run2_pool_p"] = fp(pool["p"])
    V["run2_mwpm"] = f4(D["run2_mwpm"])
    V["run2_mwpm_50k"] = f4(D["mwpm_d3_50k"]["logical_error_rate"])
    V["run2_flip"] = f"{D['mwpm_d3_50k']['observable_flip_rate']:.3f}"
    gaps = {a: s2[a]["mean"] - D["run2_mwpm"] for a in ARMS}
    se2 = se(statistics_mean([s2[a]["mean"] for a in ARMS]), r2[0].n)
    V["run2_se"] = f4(se2)
    V["run2_gap_lo"], V["run2_gap_hi"] = f4(min(gaps.values())), f4(max(gaps.values()))
    V["run2_gap_lo_se"] = f"{min(gaps.values()) / se2:.1f}"
    V["run2_gap_hi_se"] = f"{max(gaps.values()) / se2:.1f}"
    V["run2_arm_lo"] = f4(min(s2[a]["mean"] for a in ARMS))
    V["run2_arm_hi"] = f4(max(s2[a]["mean"] for a in ARMS))
    s1 = next(r for r in r2 if r.seed == 1)
    V["run2_s1_zx"], V["run2_s1_raw"] = s1.mc_zx_raw["zx_only"], s1.mc_zx_raw["other_only"]
    V["run2_s1_p"] = fp(s1.mc_zx_raw["p"])

    # Aux table: diagnostics (before/after) and every file-backed Run 4 seed.
    def aux_cells(row, a):
        x = row.aux[a]
        return [f3(1 - x["target_nonidentity_density"]), f3(x["frac_pred_identity"]),
                f3(x["aux_acc_nonidentity"])]

    dd, df = D["diag_default"], D["diag_fixed"]
    aux_rows = [
        [f"Diagnostic, plain CE, all {dd.extra['aux_slots']['Raw']} slots (d=3)"]
        + aux_cells(dd, "Raw") + aux_cells(dd, "ZX"),
        [f"Diagnostic, inv-freq CE, {df.extra['aux_slots']['Raw']} data qubits (d=3)"]
        + aux_cells(df, "Raw") + aux_cells(df, "ZX"),
    ]
    for r in r4:
        aux_rows.append([f"Run 4, Kaggle seed {r.seed}, {r.extra['aux_slots']['Raw']} data qubits (d=5)"]
                        + aux_cells(r, "Raw") + aux_cells(r, "ZX"))
    V["table_aux"] = table(
        ["Run", "Raw: target I", "Raw: pred. I", "Raw: non-I acc.",
         "ZX: target I", "ZX: pred. I", "ZX: non-I acc."], aux_rows, "num aux")
    V["diag_commit"] = dd.commit
    V["diag_shots"] = "4,000"
    V["diag_flip_ler"] = f4(dd.ler["A"])
    V["diag_collapsed"] = "true" if all(dd.aux[a]["collapsed_to_identity"] for a in ("Raw", "ZX")) else "false"
    V["diag_fixed_raw"], V["diag_fixed_zx"] = f3(df.aux["Raw"]["aux_acc_nonidentity"]), f3(df.aux["ZX"]["aux_acc_nonidentity"])
    V["diag_w_raw"] = " / ".join(f"{w:.2f}" for w in df.extra["aux_w"]["Raw"])
    s0 = r4[0]
    V["r4_w_raw"] = " / ".join(f"{w:.2f}" for w in s0.extra["aux_w"]["Raw"])
    V["r4_w_zx"] = " / ".join(f"{w:.2f}" for w in s0.extra["aux_w"]["ZX"])
    prec = {a: precision_per_pauli(s0.aux[a]) for a in ("Raw", "ZX")}
    for a in ("Raw", "ZX"):
        V[f"prec_{a.lower()}"] = " / ".join(f3(prec[a][q]) for q in "XYZ")
        V[f"r4s0_acc_{a.lower()}"] = f3(s0.aux[a]["aux_acc_nonidentity"])
        V[f"r4s0_predI_{a.lower()}"] = f3(s0.aux[a]["frac_pred_identity"])
        V[f"r4s0_tgtI_{a.lower()}"] = f3(1 - s0.aux[a]["target_nonidentity_density"])
    V["prec_seed_note"] = ("" if k4 == 1 else f" (Kaggle seed {s0.seed})")

    # Run 4 table: one row per file-backed Kaggle seed, plus a mean row if >= 2.
    rows = []
    for r in r4:
        rows.append([f"Kaggle seed {r.seed}", f4(r.ler["A"]),
                     f"{f4(r.ler['Raw'])} <span class=lam>({lam(r.lam['Raw'])})</span>",
                     f"{f4(r.ler['ZX'])} <span class=lam>({lam(r.lam['ZX'])})</span>",
                     f4(r.mwpm),
                     f"{r.mc_zx_raw['zx_only']} / {r.mc_zx_raw['other_only']}, p = {fp(r.mc_zx_raw['p'])}",
                     f"{r.mc_zx_mwpm['zx_only']} / {r.mc_zx_mwpm['other_only']}, p = {fp(r.mc_zx_mwpm['p'])}"])
    if k4 >= 2:
        rows.append([f"Mean &plusmn; std ({k4} seeds)"]
                    + [f"{f4(s4[a]['mean'])} &plusmn; {f4(s4[a]['std'])}" for a in ARMS]
                    + [f4(D["run4_mwpm"]), "", ""])
    V["table_run4"] = table(["", "GNN-A", "GNN-Raw (&lambda;)", "GNN-ZX (&lambda;)", "MWPM",
                             "ZX vs Raw (ZX-only / Raw-only)", "ZX vs MWPM (ZX-only / MWPM-only)"],
                            rows, "num run4")

    V["n_run4"] = str(k4)
    seeds = ", ".join(str(r.seed) for r in r4)
    V["run4_seeds"] = seeds
    missing = sorted({0, 1, 2} - {r.seed for r in r4})
    if k4 == 1:
        V["run4_seed_sentence"] = (
            f"Only Kaggle seed {seeds} is file-backed. "
            + (f"Kaggle seed{'s' if len(missing) > 1 else ''} {' and '.join(map(str, missing))} "
               f"{'are' if len(missing) > 1 else 'is'} being rerun as saved Kaggle versions. "
               if missing else "")
            + "Bars and lines therefore show a single seed with 95% binomial intervals; "
              "there is no mean over seeds yet.")
        V["run4_fig1_caption_seed"] = f"Kaggle seed {seeds}, the only file-backed seed"
        V["run4_fig2_caption_seed"] = f"Kaggle seed {seeds}"
    else:
        V["run4_seed_sentence"] = (
            f"{k4} Kaggle seeds are file-backed (seeds {seeds}). "
            + (f"Seed{'s' if len(missing) > 1 else ''} {' and '.join(map(str, missing))} "
               f"{'are' if len(missing) > 1 else 'is'} missing. " if missing else "")
            + "Figures show each seed and the mean over seeds.")
        V["run4_fig1_caption_seed"] = f"Kaggle seeds {seeds} (light bars) and their mean (solid bar)"
        V["run4_fig2_caption_seed"] = f"Kaggle seeds {seeds}: thin lines per seed, thick line is the mean"

    V["r4_mwpm"] = f4(D["run4_mwpm"])
    V["r4_mwpm_errors"] = str(errors(D["run4_mwpm"], n))
    lers = {a: s4[a]["mean"] for a in ARMS}
    V["r4_lo"], V["r4_hi"] = f4(min(lers.values())), f4(max(lers.values()))
    V["r4_spread"] = f4(max(lers.values()) - min(lers.values()))
    V["r4_ratio_lo"] = f"{min(lers.values()) / D['run4_mwpm']:.1f}"
    V["r4_ratio_hi"] = f"{max(lers.values()) / D['run4_mwpm']:.1f}"
    for a in ARMS:
        V[f"r4_{a.lower()}"] = f4(lers[a])
        V[f"r4_{a.lower()}_errors"] = str(errors(s0.ler[a], n))
    def paired(key, other):
        parts = [f"{getattr(r, key)['zx_only']} vs {getattr(r, key)['other_only']} discordant shots, "
                 f"p = {fp(getattr(r, key)['p'])}" + ("" if k4 == 1 else f" (seed {r.seed})")
                 for r in r4]
        head = f"The paired ZX vs {other} test gives " if k4 == 1 else f"Paired ZX vs {other} tests per seed give "
        return head + "; ".join(parts)
    V["r4_zr_sentence"] = paired("mc_zx_raw", "Raw")
    V["r4_zm_sentence"] = paired("mc_zx_mwpm", "MWPM")
    V["fig2_marks"] = ("Rings mark the &lambda; selected by validation LER. "
                       "Error bars are 95% binomial intervals." if k4 == 1 else
                       "Selected &lambda; per seed is listed in the Run 4 table.")
    V["r4_lam_raw"], V["r4_lam_zx"] = lam(s0.lam["Raw"]), lam(s0.lam["ZX"])
    lams = sorted(s0.grid["Raw"])
    for a in ("Raw", "ZX"):
        V[f"grid_{a.lower()}"] = " / ".join(f4(sum(r.grid[a][l] for r in r4) / k4) for l in lams)
    V["grid_lams"] = " / ".join(lam(l) for l in lams)
    V["grid_note"] = "" if k4 == 1 else f" (mean over {k4} seeds)"
    V["r3_a"], V["r3_raw"], V["r3_zx"] = f4(r3.ler["A"]), f4(r3.ler["Raw"]), f4(r3.ler["ZX"])
    V["r3_zm_p"] = fp(r3.mc_zx_mwpm["p"])
    V["a_shift"] = f4(abs(s0.ler["A"] - r3.ler["A"]))
    V["a_shift_seed"] = str(s0.seed)

    # One-line summaries of the per-seed paired tests, for the summary box.
    V["r4_zr_ps"] = ", ".join(fp(r.mc_zx_raw["p"]) for r in r4)
    zm_max = max(r.mc_zx_mwpm["p"] for r in r4)
    V["r4_zm_pstr"] = ("p = " if k4 == 1 else "every p &le; ") + fp(zm_max)
    V["r4_acc_raw_rng"] = rng([r.aux["Raw"]["aux_acc_nonidentity"] for r in r4])
    V["r4_acc_zx_rng"] = rng([r.aux["ZX"]["aux_acc_nonidentity"] for r in r4])

    # GNN-A vs aux arms across every full-budget run (Run 1, the 30-epoch smoke test,
    # is excluded). Salvaged rows come from LOG.md and are marked; nothing is pooled.
    arows = [(f"Run 2, seed {r.seed}", "d=3", "file", r.ler, r.n) for r in r2]
    arows.append((f"Run 3, seed {r3.seed} (default aux, unseeded init)", "d=5", "file", r3.ler, r3.n))
    arows += [(f"Run 4, Kaggle seed {r.seed}", "d=5", "file", r.ler, r.n) for r in r4]
    arows += [(f"Run 4, {s['label']}", "d=5", "salvaged", {a: s[a] for a in ARMS}, n)
              for s in D["salvaged"] if all(s[a] is not None for a in ARMS)]
    trows, a_hi, zmax, zwhere, except_ = [], 0, 0.0, "", []
    for label, dist, status, ler, nn in arows:
        best = min(ler["Raw"], ler["ZX"])
        hi = max(ler.values())
        tops = [a for a in ARMS if abs(ler[a] - hi) < 1e-12]
        top_label = " = ".join(LABEL[a] for a in tops) + (" (tie)" if len(tops) > 1 else "")
        if tops == ["A"]:
            a_hi += 1
        else:
            except_.append(label.split(" (")[0])
        for aux in ("Raw", "ZX"):
            z = abs(ler["A"] - ler[aux]) / math.sqrt(se(ler["A"], nn) ** 2 + se(ler[aux], nn) ** 2)
            if z > zmax:
                zmax, zwhere = z, label.split(" (")[0]
        cls = ' class="salv"' if status == "salvaged" else ""
        trows.append([f"<span{cls}>{label}</span>", dist,
                      "file-backed" if status == "file" else "<b>salvaged</b> (LOG.md)",
                      f4(ler["A"]), f4(ler["Raw"]), f4(ler["ZX"]),
                      f"{ler['A'] - best:+.4f}", top_label])
    V["table_a_vs_aux"] = table(["Run", "d", "Source", "GNN-A", "GNN-Raw", "GNN-ZX",
                                 "A minus best aux", "Highest LER"], trows, "num avsaux")
    n_file = sum(1 for r in arows if r[2] == "file")
    n_salv = len(arows) - n_file
    V["a_rows_n"], V["a_hi_n"] = str(len(arows)), str(a_hi)
    V["a_rows_detail"] = f"{n_file} file-backed, {n_salv} salvaged"
    V["a_exceptions"] = " and ".join(except_) if except_ else "none"
    V["a_zmax"], V["a_zwhere"] = f"{zmax:.1f}", zwhere

    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    counts = [len(r2), 1, k4]
    lo_c, hi_c = min(counts), max(counts)
    V["seed_range"] = (words.get(lo_c, str(lo_c)) if lo_c == hi_c
                       else f"{words.get(lo_c, lo_c)} to {words.get(hi_c, hi_c)}")

    V["r4_commits"] = ", ".join(sorted({r.commit for r in r4}))
    spread = max(lers.values()) - min(lers.values())
    V["gap_se_1e6"] = f"{spread / se(zx4, 10**6):.0f}"

    # Section 4: binomial standard errors from the formula, at the Run 4 GNN-ZX LER.
    for tag, nn in (("7500", n), ("1e5", 10**5), ("1e6", 10**6)):
        V[f"se_{tag}"] = f"{se(zx4, nn):.5f}" if nn > n else f4(se(zx4, nn))

    # Sources table.
    V["src_run4"] = "<br>".join(f"<code>{r.path}</code>" for r in r4)
    V["src_run2"] = "<br>".join(f"<code>{r.path}</code>" for r in r2)
    V["src_run3"] = f"<code>{r3.path}</code>"
    V["src_salv_note"] = (f"; salvaged rows: <code>results/LOG.md</code>" if n_salv else "")
    V["src_diag"] = f"<code>{dd.path}</code><br><code>{df.path}</code>"
    return V


def statistics_mean(xs):
    return sum(xs) / len(xs)


# ---- build -----------------------------------------------------------------------
def fill(template, V):
    def sub(m):
        key = m.group(1)
        if key not in V:
            raise KeyError(f"report.md uses unknown placeholder {{{{{key}}}}}")
        return str(V[key])
    return re.sub(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}", sub, template)


def browser():
    for b in BROWSERS:
        if os.path.exists(b):
            return b
    sys.exit("No Chrome or Edge found for printing to PDF.")


def main():
    make_figures.main()
    D = collect()
    V = values(D)
    with open(os.path.join(HERE, "report.md"), encoding="utf-8") as f:
        src = fill(f.read(), V)
    body = markdown.markdown(src, extensions=["tables", "md_in_html", "attr_list"])
    with open(os.path.join(HERE, "report.css"), encoding="utf-8") as f:
        css = f.read()
    html = ("<!doctype html><html lang=en><head><meta charset=utf-8>"
            "<title>SpiderTrace: Flywheel prototype report</title>"
            f"<style>{css}</style></head><body>{body}</body></html>")

    bad = [c for c in ("\u2014", "\u2013") if c in html]
    for svg in ("fig_run4_ler.svg", "fig_lambda.svg"):
        with open(os.path.join(HERE, svg), encoding="utf-8") as f:
            if any(c in f.read() for c in ("\u2014", "\u2013")):
                bad.append(svg)
    if bad:
        sys.exit(f"em/en dash found: {bad!r}")
    if "{{" in html:
        sys.exit("unfilled placeholder in output")

    html_path = os.path.join(HERE, "report.html")
    pdf_path = os.path.join(HERE, "report.pdf")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    subprocess.run([browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--run-all-compositor-stages-before-draw", "--virtual-time-budget=5000",
                    f"--print-to-pdf={pdf_path}", "file:///" + html_path.replace(os.sep, "/")],
                   check=True, capture_output=True)
    pages = len(PdfReader(pdf_path).pages)
    print(f"wrote {os.path.relpath(pdf_path)}: {pages} page(s), "
          f"{len(D['run4'])} file-backed Kaggle Run 4 seed(s)")
    if len(D["run4"]) > 1:
        print("NOTE: more than one Run 4 seed. Tables, figures and numbers are regenerated, "
              "but re-read the interpretive sentences in report.md sections 3.2 and 3.3 "
              "(lambda dependence, precision comparison) against the new data.")
    if pages > MAX_PAGES:
        sys.exit(f"PDF has {pages} pages; the limit is {MAX_PAGES}.")


if __name__ == "__main__":
    main()
