"""Figures for the Flywheel report, drawn from the run JSONs via report_data.

    python docs/flywheel_report/make_figures.py

Writes fig_run4_ler.svg and fig_lambda.svg next to this file. Only file-backed,
clean-tree Kaggle Run 4 seeds are plotted; salvaged numbers from LOG.md are never
plotted. With two or more seeds a mean bar / mean line is added automatically.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MultipleLocator  # noqa: E402

from report_data import ARMS, collect, se  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Reference categorical palette, slots 1 to 3 (validated all-pairs in light mode).
COLOR = {"A": "#2a78d6", "Raw": "#eb6834", "ZX": "#1baf7a"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LABEL = {"A": "GNN-A", "Raw": "GNN-Raw", "ZX": "GNN-ZX"}
Z95 = 1.96

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.linewidth": 0.6,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
})


def _axes(fig):
    ax = fig.add_subplot(111)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    return ax


def fig_run4_ler(D):
    rows, mwpm = D["run4"], D["run4_mwpm"]
    k = len(rows)
    nbars = k + (1 if k >= 2 else 0)
    gw = 0.7 if k == 1 else 0.86
    width = gw / nbars
    fig = plt.figure(figsize=(3.4, 2.35))
    ax = _axes(fig)
    ticks, ticklabels = [], []
    ymax = 0.0
    for gi, arm in enumerate(ARMS):
        x0 = gi - gw / 2 + width / 2
        top = 0.0
        for bi, r in enumerate(rows):
            x = x0 + bi * width
            ler = r.ler[arm]
            ci = Z95 * se(ler, r.n)
            ax.bar(x, ler, width * 0.9, color=COLOR[arm], alpha=1.0 if k == 1 else 0.45,
                   linewidth=0)
            ax.errorbar(x, ler, yerr=ci, fmt="none", ecolor=INK2, elinewidth=0.7, capsize=2)
            ymax = max(ymax, ler + ci)
            top = max(top, ler + ci)
            if k == 1:
                ax.text(x, ler + ci + 0.0004, f"{ler:.4f}", ha="center", va="bottom",
                        color=INK, fontsize=7)
            if k >= 2:
                ticks.append(x)
                ticklabels.append(f"{r.seed}")
        if k >= 2:
            x = x0 + k * width
            m = D["run4_summary"][arm]["mean"]
            ax.bar(x, m, width * 0.9, color=COLOR[arm], linewidth=0)
            ax.text(x, top + 0.0004, f"{m:.4f}", ha="center", va="bottom", color=INK, fontsize=6.5)
            ticks.append(x)
            ticklabels.append("mean")
    ax.axhline(mwpm, color=INK, linewidth=1.0, linestyle=(0, (4, 2)))
    ax.text(2 + gw / 2 + 0.06, mwpm + 0.0003, f"MWPM {mwpm:.4f}\n(same shots)", ha="left", va="bottom",
            color=INK, fontsize=6.5)
    if k >= 2:
        ax.set_xticks(ticks, ticklabels, fontsize=6)
        for gi, arm in enumerate(ARMS):
            ax.text(gi, -0.16, LABEL[arm], transform=ax.get_xaxis_transform(), ha="center",
                    va="top", color=INK, fontsize=7.5)
    else:
        ax.set_xticks(range(len(ARMS)), [LABEL[a] for a in ARMS], color=INK)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, 3.05)
    ax.set_ylim(0, ymax * 1.18)
    ax.yaxis.set_major_locator(MultipleLocator(0.005))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.set_ylabel(f"Test LER ({rows[0].n:,} shots)")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(HERE, "fig_run4_ler.svg"))
    plt.close(fig)


def fig_lambda(D):
    rows, mwpm = D["run4"], D["run4_mwpm"]
    k = len(rows)
    lams = sorted(rows[0].grid["Raw"])
    fig = plt.figure(figsize=(3.4, 2.35))
    ax = _axes(fig)
    ymax = 0.0
    ends = []
    for arm in ("Raw", "ZX"):
        per = [[r.grid[arm][l] for l in lams] for r in rows]
        if k >= 2:
            for ys in per:
                ax.plot(lams, ys, color=COLOR[arm], linewidth=0.7, alpha=0.4)
        ys = [sum(c) / k for c in zip(*per)]
        if k == 1:
            ci = [Z95 * se(y, rows[0].n) for y in ys]
            ax.errorbar(lams, ys, yerr=ci, fmt="none", ecolor=COLOR[arm], elinewidth=0.7,
                        capsize=2)
            ymax = max(ymax, max(y + c for y, c in zip(ys, ci)))
            sel = rows[0].lam[arm]
            ax.plot([sel], [rows[0].grid[arm][sel]], marker="o", markersize=8.5,
                    markerfacecolor="none", markeredgecolor=INK, markeredgewidth=0.8,
                    linestyle="none")
        ymax = max(ymax, max(max(p) for p in per))
        ax.plot(lams, ys, color=COLOR[arm], linewidth=1.5, marker="o", markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.6)
        ends.append((ys[-1], arm))
    # End labels, nudged apart so they never overlap.
    (lo, lo_arm), (hi, hi_arm) = sorted(ends)
    gap = max(0.0, 0.0018 - (hi - lo)) / 2
    for y, arm in ((lo - gap, lo_arm), (hi + gap, hi_arm)):
        ax.text(lams[-1] * 1.12, y, LABEL[arm], color=INK, va="center", fontsize=7)
    a_mean = D["run4_summary"]["A"]["mean"]
    ax.axhline(a_mean, color=COLOR["A"], linewidth=1.0, linestyle=(0, (4, 2)))
    ax.text(0.19, a_mean - 0.0005, f"GNN-A, no aux ({a_mean:.4f})", color=INK, fontsize=6.5,
            va="top")
    ax.axhline(mwpm, color=INK, linewidth=1.0, linestyle=(0, (1, 1.5)))
    ax.text(0.0105, mwpm + 0.0004, f"MWPM ({mwpm:.4f})", color=INK, fontsize=6.5, va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(lams, [f"{l:g}" for l in lams])
    ax.minorticks_off()
    ax.set_xlim(0.008, 2.3)
    ax.set_ylim(0, ymax * 1.12)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.set_xlabel("Aux loss weight λ")
    ax.set_ylabel(f"Test LER ({rows[0].n:,} shots)")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(HERE, "fig_lambda.svg"))
    plt.close(fig)


def main():
    D = collect()
    fig_run4_ler(D)
    fig_lambda(D)
    print(f"figures written from {len(D['run4'])} file-backed Kaggle Run 4 seed(s): "
          + ", ".join(r.path for r in D["run4"]))


if __name__ == "__main__":
    main()
