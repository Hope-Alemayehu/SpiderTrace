"""
plot_results.py  —  visualize QEC decoder results across d=3, d=5, d=7

Usage:
    python plot_results.py                  # saves all figures to figures/
    python plot_results.py --show           # also opens interactive windows
    python plot_results.py --results-dir results  # custom results directory

Outputs:
    figures/ler_curves.pdf      — LER vs p for all models, one subplot per distance
    figures/gap_analysis.pdf    — relative LER improvement of Proposed over BaselineA
    figures/lambda_sweep.pdf    — best val loss per lambda value at each distance
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Style ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.5,
})

COLORS = {
    "BaselineA":    "#378ADD",
    "ProposedModel": "#1D9E75",
    "BaselineB":    "#D85A30",
    "MWPM":         "#888780",
}

MARKERS = {
    "BaselineA":    "o",
    "ProposedModel": "s",
    "BaselineB":    "^",
    "MWPM":         "D",
}

LINESTYLES = {
    "BaselineA":    "-",
    "ProposedModel": "--",
    "BaselineB":    "-",
    "MWPM":         ":",
}

LABELS = {
    "BaselineA":    "Baseline A (syndrome only)",
    "ProposedModel": "Proposed (ZX-distilled)",
    "BaselineB":    "Baseline B (ZX at inference)",
    "MWPM":         "MWPM",
}

P_VALUES = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]
DISTANCES = [3, 5, 7]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(results_dir: str) -> dict:
    """Load all result JSON files from results_dir."""
    rdir = Path(results_dir)
    results = {}
    for d in DISTANCES:
        path = rdir / f"d{d}_results.json"
        if path.exists():
            with open(path) as f:
                results[d] = json.load(f)
        else:
            print(f"Warning: {path} not found — skipping d={d}")

        mwpm_path = rdir / f"d{d}_mwpm.json"
        if mwpm_path.exists():
            with open(mwpm_path) as f:
                mwpm = json.load(f)
            if d not in results:
                results[d] = {}
            results[d]["MWPM"] = mwpm
    return results


def extract_ler(results: dict, d: int, model: str) -> tuple[list, list]:
    """Return (p_values, ler_values) for a given distance and model."""
    if d not in results:
        return [], []
    data = results[d]

    if model == "MWPM":
        if "MWPM" not in data:
            return [], []
        model_data = data["MWPM"]
    else:
        if model not in data:
            return [], []
        model_data = data[model]

    ps, lers = [], []
    for p_str in sorted(model_data.keys(), key=float):
        try:
            p = float(p_str)
            ler = model_data[p_str]["logical_error_rate"]
            ps.append(p)
            lers.append(max(ler, 1e-6))
        except (KeyError, ValueError):
            continue
    return ps, lers


# ── Figure 1: LER curves ──────────────────────────────────────────────────────

def plot_ler_curves(results: dict, save_path: Path, show: bool):
    n_distances = sum(1 for d in DISTANCES if d in results)
    if n_distances == 0:
        print("No results to plot.")
        return

    fig, axes = plt.subplots(1, n_distances, figsize=(4.5 * n_distances, 4.2), sharey=False)
    if n_distances == 1:
        axes = [axes]

    models_to_plot = ["MWPM", "BaselineA", "ProposedModel", "BaselineB"]

    for ax, d in zip(axes, [d for d in DISTANCES if d in results]):
        for model in models_to_plot:
            ps, lers = extract_ler(results, d, model)
            if not ps:
                continue
            ax.plot(
                ps, lers,
                color=COLORS[model],
                linestyle=LINESTYLES[model],
                marker=MARKERS[model],
                markersize=5,
                linewidth=1.5,
                label=LABELS[model],
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("physical error rate p")
        ax.set_ylabel("logical error rate" if ax == axes[0] else "")
        ax.set_title(f"d = {d}")
        ax.set_xticks(P_VALUES)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.tick_params(axis="x", rotation=45)

        if ax == axes[-1]:
            ax.legend(loc="upper left", framealpha=0.9, edgecolor="none")

    fig.suptitle("Logical error rate vs physical error rate", y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ── Figure 2: Gap analysis ────────────────────────────────────────────────────

def plot_gap_analysis(results: dict, save_path: Path, show: bool):
    p_subset = [0.01, 0.03, 0.05, 0.1]
    p_labels  = ["p=0.01", "p=0.03", "p=0.05", "p=0.1"]

    available = [d for d in DISTANCES if d in results]
    if not available:
        return

    x = np.arange(len(p_subset))
    width = 0.25
    offsets = np.linspace(-(len(available)-1)/2, (len(available)-1)/2, len(available)) * width

    fig, ax = plt.subplots(figsize=(8, 4))

    dist_colors = {3: "#B5D4F4", 5: "#1D9E75", 7: "#F09595"}

    for offset, d in zip(offsets, available):
        gaps = []
        for p in p_subset:
            p_str = str(p)
            try:
                ler_a = results[d]["BaselineA"][p_str]["logical_error_rate"]
                ler_p = results[d]["ProposedModel"][p_str]["logical_error_rate"]
                if ler_a == 0:
                    gaps.append(0.0)
                else:
                    gaps.append((ler_p - ler_a) / ler_a * 100)
            except (KeyError, TypeError):
                gaps.append(0.0)

        bar_colors = ["#1D9E75" if g < 0 else "#E24B4A" for g in gaps]
        bars = ax.bar(x + offset, gaps, width * 0.85, color=bar_colors,
                      alpha=0.85, label=f"d={d}")

        for bar, gap in zip(bars, gaps):
            ypos = bar.get_height() + 0.3 if gap >= 0 else bar.get_height() - 1.5
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                    f"{gap:+.1f}%", ha="center", va="bottom",
                    fontsize=7.5, color="#444441")

    ax.axhline(0, color="#888780", linewidth=0.8, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(p_labels)
    ax.set_ylabel("relative LER change: (Proposed − BaselineA) / BaselineA (%)")
    ax.set_title("Gap analysis: ZX distillation improvement over syndrome-only baseline")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1D9E75", label="improvement (Proposed wins)"),
        Patch(facecolor="#E24B4A", label="regression (Proposed loses)"),
    ]
    dist_patches = [Patch(facecolor=dist_colors[d], label=f"d={d}") for d in available]
    ax.legend(handles=legend_elements + dist_patches,
              loc="lower left", framealpha=0.9, edgecolor="none", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ── Figure 3: Lambda sweep ────────────────────────────────────────────────────

def plot_lambda_sweep(results: dict, save_path: Path, show: bool):
    available = [d for d in DISTANCES if d in results
                 and "ProposedModel_lambda_sweep" in results[d]]
    if not available:
        print("No lambda sweep data found.")
        return

    lambdas = [0.01, 0.1, 0.5, 1.0]
    lambda_labels = ["0.01", "0.10", "0.50", "1.00"]

    x = np.arange(len(lambdas))
    width = 0.25
    offsets = np.linspace(-(len(available)-1)/2, (len(available)-1)/2, len(available)) * width

    dist_colors = {3: "#85B7EB", 5: "#5DCAA5", 7: "#EF9F27"}

    fig, ax = plt.subplots(figsize=(7, 4))

    for offset, d in zip(offsets, available):
        sweep = results[d]["ProposedModel_lambda_sweep"]
        val_losses = []
        for lam in lambdas:
            key = str(lam)
            try:
                val_losses.append(sweep[key]["best_val_loss"])
            except KeyError:
                val_losses.append(float("nan"))

        ax.bar(x + offset, val_losses, width * 0.85,
               color=dist_colors[d], label=f"d={d}", alpha=0.9)

        baseline_val = results[d].get("BaselineA_best_val_loss", None)

    ax.set_xticks(x)
    ax.set_xticklabels([f"λ = {l}" for l in lambda_labels])
    ax.set_ylabel("best validation loss")
    ax.set_title("Lambda sweep: ProposedModel auxiliary loss weight")
    ax.legend(framealpha=0.9, edgecolor="none")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ── Figure 4: Summary table heatmap ──────────────────────────────────────────

def plot_summary_table(results: dict, save_path: Path, show: bool):
    models = ["MWPM", "BaselineA", "ProposedModel", "BaselineB"]
    p_subset = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]
    p_labels = [str(p) for p in p_subset]

    available = [d for d in DISTANCES if d in results]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available),
                             figsize=(5 * len(available), 3.5))
    if len(available) == 1:
        axes = [axes]

    for ax, d in zip(axes, available):
        table_data = []
        row_labels = []
        for model in models:
            row = []
            has_data = False
            for p in p_subset:
                p_str = str(p)
                try:
                    if model == "MWPM":
                        ler = results[d]["MWPM"][p_str]["logical_error_rate"]
                    else:
                        ler = results[d][model][p_str]["logical_error_rate"]
                    row.append(ler)
                    has_data = True
                except (KeyError, TypeError):
                    row.append(float("nan"))
            if has_data:
                table_data.append(row)
                row_labels.append(LABELS.get(model, model))

        if not table_data:
            ax.axis("off")
            continue

        arr = np.array(table_data, dtype=float)
        arr_display = np.where(np.isnan(arr), np.nan, arr)

        im = ax.imshow(arr_display, aspect="auto", cmap="RdYlGn_r",
                       vmin=0, vmax=0.3)

        ax.set_xticks(range(len(p_labels)))
        ax.set_xticklabels(p_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_title(f"d = {d}", fontsize=11)

        for i in range(len(row_labels)):
            for j in range(len(p_labels)):
                val = arr_display[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                            fontsize=7, color="black" if val < 0.15 else "white")

        plt.colorbar(im, ax=ax, shrink=0.8, label="LER")

    fig.suptitle("Logical error rate summary (lower = better)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot QEC decoder results")
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing d*_results.json and d*_mwpm.json files")
    parser.add_argument("--figures-dir", default="figures",
                        help="Output directory for figures")
    parser.add_argument("--show", action="store_true",
                        help="Open interactive matplotlib windows after saving")
    args = parser.parse_args()

    Path(args.figures_dir).mkdir(exist_ok=True)
    results = load_results(args.results_dir)

    if not results:
        print("No result files found. Run train.py first.")
        raise SystemExit(1)

    fdir = Path(args.figures_dir)

    plot_ler_curves(results,    fdir / "ler_curves.pdf",    args.show)
    plot_gap_analysis(results,  fdir / "gap_analysis.pdf",  args.show)
    plot_lambda_sweep(results,  fdir / "lambda_sweep.pdf",  args.show)
    plot_summary_table(results, fdir / "summary_table.pdf", args.show)

    print(f"\nAll figures saved to {fdir}/")
    print("Files: ler_curves.pdf  gap_analysis.pdf  lambda_sweep.pdf  summary_table.pdf")