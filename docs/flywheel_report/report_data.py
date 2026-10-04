"""Single source of numbers for the Flywheel report.

Every value the report prints comes from this module. It reads the run JSONs in
results/ directly and selects runs by their recorded config, not by filename, so
new Kaggle seeds dropped into results/ are picked up on the next build.

The only hand-entered values are the ones that exist solely in results/LOG.md
(see LOG_* constants below); each names the LOG.md section it comes from.
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics
from dataclasses import dataclass, field

from scipy.stats import binomtest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "results")

ARMS = ("A", "Raw", "ZX")

# ---- values that exist only in results/LOG.md ---------------------------------
# LOG.md, "Run 2", MWPM row: recomputed after the run on the same 7,500 test shots
# (commit d640f39, make_datasets(3, 0.01, 50000, 12345) + mwpm_on_test).
LOG_RUN2_MWPM_SAME_SPLIT = 0.0545
# LOG.md, "Run 4: salvaged from printed output (JSON lost)". Text only, never pooled
# and never plotted.
LOG_SALVAGED = [
    {"label": "Kaggle seed 1", "A": 0.0123, "Raw": 0.0115, "ZX": 0.0113,
     "raw_aux": (0.5534, 0.440), "zx_aux": (0.6602, 0.643),
     "p_zx_raw": "1.0", "p_zx_mwpm": "8.396e-09"},
    {"label": "Colab seed 0", "A": 0.0131, "Raw": 0.0127, "ZX": 0.0121,
     "raw_aux": (0.7591, 0.681), "zx_aux": (0.6195, 0.600),
     "p_zx_raw": "0.5572", "p_zx_mwpm": "3.808e-10"},
    {"label": "Colab seed 1 (partial)", "A": None, "Raw": None, "ZX": 0.0125,
     "raw_aux": None, "zx_aux": (0.4535, 0.519),
     "p_zx_raw": "0.3915", "p_zx_mwpm": "7.864e-11"},
]

# make_datasets() in qec_run.py: test_frac = 0.15 of `shots`.
TEST_FRAC = 0.15


def rel(path):
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def load(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    d["_path"] = rel(path)
    d["_cfg"] = d["provenance"]["config"]
    return d


def n_test(run):
    if "mwpm" in run:
        return run["mwpm"]["n_test"]
    return int(round(TEST_FRAC * run["_cfg"]["shots"]))


def errors(ler, n):
    """Number of test errors behind an LER. Asserts it is an integer count."""
    k = ler * n
    assert abs(k - round(k)) < 1e-6, f"LER {ler} is not k/{n}"
    return int(round(k))


def se(ler, n):
    return math.sqrt(ler * (1 - ler) / n)


def mcnemar(run, key, seed):
    m = run.get(key, {}).get(str(seed))
    if m is None:
        return None
    return {"zx_only": m["discordant_a_right_b_wrong"],
            "other_only": m["discordant_a_wrong_b_right"],
            "p": m["p_value"]}


def per_seed(run, arm):
    """Selected-lambda results of one arm, one entry per model seed in the file."""
    return run["arms"][arm]["selected_per_seed"]


def is_clean(run):
    return run["provenance"].get("git_dirty") is False


def _match(cfg, **want):
    return all(cfg.get(k, dflt) == v for k, (v, dflt) in want.items())


NO_AUX_FLAGS = dict(aux_data_qubits_only=(False, False), aux_class_weight=("none", "none"))
AUX_FLAGS = dict(aux_data_qubits_only=(True, False), aux_class_weight=("inv-freq", "none"))


def select(runs, **want):
    return [r for r in runs if _match(r["_cfg"], **want)]


@dataclass
class SeedRow:
    seed: int
    path: str
    commit: str
    platform: str
    n: int
    ler: dict                    # arm -> test LER
    lam: dict                    # arm -> selected lambda
    val: dict                    # arm -> val LER
    grid: dict                   # arm -> {lambda: test LER}
    aux: dict                    # arm -> aux diagnostics (Raw/ZX) or None
    mwpm: float | None
    mc_zx_raw: dict | None
    mc_zx_mwpm: dict | None
    extra: dict = field(default_factory=dict)


def seed_rows(runs):
    rows = []
    for run in runs:
        seeds = [s["seed"] for s in per_seed(run, "A")]
        for sd in seeds:
            def pick(arm):
                return next(s for s in per_seed(run, arm) if s["seed"] == sd)
            grid = {arm: {g["lambda"]: g["test_ler"]
                          for g in run["arms"][arm]["lambda_grid"] if g["seed"] == sd}
                    for arm in ARMS}
            outdir = run["_cfg"].get("outdir", "")
            plat = ("Kaggle" if outdir.startswith("/kaggle")
                    else "Colab" if outdir.startswith("/content") else "local")
            rows.append(SeedRow(
                seed=sd, path=run["_path"], commit=run["provenance"]["git_commit"][:7],
                platform=plat, n=n_test(run),
                ler={a: pick(a)["test_ler"] for a in ARMS},
                lam={a: pick(a)["selected_lambda"] for a in ARMS},
                val={a: pick(a)["val_ler"] for a in ARMS},
                grid=grid,
                aux={a: pick(a).get("aux") for a in ARMS},
                mwpm=run.get("mwpm", {}).get("test_ler"),
                mc_zx_raw=mcnemar(run, "mcnemar_zx_vs_raw", sd),
                mc_zx_mwpm=mcnemar(run, "mcnemar_zx_vs_mwpm", sd),
                extra={"aux_freq": {a: run["arms"][a].get("aux_class_freq_train") for a in ARMS},
                       "aux_w": {a: run["arms"][a].get("aux_class_weights") for a in ARMS},
                       "aux_slots": {a: run["arms"][a].get("aux_slots") for a in ARMS}},
            ))
    rows.sort(key=lambda r: r.seed)
    seen = [r.seed for r in rows]
    assert len(seen) == len(set(seen)), f"duplicate model seeds across files: {seen}"
    for r in rows:
        for a in ARMS:
            errors(r.ler[a], r.n)
    return rows


def precision_per_pauli(aux):
    """Precision of X, Y, Z predictions from confusion_true_by_pred (rows=true, cols=pred)."""
    cm = aux["confusion_true_by_pred"]
    out = {}
    for j, name in enumerate("IXYZ"):
        if name == "I":
            continue
        col = sum(cm[i][j] for i in range(4))
        out[name] = cm[j][j] / col if col else float("nan")
    return out


def summarise(rows):
    out = {}
    for a in ARMS:
        v = [r.ler[a] for r in rows]
        out[a] = {"mean": statistics.fmean(v),
                  "std": statistics.stdev(v) if len(v) > 1 else None,
                  "per_seed": v}
    return out


def pooled_mcnemar(rows):
    zx = sum(r.mc_zx_raw["zx_only"] for r in rows)
    other = sum(r.mc_zx_raw["other_only"] for r in rows)
    return {"zx_only": zx, "other_only": other,
            "p": binomtest(zx, zx + other, 0.5).pvalue}


def collect():
    # Only files that recorded a clean git tree are used anywhere in the report.
    runs = [load(p) for p in sorted(glob.glob(os.path.join(RESULTS, "run_*.json")))]
    runs = [r for r in runs if is_clean(r)]
    diag = [load(p) for p in sorted(glob.glob(os.path.join(RESULTS, "diagnostic", "run_*.json")))]

    # Run 2: d=3, p=0.01, 50k shots, 100 epochs, lambdas {0.01, 0.1, 0.5}, default aux.
    run2 = select(runs, d=(3, None), p=(0.01, None), shots=(50000, None), epochs=(100, None),
                  lambdas=([0.01, 0.1, 0.5], None), **NO_AUX_FLAGS)
    # Run 3: d=5, p=0.003, default aux (64 slots, plain CE). Context only.
    run3 = select(runs, d=(5, None), p=(0.003, None), shots=(50000, None), epochs=(100, None),
                  lambdas=([0.01, 0.1, 0.5, 1.0], None), **NO_AUX_FLAGS)
    # Run 4: same as Run 3 plus both aux flags; Kaggle only.
    run4_all = select(runs, d=(5, None), p=(0.003, None), shots=(50000, None), epochs=(100, None),
                      lambdas=([0.01, 0.1, 0.5, 1.0], None), **AUX_FLAGS)
    run4 = [r for r in run4_all if r["_cfg"].get("outdir", "").startswith("/kaggle")]
    run4_skipped = [r["_path"] for r in run4_all if r not in run4]

    # Short CPU diagnostics: d=3, 4k shots, 25 epochs, lambda 0.1. Clean tree only.
    diag = [r for r in diag if is_clean(r)]
    diag_default = select(diag, **NO_AUX_FLAGS)
    diag_fixed = select(diag, **AUX_FLAGS)
    assert len(diag_default) == 1 and len(diag_fixed) == 1, "expected one diagnostic of each kind"

    r2, r3, r4 = seed_rows(run2), seed_rows(run3), seed_rows(run4)
    assert [r.seed for r in r2] == [0, 1, 2], [r.seed for r in r2]
    assert len(r3) == 1 and len(r4) >= 1

    mwpm4 = {r.mwpm for r in r4}
    assert len(mwpm4) == 1, f"Run 4 files disagree on MWPM: {mwpm4}"

    with open(os.path.join(RESULTS, "mwpm_d3_p0.01.json"), encoding="utf-8") as f:
        mwpm_d3_50k = json.load(f)["results"]["3"]["0.01"]

    return {
        "run2": r2, "run2_summary": summarise(r2), "run2_pooled": pooled_mcnemar(r2),
        "run2_mwpm": LOG_RUN2_MWPM_SAME_SPLIT,
        "mwpm_d3_50k": mwpm_d3_50k,
        "run3": r3[0],
        "run4": r4, "run4_summary": summarise(r4), "run4_mwpm": mwpm4.pop(),
        "run4_skipped": run4_skipped,
        "diag_default": seed_rows(diag_default)[0],
        "diag_fixed": seed_rows(diag_fixed)[0],
        "salvaged": LOG_SALVAGED,
    }


if __name__ == "__main__":
    D = collect()
    for r in D["run4"]:
        print("Run 4 file-backed:", r.path, r.platform, "seed", r.seed, r.commit, r.ler)
    print("Run 4 skipped (non-Kaggle):", D["run4_skipped"])
    print("Run 2 pooled McNemar:", D["run2_pooled"])
