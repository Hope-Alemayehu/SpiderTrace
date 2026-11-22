"""Phase 3, S2 details (training recipe and run-to-run nondeterminism).

    python analysis/phase3_s2_details.py  ->  analysis/phase3_results/s2_details.json

Inputs:
  C0_16   results/run_20261116T154357Z_d5_p0.003.json  (S1 C0, commit 5110582, constant lr)
  C2_16   results/run_20261116T183801Z_d5_p0.003.json  (S1 C2, commit 5110582, --readout fired)
  C0_17   results/run_20261117T044139Z_d5_p0.003.json  (C0 rerun, same commit and flags as C0_16;
          JSON recovered from the Kaggle notebook output)
  REC     results/run_20261117T095238Z_d5_p0.003.json  (C0_recipe, commit 9b2a066,
          --lr-schedule cosine --lr-min 1e-5 --grad-clip 1.0)
  C2_17   analysis/kaggle_logs/c2_rerun_log_20261117T072706Z.txt  (C2 rerun, LOG ONLY: the JSON
          was deleted on Kaggle, so no test_wrong_idx and no paired statistics)

McNemar: exact binomial on test_wrong_idx; b = shots the reference gets wrong and the
other run gets right, c = the reverse. Spike: epoch-to-epoch ratio above 1.3 (val; train
reported too). Epochs are 1-based. Every epoch has the same number of steps, so the
run-wide clipped fraction is the mean of the per-epoch fractions.
"""
import json
import math
import re
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
RUNS = {"C0_16": "results/run_20261116T154357Z_d5_p0.003.json",
        "C2_16": "results/run_20261116T183801Z_d5_p0.003.json",
        "C0_17": "results/run_20261117T044139Z_d5_p0.003.json",
        "REC": "results/run_20261117T095238Z_d5_p0.003.json"}
C2_LOG = "analysis/kaggle_logs/c2_rerun_log_20261117T072706Z.txt"
SPIKE = 1.3
OUT = ROOT / "analysis" / "phase3_results" / "s2_details.json"


def mcnemar(w_ref, w_other):
    b, c = len(w_ref - w_other), len(w_other - w_ref)
    return {"fixed": b, "newly_wrong": c, "wrong_in_both": len(w_ref & w_other),
            "p": binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0}


def spikes(xs):
    return [{"epoch": i + 1, "from": xs[i - 1], "to": xs[i], "ratio": xs[i] / xs[i - 1]}
            for i in range(1, len(xs)) if xs[i] / xs[i - 1] > SPIKE]


def cosine_lr(epoch, total, lr, lr_min):
    """lr in a 1-based epoch for a per-epoch cosine from lr to lr_min over `total` epochs."""
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * (epoch - 1) / total))


def train_summary(t):
    tl, vl, be = t["train_loss"], t["val_loss"], t["best_epoch"]
    s = {"epochs_run": t["epochs_run"], "best_epoch": be, "wall_time_s": t.get("wall_time_s"),
         "epoch1_val": vl[0], "epoch1_train": tl[0],
         "best_val": vl[be - 1], "best_train": tl[be - 1],
         "spikes_val": spikes(vl), "spikes_train": spikes(tl)}
    if "lr" in t:
        s["lr_first"], s["lr_at_best"], s["lr_last"] = t["lr"][0], t["lr"][be - 1], t["lr"][-1]
    if "grad_norm_mean" in t:
        g, gm, cf = t["grad_norm_mean"], t["grad_norm_max"], t["clipped_frac"]
        s["grad"] = {"epoch1_mean": g[0], "epoch1_max": gm[0], "epoch1_clipped_frac": cf[0],
                     "mean_of_epoch_means_from_2": sum(g[1:]) / len(g[1:]),
                     "max_overall": max(gm), "max_overall_epoch": gm.index(max(gm)) + 1,
                     "max_from_epoch_2": max(gm[1:]),
                     "clipped_frac_run": sum(cf) / len(cf),
                     "clipped_frac_from_2": sum(cf[1:]) / len(cf[1:]),
                     "clipped_frac_last": cf[-1], "grad_norm_mean_last": g[-1]}
    else:
        s["grad"] = None
    return s


def parse_c2_log(path):
    text = (ROOT / path).read_text(encoding="utf-8")
    seeds, cur = [], None
    for line in text.splitlines():
        m = re.search(r"epoch\s+(\d+)/\d+\s+train_loss=([\d.]+)\s+val_loss=([\d.]+)", line)
        if m:
            if int(m.group(1)) == 1:
                cur = {"train_loss": [], "val_loss": []}
                seeds.append(cur)
            cur["train_loss"].append(float(m.group(2)))
            cur["val_loss"].append(float(m.group(3)))
        m = re.search(r"early stop at epoch (\d+)", line)
        if m:
            cur["early_stop_epoch"] = int(m.group(1))
    out = {"note": "from the printed log only (losses rounded to 4 decimals); JSON deleted",
           "seeds": {}}
    for i, s in enumerate(seeds):
        vl = s["val_loss"]
        be = vl.index(min(vl)) + 1
        out["seeds"][str(i)] = {"epochs_run": len(vl), "early_stop_epoch": s.get("early_stop_epoch"),
                                "best_epoch_from_log": be, "epoch1_val": vl[0],
                                "epoch1_train": s["train_loss"][0], "best_val": vl[be - 1],
                                "spikes_val": spikes(vl), "spikes_train": spikes(s["train_loss"])}
    m = re.search(r"\[A\] test LER = ([\d.]+) \+/- ([\d.]+) over (\d+) seeds", text)
    out["primary_test_ler_mean"], out["primary_test_ler_pm"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"\[McNemar A vs MWPM, seed 0\] p = ([\deE.+-]+)", text)
    out["primary_mcnemar_vs_mwpm_seed0_p"] = float(m.group(1))
    m = re.search(r"\[big test\] \[A\] LER = ([\d.]+) \((\d+) errors\), seed 0", text)
    out["big_seed0_ler"], out["big_seed0_errors"] = float(m.group(1)), int(m.group(2))
    m = re.search(r"\[McNemar A vs MWPM, big test, seed 0\] p = ([\deE.+-]+)", text)
    out["big_mcnemar_vs_mwpm_seed0_p"] = float(m.group(1))
    m = re.search(r"\[MWPM\] test LER = [\d.]+ \((\d+) errors\)", text)
    out["mwpm_primary_errors"] = int(m.group(1))
    m = re.search(r"MWPM LER = [\d.]+ \((\d+) errors\)", text)
    out["mwpm_big_errors"] = int(m.group(1))
    out["wrote"] = re.search(r"Wrote (\S+)", text).group(1)
    return out


data = {k: json.loads((ROOT / v).read_text(encoding="utf-8")) for k, v in RUNS.items()}
ref = data["C0_16"]
out = {"files": dict(RUNS, C2_17=C2_LOG), "runs": {}}
for k, r in data.items():
    pv, cfg = r["provenance"], r["provenance"]["config"]
    assert r["mwpm"]["test_wrong_idx"] == ref["mwpm"]["test_wrong_idx"], k
    assert r["big_test"]["mwpm"]["test_wrong_idx"] == ref["big_test"]["mwpm"]["test_wrong_idx"], k
    rec = {"commit": pv["git_commit"][:7], "dirty": pv["git_dirty"], "timestamp_utc": pv["timestamp_utc"],
           "outdir": cfg["outdir"], "lr_schedule": cfg["lr_schedule"], "lr_min": cfg["lr_min"],
           "grad_clip": cfg["grad_clip"], "readout": cfg["readout"], "epochs_cap": cfg["epochs"],
           "torch": pv["library_versions"].get("torch"), "seeds": {}}
    for sel, grid in zip(r["arms"]["A"]["selected_per_seed"], r["arms"]["A"]["lambda_grid"]):
        s = str(sel["seed"])
        bt = r["big_test"]["arms"]["A"][s]
        ts = train_summary(grid["train"])
        if cfg["lr_schedule"] == "cosine":
            ts["cosine_check_lr_last"] = cosine_lr(ts["epochs_run"], cfg["epochs"], cfg["lr"], cfg["lr_min"])
            ts["lr_last_over_lr0"] = ts["lr_last"] / cfg["lr"]
        rec["seeds"][s] = {"primary_errors": len(sel["test_wrong_idx"]), "primary_ler": sel["test_ler"],
                           "val_ler": sel["val_ler"], "big_errors": bt["n_errors"], "big_ler": bt["test_ler"],
                           "big_vs_mwpm": r["big_test"]["mcnemar_a_vs_mwpm"][s],
                           "primary_vs_mwpm": r["mcnemar_a_vs_mwpm"][s],
                           "zero_fired_big": bt.get("zero_fired"), "train": ts}
    out["runs"][k] = rec


def wrong(k, s, where):
    r = data[k]
    if where == "big":
        return set(r["big_test"]["arms"]["A"][s]["test_wrong_idx"])
    return set(next(x for x in r["arms"]["A"]["selected_per_seed"] if str(x["seed"]) == s)["test_wrong_idx"])


pairs = {"REC_vs_C0_16": ("C0_16", "REC"), "REC_vs_C0_17": ("C0_17", "REC"),
         "C0_17_vs_C0_16": ("C0_16", "C0_17")}
out["paired"] = {}
for name, (a, b) in pairs.items():
    out["paired"][name] = {s: {w: mcnemar(wrong(a, s, w), wrong(b, s, w)) for w in ("big", "primary")}
                           for s in ("0", "1")}
out["seed_spread_big"] = {k: mcnemar(wrong(k, "0", "big"), wrong(k, "1", "big")) for k in data}
out["C2_17_partial"] = parse_c2_log(C2_LOG)
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
