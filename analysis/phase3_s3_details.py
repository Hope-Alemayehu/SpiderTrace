"""Phase 3, S3 details (recipe v2: C0 and C4).

    python analysis/phase3_s3_details.py  ->  analysis/phase3_results/s3_details.json

Inputs:
  C0v2   results/run_20261122T202714Z_d5_p0.003.json  (C0, recipe v2, commit 76da5a7)
  C4v2   results/run_20261123T003558Z_d5_p0.003.json  (C4 = --residual --norm layer, recipe v2,
         commit 76da5a7)
  REC    results/run_20261117T095238Z_d5_p0.003.json  (S2 C0_recipe, commit 9b2a066)
  RUN11  results/run_20261019T100602Z_d5_p0.003.json  (Run 11, 1M shots, constant lr, single run)
Recipe v2: cosine over 60 epochs (lr 1e-3 to lr_min 1e-5), --grad-clip 1.0, patience 60,
3 seeds. Here "seed" is the training seed; the three seeds of a config are three runs.

Per-seed McNemar: exact binomial on test_wrong_idx; b = reference wrong and other right
(fixed), c = the reverse (newly wrong). Cross-run comparison: the 3 C4 big-test error counts
against the 3 C0 counts as independent samples (exact Mann-Whitney U and Welch t), plus all
9 cross pairs (each C4 run against each C0 run) by paired McNemar. Spread across runs: mean,
sample SD (n-1) and min-max. Spike: epoch-to-epoch ratio above 1.3. Epochs are 1-based.

Decision rule (from S3 on): config B beats config A only if every B run beats every A run on
the big test (3 runs each), and the mean gap is larger than the run-to-run range within either
config. Per-seed McNemar is reported, but it isn't decisive.
"""
import json
import math
import statistics as st
from itertools import product
from pathlib import Path

from scipy.stats import binomtest, mannwhitneyu, ttest_ind

ROOT = Path(__file__).resolve().parent.parent
RUNS = {"C0v2": "results/run_20261122T202714Z_d5_p0.003.json",
        "C4v2": "results/run_20261123T003558Z_d5_p0.003.json",
        "REC": "results/run_20261117T095238Z_d5_p0.003.json",
        "RUN11": "results/run_20261019T100602Z_d5_p0.003.json"}
SPIKE = 1.3
OUT = ROOT / "analysis" / "phase3_results" / "s3_details.json"


def mcnemar(w_ref, w_other):
    b, c = len(w_ref - w_other), len(w_other - w_ref)
    return {"fixed": b, "newly_wrong": c, "wrong_in_both": len(w_ref & w_other),
            "p": binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0}


def spikes(xs):
    return [{"epoch": i + 1, "from": xs[i - 1], "to": xs[i], "ratio": xs[i] / xs[i - 1]}
            for i in range(1, len(xs)) if xs[i] / xs[i - 1] > SPIKE]


def cosine_lr(epoch, total, lr, lr_min):
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * (epoch - 1) / total))


def spread(xs):
    return {"values": xs, "mean": st.mean(xs), "sd": st.stdev(xs) if len(xs) > 1 else None,
            "min": min(xs), "max": max(xs), "range": max(xs) - min(xs)}


data = {k: json.loads((ROOT / v).read_text(encoding="utf-8")) for k, v in RUNS.items()}
ref = data["C0v2"]
out = {"files": RUNS, "runs": {}}
for k, r in data.items():
    pv, cfg = r["provenance"], r["provenance"]["config"]
    assert r["mwpm"]["test_wrong_idx"] == ref["mwpm"]["test_wrong_idx"], k
    assert r["big_test"]["mwpm"]["test_wrong_idx"] == ref["big_test"]["mwpm"]["test_wrong_idx"], k
    a = r["arms"]["A"]
    rec = {"commit": pv["git_commit"][:7], "dirty": pv["git_dirty"], "timestamp_utc": pv["timestamp_utc"],
           "argv": pv["argv"], "torch": pv["library_versions"].get("torch"),
           "train_shots": cfg.get("train_shots"), "extra_train_seed": cfg.get("extra_train_seed"),
           "epochs_cap": cfg["epochs"], "patience": cfg.get("patience"), "lr_schedule": cfg.get("lr_schedule"),
           "lr_min": cfg.get("lr_min"), "grad_clip": cfg.get("grad_clip"), "residual": cfg.get("residual"),
           "norm": cfg.get("norm"), "n_params": a["model"]["n_params"], "resume": cfg.get("resume"),
           "mwpm_primary": r["mwpm"]["n_errors"], "mwpm_big": r["big_test"]["mwpm"]["n_errors"],
           "big_seed": r["big_test"].get("seed"), "seeds": {}}
    mwpm_big = r["big_test"]["mwpm"]["n_errors"]
    for sel, grid in zip(a["selected_per_seed"], a["lambda_grid"]):
        s = str(sel["seed"])
        bt = r["big_test"]["arms"]["A"][s]
        t = grid["train"]
        tl, vl, be = t["train_loss"], t["val_loss"], t["best_epoch"]
        tr = {"epochs_run": t["epochs_run"], "best_epoch": be, "wall_time_s": t.get("wall_time_s"),
              "epoch1_val": vl[0], "best_val": vl[be - 1], "best_train": tl[be - 1],
              "final_val": vl[-1], "spikes_val": spikes(vl), "spikes_train": spikes(tl)}
        if "lr" in t:
            tr.update(lr_first=t["lr"][0], lr_at_best=t["lr"][be - 1], lr_last=t["lr"][-1])
            if cfg.get("lr_schedule") == "cosine":
                tr["cosine_check_lr_last"] = cosine_lr(t["epochs_run"], cfg["epochs"], cfg["lr"], cfg["lr_min"])
        if "grad_norm_mean" in t:
            g, gm, cf = t["grad_norm_mean"], t["grad_norm_max"], t["clipped_frac"]
            tr["grad"] = {"epoch1_mean": g[0], "epoch1_max": gm[0], "epoch1_clipped_frac": cf[0],
                          "mean_of_epoch_means_from_2": st.mean(g[1:]), "max_from_epoch_2": max(gm[1:]),
                          "clipped_frac_from_2": st.mean(cf[1:]), "clipped_frac_run": st.mean(cf),
                          "clipped_frac_last": cf[-1]}
        rec["seeds"][s] = {"primary_errors": len(sel["test_wrong_idx"]), "primary_ler": sel["test_ler"],
                           "val_ler": sel["val_ler"], "primary_vs_mwpm": r["mcnemar_a_vs_mwpm"][s],
                           "big_errors": bt["n_errors"], "big_ler": bt["test_ler"],
                           "big_ratio_to_mwpm": bt["n_errors"] / mwpm_big,
                           "big_vs_mwpm": r["big_test"]["mcnemar_a_vs_mwpm"][s],
                           "zero_fired_big": bt.get("zero_fired"), "train": tr}
    seeds = rec["seeds"]
    rec["across_runs"] = {"big_errors": spread([v["big_errors"] for v in seeds.values()]),
                          "primary_errors": spread([v["primary_errors"] for v in seeds.values()])}
    rec["across_runs"]["big_ratio_mean"] = rec["across_runs"]["big_errors"]["mean"] / mwpm_big
    out["runs"][k] = rec


def wrong(k, s, where):
    r = data[k]
    if where == "big":
        return set(r["big_test"]["arms"]["A"][s]["test_wrong_idx"])
    return set(next(x for x in r["arms"]["A"]["selected_per_seed"] if str(x["seed"]) == s)["test_wrong_idx"])


def seeds_of(k):
    return [str(x["seed"]) for x in data[k]["arms"]["A"]["selected_per_seed"]]


# C4 vs C0 under recipe v2, same seed.
out["C4v2_vs_C0v2_same_seed"] = {s: {w: mcnemar(wrong("C0v2", s, w), wrong("C4v2", s, w))
                                     for w in ("big", "primary")} for s in seeds_of("C0v2")}

# Cross-run: all C4 runs vs all C0 runs.
cross = {}
for w, key in (("big", "big_errors"), ("primary", "primary_errors")):
    c0 = out["runs"]["C0v2"]["across_runs"][key]["values"]
    c4 = out["runs"]["C4v2"]["across_runs"][key]["values"]
    cross[w] = {"C0": c0, "C4": c4, "mean_diff_C4_minus_C0": st.mean(c4) - st.mean(c0),
                "max_C4": max(c4), "min_C0": min(c0), "every_C4_below_every_C0": max(c4) < min(c0),
                "mannwhitney_exact_two_sided_p": mannwhitneyu(c4, c0, alternative="two-sided", method="exact").pvalue,
                "mannwhitney_exact_one_sided_p": mannwhitneyu(c4, c0, alternative="less", method="exact").pvalue,
                "welch_t_two_sided_p": ttest_ind(c4, c0, equal_var=False).pvalue}
    pairs = {f"C0s{a}_vs_C4s{b}": mcnemar(wrong("C0v2", a, w), wrong("C4v2", b, w))
             for a, b in product(seeds_of("C0v2"), seeds_of("C4v2"))}
    cross[w]["all_9_pairs"] = pairs
    cross[w]["pairs_C4_fewer_and_p_below_0.05"] = sum(
        1 for v in pairs.values() if v["newly_wrong"] < v["fixed"] and v["p"] < 0.05)
    cross[w]["pairs_C4_more_and_p_below_0.05"] = sum(
        1 for v in pairs.values() if v["newly_wrong"] > v["fixed"] and v["p"] < 0.05)
out["C4v2_vs_C0v2_cross_run"] = cross


def decision(a, b):
    """Does config b beat config a under the S3 decision rule (big test)?"""
    ea = out["runs"][a]["across_runs"]["big_errors"]
    eb = out["runs"][b]["across_runs"]["big_errors"]
    every = eb["max"] < ea["min"]
    gap = ea["mean"] - eb["mean"]
    gap_ok = gap > max(ea["range"], eb["range"])
    return {"every_B_run_beats_every_A_run": every, "mean_gap": gap,
            "range_A": ea["range"], "range_B": eb["range"],
            "gap_exceeds_both_ranges": gap_ok, "B_beats_A": every and gap_ok}


out["decision_C4v2_beats_C0v2"] = decision("C0v2", "C4v2")

# C0 recipe v2 vs S2 C0_recipe, same seed (seeds 0 and 1 only in S2).
out["C0v2_vs_REC_same_seed"] = {s: {w: mcnemar(wrong("REC", s, w), wrong("C0v2", s, w))
                                    for w in ("big", "primary")} for s in seeds_of("REC")}

# Within-config disagreement between runs (big test), every pair of seeds.
out["seed_pairs_big"] = {k: {f"{a}_vs_{b}": mcnemar(wrong(k, a, "big"), wrong(k, b, "big"))
                             for i, a in enumerate(seeds_of(k)) for b in seeds_of(k)[i + 1:]}
                         for k in ("C0v2", "C4v2", "REC")}

# Context: each C4 run vs Run 11 (1M shots, constant lr, seed 0, single run).
out["C4v2_vs_RUN11_big"] = {s: mcnemar(wrong("RUN11", "0", "big"), wrong("C4v2", s, "big"))
                            for s in seeds_of("C4v2")}
out["C0v2_vs_RUN11_big"] = {s: mcnemar(wrong("RUN11", "0", "big"), wrong("C0v2", s, "big"))
                            for s in seeds_of("C0v2")}

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))
