"""Phase 3, S1 details (C0 vs C2): everything the LOG entry needs, from the two JSONs.

    python analysis/phase3_s1_details.py  ->  analysis/phase3_results/s1_details.json

Primary-set McNemar C2 vs C0 is computed here from test_wrong_idx (same rule as
phase3_compare.py: b = C0 wrong and C2 right, c = the reverse, exact binomial).
A loss spike is an epoch-to-epoch ratio above 1.3 in train or val loss (as in LOG.md).
Epochs are 1-based (loss lists index 0 = epoch 1).
"""
import json
import math
from pathlib import Path

from scipy.stats import binomtest

RUNS = {"C0": "results/run_20261116T154357Z_d5_p0.003.json",
        "C2": "results/run_20261116T183801Z_d5_p0.003.json"}
SPIKE = 1.3
OUT = Path(__file__).resolve().parent / "phase3_results" / "s1_details.json"


def mcnemar(w_base, w_cfg):
    b, c = len(w_base - w_cfg), len(w_cfg - w_base)
    return {"fixed_vs_C0": b, "newly_wrong_vs_C0": c, "wrong_in_both": len(w_base & w_cfg),
            "p": binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0}


def spikes(xs):
    return [{"epoch": i + 1, "from": xs[i - 1], "to": xs[i], "ratio": xs[i] / xs[i - 1]}
            for i in range(1, len(xs)) if xs[i] / xs[i - 1] > SPIKE]


def first_below(xs, thr):
    return next((i + 1 for i, v in enumerate(xs) if v < thr), None)


data = {k: json.loads(Path(v).read_text()) for k, v in RUNS.items()}
out = {"files": RUNS}
for name, r in data.items():
    pv, cfg = r["provenance"], r["provenance"]["config"]
    a = r["arms"]["A"]
    rec = {"commit": pv["git_commit"], "dirty": pv["git_dirty"], "dirty_files": pv["git_dirty_files"],
           "timestamp_utc": pv["timestamp_utc"], "argv": pv["argv"], "hostname": pv["hostname"],
           "torch": pv["library_versions"].get("torch"), "resume": cfg["resume"],
           "readout": cfg["readout"], "n_params": a["model"]["n_params"],
           "splits": r["splits"], "mwpm_primary_errors": r["mwpm"]["n_errors"],
           "mwpm_big_errors": r["big_test"]["mwpm"]["n_errors"],
           "big_build_time_s": r["big_test"]["build_time_s"],
           "big_mwpm_decode_s": r["big_test"]["mwpm"].get("decode_time_s"), "seeds": {}}
    n_big = r["big_test"]["n_test"]
    n_pri = r["mwpm"]["n_test"]
    for sel, grid in zip(a["selected_per_seed"], a["lambda_grid"]):
        s = str(sel["seed"])
        t = grid["train"]
        tl, vl = t["train_loss"], t["val_loss"]
        be = t["best_epoch"]
        bt = r["big_test"]["arms"]["A"][s]
        ler = bt["test_ler"]
        rec["seeds"][s] = {
            "primary": {"errors": len(sel["test_wrong_idx"]), "test_ler": sel["test_ler"],
                        "val_ler": sel["val_ler"], "zero_fired": sel.get("zero_fired_test"),
                        "mcnemar_vs_mwpm": r["mcnemar_a_vs_mwpm"][s]},
            "big": {"errors": bt["n_errors"], "test_ler": ler,
                    "ler_se": math.sqrt(ler * (1 - ler) / n_big), "eval_time_s": bt["eval_time_s"],
                    "zero_fired": bt.get("zero_fired"),
                    "mcnemar_vs_mwpm": r["big_test"]["mcnemar_a_vs_mwpm"][s]},
            "train": {"epochs_run": t["epochs_run"], "best_epoch": be, "wall_time_s": t["wall_time_s"],
                      "s_per_epoch": t["wall_time_s"] / t["epochs_run"], "n_train": t["n_train"],
                      "lr_unique": sorted(set(t["lr"])),
                      "epoch1": {"train": tl[0], "val": vl[0]},
                      "best": {"train": tl[be - 1], "val": vl[be - 1]} if be else None,
                      "final": {"train": tl[-1], "val": vl[-1]},
                      "min_val": min(vl), "min_val_epoch": vl.index(min(vl)) + 1,
                      "val_first_10": vl[:10],
                      "val_at": {e: vl[e - 1] for e in (1, 2, 3, 5, 10, 20, 30, 40) if e <= len(vl)},
                      "first_val_below": {thr: first_below(vl, thr) for thr in (0.40, 0.05, 0.03, 0.02)},
                      "spikes_train": spikes(tl), "spikes_val": spikes(vl)}}
    rec["n_primary"], rec["n_big"] = n_pri, n_big
    out[name] = rec

out["C2_vs_C0"] = {}
for s in out["C0"]["seeds"]:
    sel = {k: next(x for x in data[k]["arms"]["A"]["selected_per_seed"] if str(x["seed"]) == s)
           for k in data}
    out["C2_vs_C0"][s] = {
        "primary": mcnemar(set(sel["C0"]["test_wrong_idx"]), set(sel["C2"]["test_wrong_idx"])),
        "big": mcnemar(set(data["C0"]["big_test"]["arms"]["A"][s]["test_wrong_idx"]),
                       set(data["C2"]["big_test"]["arms"]["A"][s]["test_wrong_idx"]))}
# Seed-to-seed disagreement of the same config, for scale.
for k in data:
    w = [set(data[k]["big_test"]["arms"]["A"][s]["test_wrong_idx"]) for s in ("0", "1")]
    out.setdefault("seed0_vs_seed1_big", {})[k] = {"only_seed0": len(w[0] - w[1]),
                                                   "only_seed1": len(w[1] - w[0]),
                                                   "both": len(w[0] & w[1])}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
