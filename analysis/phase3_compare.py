"""Phase 3 analysis: each ablation config vs the baseline C0 on the 200k big test.

    python analysis/phase3_compare.py --runs C0=path.json C2=path.json C5=... \
        [--out analysis/phase3_results]

Each path is one qec_run.py JSON run with --seeds 0 1 --big-test-shots 200000.
Checks before comparing (any failure stops the script):
  * every run has the same settings except the architecture flags
    (norm, readout, head, residual, depth, boundary, sublattice_types) and outdir;
  * the same big test set (n, sampler seed, identical MWPM wrong-shot list) and
    the same primary test set (identical MWPM wrong-shot list);
  * the same seeds.
Per config and seed: GNN-A errors on the big test, ratio to MWPM, and an exact
binomial McNemar vs C0 at the same seed, from test_wrong_idx:
  b = shots C0 gets wrong and the config gets right, c = the reverse,
  p = two-sided binomial(min(b, c), b + c, 0.5).
Pooled over seeds: b and c summed, same test. The two seeds are scored on the SAME
shots, so the pooled test treats correlated pairs as independent and overstates
significance; the decision therefore uses the per-seed tests.
Decision rule (fixed before the runs): a change "beats C0" only if it has fewer
errors than C0 AND p < 0.05 at EVERY seed. Only such changes enter the combined
config.
"""
import argparse
import json
from pathlib import Path

from scipy.stats import binomtest

ARCH_KEYS = {"norm", "readout", "head", "residual", "depth", "boundary", "sublattice_types"}
FREE_KEYS = ARCH_KEYS | {"outdir", "resume", "device"}
ALPHA = 0.05


def load(path):
    r = json.loads(Path(path).read_text())
    if "big_test" not in r:
        raise SystemExit(f"{path}: no big_test block")
    return r


def mcnemar(w_base, w_cfg):
    b, c = len(w_base - w_cfg), len(w_cfg - w_base)
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="NAME=path.json, one must be C0")
    ap.add_argument("--out", default="analysis/phase3_results")
    a = ap.parse_args()
    runs = dict(x.split("=", 1) for x in a.runs)
    if "C0" not in runs:
        raise SystemExit("one run must be named C0")
    data = {k: load(v) for k, v in runs.items()}
    base = data["C0"]
    bcfg = base["provenance"]["config"]
    seeds = [str(s) for s in bcfg["seeds"]]
    bt0 = base["big_test"]
    for name, r in data.items():
        cfg = r["provenance"]["config"]
        diff = {k for k in set(cfg) | set(bcfg) if cfg.get(k) != bcfg.get(k)} - FREE_KEYS
        if diff:
            raise SystemExit(f"{name}: settings differ from C0 beyond the architecture: {sorted(diff)}")
        bt = r["big_test"]
        if (bt["n_test"], bt["seed"]) != (bt0["n_test"], bt0["seed"]) or \
                bt["mwpm"]["test_wrong_idx"] != bt0["mwpm"]["test_wrong_idx"]:
            raise SystemExit(f"{name}: different big test set from C0")
        if r["mwpm"]["test_wrong_idx"] != base["mwpm"]["test_wrong_idx"]:
            raise SystemExit(f"{name}: different primary test set from C0")
        if [str(s) for s in cfg["seeds"]] != seeds:
            raise SystemExit(f"{name}: different seeds from C0")

    mwpm_err = bt0["mwpm"]["n_errors"]
    rows, out = [], {"mwpm_big_test_errors": mwpm_err, "n_big_test": bt0["n_test"],
                     "seeds": seeds, "configs": {}}
    for name, r in data.items():
        cfg = r["provenance"]["config"]
        flags = {k: cfg.get(k) for k in sorted(ARCH_KEYS) if cfg.get(k) != bcfg.get(k)}
        rec = {"flags": flags, "commit": r["provenance"].get("git_commit"),
               "n_params": r["arms"]["A"]["model"]["n_params"], "per_seed": {}}
        pooled_b = pooled_c = 0
        wins = []
        for s in seeds:
            w = set(r["big_test"]["arms"]["A"][s]["test_wrong_idx"])
            w0 = set(base["big_test"]["arms"]["A"][s]["test_wrong_idx"])
            b, c, p = mcnemar(w0, w)
            pooled_b += b; pooled_c += c
            zf = r["big_test"]["arms"]["A"][s].get("zero_fired")
            ep = next(g["train"] for g in r["arms"]["A"]["lambda_grid"] if str(g["seed"]) == s)
            rec["per_seed"][s] = {"errors": len(w), "errors_C0": len(w0),
                                  "ratio_to_mwpm": len(w) / mwpm_err,
                                  "fixed_vs_C0": b, "newly_wrong_vs_C0": c, "p": p,
                                  "zero_fired": zf, "epochs_run": ep["epochs_run"],
                                  "best_epoch": ep["best_epoch"], "wall_time_s": ep["wall_time_s"]}
            wins.append(len(w) < len(w0) and p < ALPHA)
        pp = binomtest(min(pooled_b, pooled_c), pooled_b + pooled_c, 0.5).pvalue \
            if pooled_b + pooled_c else 1.0
        rec["pooled"] = {"fixed_vs_C0": pooled_b, "newly_wrong_vs_C0": pooled_c, "p": pp}
        rec["beats_C0_at_every_seed"] = name != "C0" and all(wins)
        out["configs"][name] = rec
        for s in seeds:
            q = rec["per_seed"][s]
            zf = q["zero_fired"]
            rows.append(f"| {name} | {s} | {q['errors']} | {q['ratio_to_mwpm']:.3f} | "
                        f"{q['fixed_vs_C0']} | {q['newly_wrong_vs_C0']} | {q['p']:.3g} | "
                        f"{zf['pred_flip'] if zf else 'n/a'}/{zf['n'] if zf else 'n/a'} | "
                        f"{q['epochs_run']} ({q['best_epoch']}) |")
        rows.append(f"| {name} | pooled | | | {pooled_b} | {pooled_c} | {pp:.3g} | | |")

    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    (od / "phase3_compare.json").write_text(json.dumps(out, indent=1))
    table = ["| Config | Seed | Big-test errors | Ratio to MWPM | Fixed vs C0 | Newly wrong vs C0 | McNemar p | Zero-fired predicted flips | Epochs (best) |",
             "|---|---|---|---|---|---|---|---|---|"] + rows
    winners = [k for k, v in out["configs"].items() if v["beats_C0_at_every_seed"]]
    text = (f"MWPM: {mwpm_err} errors on {bt0['n_test']} big-test shots.\n\n" + "\n".join(table)
            + f"\n\nBeats C0 at every seed (fewer errors and p < {ALPHA}): "
            + (", ".join(winners) if winners else "none") + "\n")
    (od / "phase3_compare.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
