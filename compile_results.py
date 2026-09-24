import json, glob, statistics
from collections import defaultdict
from scipy.stats import binomtest

# 1. Load every run file and group runs with identical settings
groups = defaultdict(list)
for path in sorted(glob.glob("results/run_*.json")):
    data = json.load(open(path))
    c = data["provenance"]["config"]
    key = (c["d"], c["p"], c["shots"], c["epochs"], tuple(c["lambdas"]))
    groups[key].append(data)

# 2. For each group, combine seeds
for (d, p, shots, epochs, lambdas), runs in groups.items():
    print(f"\n=== d={d} p={p} shots={shots} epochs={epochs} "
          f"lambdas={list(lambdas)} ({len(runs)} files) ===")

    for arm in ("A", "Raw", "ZX"):
        lers = [s["test_ler"] for data in runs
                for s in data["arms"][arm]["selected_per_seed"]]
        mean = statistics.fmean(lers)
        std = statistics.stdev(lers) if len(lers) > 1 else 0.0
        print(f"  {arm:>3}: {mean:.4f} ± {std:.4f}  "
              f"(per seed: {[round(x, 4) for x in lers]})")

    # 3. Pool McNemar disagreements across seeds
    zx_only = raw_only = 0
    for data in runs:
        for seed, m in data["mcnemar_zx_vs_raw"].items():
            if seed.startswith("_"):
                continue
            zx_only += m["discordant_a_right_b_wrong"]
            raw_only += m["discordant_a_wrong_b_right"]
    n = zx_only + raw_only
    p_val = binomtest(zx_only, n, 0.5).pvalue if n else 1.0
    print(f"  McNemar pooled: ZX right & Raw wrong = {zx_only}, "
          f"Raw right & ZX wrong = {raw_only}, p = {p_val:.3f}")