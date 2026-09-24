# Experiment Log

## Run 1: smoke test
- **Date:** 2026-09-24
- **Commit:** 3b70134 (clean)
- **Command:** `python qec_run.py --d 3 --p 0.01 --shots 10000 --seeds 0 1 --lambdas 0.01 0.1 --epochs 30 --device cpu`
- **Setup:** d=3, p=0.01, 10k shots, seeds 0 1, λ ∈ {0.01, 0.1}, 30 epochs, CPU
- **JSON:** results/run_20260924T055329Z_d3_p0.01.json

| Arm | Test LER |
|---|---|
| GNN-A | 0.178 ± 0.041 |
| GNN-Raw | 0.122 ± 0.024 |
| GNN-ZX | 0.118 ± 0.018 |

- **McNemar (ZX vs Raw):** seed 0 p = 1.0, seed 1 p = 0.23. Not significant.
- **Notes:**
    - GNN-A did not learn (test LER at the trivial baseline of ~0.18). 
    - GNN-Raw and GNN-ZX learned something and performed similarly, with no significant difference between them. 
    - All arms only started learning around epoch 20 to 25, so 30 epochs and 10k shots were too small a budget. 
    - Next run: 50k shots, 100 epochs, 3 seeds, on GPU.

**MWPM baseline:** 0.0576 (50k shots, trivial baseline 0.184)

## Run 2: d=3, p=0.01, full budget
- **Date:** 2026-09-24
- **Commit:** 893a5a3 (clean)
- **Where:** Colab, T4 GPU
- **Commands** (one per seed):
  `python qec_run.py --d 3 --p 0.01 --shots 50000 --seeds <0|1|2> --lambdas 0.01 0.1 0.5 --epochs 100 --device cuda --outdir $RESULTS`
- **Setup:** d=3, p=0.01, 50k shots, seeds 0 1 2, λ ∈ {0.01, 0.1, 0.5}, 100 epochs, split_seed 12345 (7,500 test shots)
- **JSON:**
  - results/run_20260924T082552Z_d3_p0.01.json (seed 0)
  - results/run_20260924T090326Z_d3_p0.01.json (seed 1)
  - results/run_20260924T093727Z_d3_p0.01.json (seed 2)
- **Combined with:** `compile_results.py`

| Arm | Test LER (mean ± std, 3 seeds) | Per seed |
|---|---|---|
| MWPM | 0.0545 | same 7,500 test shots (see note) |
| GNN-Raw | 0.0570 ± 0.0025 | 0.0580, 0.0541, 0.0589 |
| GNN-ZX | 0.0581 ± 0.0013 | 0.0568, 0.0593, 0.0583 |
| GNN-A | 0.0592 ± 0.0005 | 0.0587, 0.0596, 0.0595 |

- **Trivial baseline (never flip):** 0.184
- **Statistical uncertainty per LER:** about ±0.003 (7,500 test shots)
- **McNemar pooled (ZX vs Raw):** ZX right & Raw wrong = 213, Raw right & ZX wrong = 238, p = 0.258. Not significant. (Caveat: all seeds use the same test shots, so pooling overcounts.)
- **MWPM note:** Recomputed after the run on the exact same test split, on Colab, commit d640f39 (`make_datasets(3, 0.01, 50000, 12345)` + `mwpm_on_test`). Flip rate 0.1840 matches, confirming the same test set. A separate 50k-shot MWPM run gave 0.0576 (results/mwpm_d3_p0.01.json); the 0.003 gap is sampling noise, since this test set happens to be slightly easier than average. No ZX vs MWPM McNemar for this run (GNN models weren't saved).
- **Notes:**
  - All three arms perform close to MWPM but slightly behind it on the same shots (1 to 2 standard errors). They are statistically indistinguishable from each other.
  - Comparing against MWPM on different shots (0.0576) made the GNNs look like they matched or beat it. On the same shots they don't. Always compare on the same test set.
  - Raw has the lowest GNN mean, but not significantly. After seed 0 alone ZX looked best; across 3 seeds the ranking changed. One seed is not enough.
  - With 50k shots, models started learning around epoch 4 to 6 (vs 20 to 25 in Run 1).
  - Ceiling effect: d=3, p=0.01 is too easy to separate the arms.
  - Both aux arms often picked λ = 0.5, the largest value tried. Add λ = 1.0 next.
  - Next: Run 3 at d=5, p=0.003, on commit d640f39 (MWPM on the test split and ZX vs MWPM McNemar built in).