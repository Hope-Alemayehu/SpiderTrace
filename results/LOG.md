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