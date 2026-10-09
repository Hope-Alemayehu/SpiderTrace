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

## Run 3: d=5, p=0.003, DEM graph
- **Date:** 2026-09-24
- **Commit:** d640f39 (clean)
- **Where:** Colab, GPU
- **Command:** `python qec_run.py --d 5 --p 0.003 --shots 50000 --seeds 0 --lambdas 0.01 0.1 0.5 1.0 --epochs 100 --device cuda --outdir /content/drive/MyDrive/SpiderTrace_results`
- **Setup:** d=5, p=0.003, 50k shots, seed 0, λ ∈ {0.01, 0.1, 0.5, 1.0}, 100 epochs, split_seed 12345 (7,500 test shots), aux over all 64 qubit slots, plain aux CE
- **JSON:** results/run_20260924T131514Z_d5_p0.003.json

| Arm | Test LER | Selected λ | Val LER |
|---|---|---|---|
| MWPM | 0.0048 | n/a | same 7,500 test shots |
| GNN-A | 0.0115 | n/a | 0.0128 |
| GNN-Raw | 0.0121 | 0.1 | 0.0117 |
| GNN-ZX | 0.0121 | 0.1 | 0.0107 |

- **Trivial baseline (never flip):** about 0.159 (flip rate in the 50k-shot MWPM run)
- **Statistical uncertainty per LER:** about ±0.0013 (7,500 test shots)
- **McNemar (ZX vs Raw):** 11 vs 11 discordant, p = 1.0. Identical LER.
- **McNemar (ZX vs MWPM):** ZX right & MWPM wrong = 12, MWPM right & ZX wrong = 67, p = 2e-10. MWPM is significantly better.
- **Notes:**
  - All three arms are within one standard error of each other. GNN-A has the lowest LER, though not significantly.
  - The GNNs are about 2.5× worse than MWPM on the same shots, even though they decode on the same (decomposed) DEM graph.
  - λ = 1.0 was worst for both aux arms (test 0.0132 for Raw and 0.0139 for ZX).
  - Val-LER λ selection is noisy. Val LER ranges only 0.0107–0.0147 across the grid (about 80–110 errors in 7,500 shots), and the val ranking doesn't match the test ranking.
  - The aux head most likely collapsed to predicting I everywhere: ~95% of the 64 slots are I, and the loss is unweighted. This JSON has no aux diagnostics, so the collapse isn't confirmed.
  - The model init was unseeded (fixed in 20758be), so the numbers aren't exactly reproducible.
  - Next: Run 4 with the aux head fixed.

## Run 4: d=5, p=0.003, aux head fixed
- **Date:** 2026-09-28
- **Commit:** 44ce02d (clean)
- **Where:** Kaggle, GPU
- **Command:** `python qec_run.py --d 5 --p 0.003 --shots 50000 --seeds 0 --lambdas 0.01 0.1 0.5 1.0 --epochs 100 --device cuda --outdir /kaggle/working/results --aux-data-qubits-only --aux-class-weight inv-freq`
- **Setup:** as in Run 3 (seed 0, split_seed 12345, 7,500 test shots). The aux head predicts data qubits only (25 slots instead of 64), and the aux loss is class-weighted by inverse Pauli frequency, taken from the training split. The model seed now controls the initial weights.
- **Aux class weights for I/X/Y/Z:** Raw ≈ 0.26 / 11.0 / 17.5 / 19.0, ZX ≈ 0.27 / 8.8 / 17.2 / 11.6
- **Compare against:** Run 3. This is not perfectly controlled: Run 3 used unseeded initial weights and a different platform (Colab).
- **JSON:** results/run_20260928T035724Z_d5_p0.003.json

| Arm | Test LER | Selected λ | Val LER | Aux non-I recall | Aux non-I precision (X / Y / Z) |
|---|---|---|---|---|---|
| MWPM | 0.0048 | n/a | n/a | n/a | n/a |
| GNN-A | 0.0132 | n/a | 0.0131 | n/a | n/a |
| GNN-Raw | 0.0121 | 0.01 | 0.0111 | 0.563 | 0.088 / 0.062 / 0.029 |
| GNN-ZX | 0.0127 | 0.1 | 0.0117 | 0.539 | 0.100 / 0.097 / 0.058 |

- **Statistical uncertainty per LER:** about ±0.0013 (7,500 test shots)
- **McNemar (ZX vs Raw):** 11 vs 15 discordant, p = 0.56. Not significant.
- **McNemar (ZX vs MWPM):** ZX right & MWPM wrong = 14, MWPM right & ZX wrong = 73, p = 9e-11
- **λ grid, test LER (0.01 / 0.1 / 0.5 / 1.0):** Raw 0.0121 / 0.0243 / 0.0176 / 0.0251; ZX 0.0133 / 0.0127 / 0.0163 / 0.0261
- **Notes:**
  - **Sanity checks pass:**
    - Commit 44ce02d, clean.
    - MWPM is 0.0048, the same test set as Run 3.
    - Neither aux head collapsed to identity. The fraction predicted I is 45% for Raw and 57% for ZX, against 95% and 94% I in the targets.
  - **ZX vs Raw:** no detectable difference (p = 0.56). ZX is nominally worse by 0.0006.
  - **Does aux help at all:** no. The aux task learns something but doesn't transfer to LER.
    - Against this run's GNN-A (0.0132), Raw and ZX are 0.0011 and 0.0005 lower. Against Run 3's GNN-A (0.0115), both are higher.
    - Every gap is within ±0.0013.
    - GNN-A itself moved 0.0017 between Runs 3 and 4 with the same settings for that arm. That shift is larger than any gap between arms.
  - **λ:** Raw was pinned at 0.01, the lowest value tried; the weighted aux loss hurts Raw badly at λ ≥ 0.1 (0.018–0.025). ZX tolerates λ = 0.1 but degrades at 1.0 (0.026).
  - **Aux accuracy:** non-I recall is about 0.55 for both arms, but precision is only 3–10%.
    - The head over-predicts X/Y/Z. That is what the heavy class weights reward when the per-qubit frame can't be identified from the syndrome.
    - ZX has higher precision than Raw on every Pauli, so its targets are slightly more learnable, but that doesn't improve LER.
  - **Overall:** this run agrees with Runs 2–3. Aux supervision gives no measurable LER gain, and all GNN arms are about 2.5× worse than MWPM on the same graph.

### Run 4: salvaged from printed output (JSON lost)
- **Status:** these numbers were copied from printed console output. Their JSON files were lost, so they are not file-backed. Do not pool them with file-backed results.
- **Setup (all):** commit 44ce02d, d=5, p=0.003, `--aux-data-qubits-only --aux-class-weight inv-freq`
- **Not captured:** selected λ for every arm and seed.
- "Aux non-I acc" is the same quantity as "aux non-I recall" in the table above (`aux_acc_nonidentity`). "Pred-I" is the fraction of aux predictions that are I.

| Platform, seed | GNN-A | GNN-Raw (aux non-I acc, pred-I) | GNN-ZX (aux non-I acc, pred-I) | MWPM | McNemar ZX vs Raw | McNemar ZX vs MWPM |
|---|---|---|---|---|---|---|
| Kaggle, seed 1 | 0.0123 | 0.0115 (0.5534, 0.440) | 0.0113 (0.6602, 0.643) | 0.0048 | p = 1.0 | p = 8.396e-09 |
| Colab, seed 0 | 0.0131 | 0.0127 (0.7591, 0.681) | 0.0121 (0.6195, 0.600) | 0.0048 | p = 0.5572 | p = 3.808e-10 |
| Colab, seed 1 (partial) | not captured | not captured | 0.0125 (0.4535, 0.519) | 0.0048 | p = 0.3915 | p = 7.864e-11 |

## 2026-10-06: ZX target consistency check (no training)
- **Commit:** 0992fc7 (clean). `qec_zx_dataset.py` is unchanged since 10d3835.
- **Check:** `tests/test_zx_target_consistency.py`. Run `python tests/test_zx_target_consistency.py` for the report below, or `pytest tests/test_zx_target_consistency.py`. Seeded DEM sampler (seed 12345), 5,000 shots per config.
- **Rule:** a correctly propagated end-of-circuit frame must reproduce the logical label. In this memory-Z circuit the label is the parity of X or Y components of the final frame on the observable qubits. `raw_target` is not a propagated frame, so its match rate is a comparison only.

| Config | ZX match | Raw match | Shots whose zx_target is wrong |
|---|---|---|---|
| d=3, p=0.003 | 0.9776 | 0.9958 | 0.0920 |
| d=5, p=0.003 (Runs 3, 4) | 0.9514 | 0.9772 | 0.3448 |
| d=3, p=0.01 (Runs 1, 2) | 0.9274 | 0.9818 | 0.2744 |

- "Shots whose zx_target is wrong" counts shots whose stored target differs from the target built by injecting each fault exactly at its noise instruction. This includes differences the label check cannot see.
- **DEM walk** (every DEM error's representative fault):
  - d=3, p=0.003: 219 errors. 9 have label-inconsistent ZX frames (total probability 0.0233). 99 have frames that differ from exact-position injection (total probability 0.0975).
  - d=5, p=0.003: 1,677 errors. 47 label-inconsistent (0.0511). 811 differ from exact-position injection (0.4404).
  - Label-inconsistent at d=3 (index, probability, tick_offset, gate, fault): 9, 0.012277, 0, X_ERROR, X1 | 35, 0.005781, 0, X_ERROR, X5 | 6, 0.002398, 2, DEPOLARIZE2, Y2 | 39, 0.000801, 9, DEPOLARIZE2, Y2 | 50, 0.000801, 4, DEPOLARIZE2, Y11 | 82, 0.000601, 11, DEPOLARIZE2, Y11 | 70, 0.000200, 11, DEPOLARIZE2, Y11 Y5 | 83, 0.000200, 11, DEPOLARIZE2, Y11 Z5 | 94, 0.000200, 11, DEPOLARIZE2, X11 Y5.
- **Cause (confirmed):** `ReferenceZXPropagator.propagate` injects each fault at the start of its tick layer, but in this circuit every noise instruction comes after the gate, reset or measurement it models, in the same layer.
  - `qec_zx_dataset.py:106-109`: with `tick_offset == 0` the fault is injected before the first instruction, and the leading `R` erases it. Post-reset X faults on data qubits get an identity frame.
  - `qec_zx_dataset.py:110-117`: with `tick_offset > 0` the fault is injected right after the TICK, so it is wrongly propagated through the operation it followed. Wrong final frames come only from `CX` here (and `R` at tick 0). Faults after `MR` get wrong detector signatures, but the next reset clears their frame either way, so the stored target is unaffected.
  - These are one root cause. Tick 0 is the case where the preceding operation is a reset.
  - Evidence: a replica of the injection reproduces every stored frame. Injecting exactly at the noise instruction reproduces the detector and observable flips of all 219 (d=3) and 1,677 (d=5) DEM errors, while layer-start injection gets 207 and 1,557 of them wrong. Every wrong frame follows a same-layer `R` (7 at d=3, 21 at d=5) or `CX` (92 at d=3, 790 at d=5).
- **SpiderTraceAdapter:** by code reading it uses the same convention (`qec_zx_dataset.py:136-138`, `:190-192`). Not executed. Its "0/7367 mismatches" validation (e2240d5) compared two implementations with the same injection convention.
- **Affected results:** every GNN-ZX result in Runs 1 to 4 (commits 3b70134, 893a5a3, d640f39, 44ce02d) and the ZX rows of the two runs in `results/diagnostic/` (7d0a835) used the affected `zx_target`. All of these commits contain 10d3835. `raw_target` is not propagated, so GNN-Raw and GNN-A are not affected.
- **Not yet known:** whether correct targets change any ZX result. The propagator is not fixed. The three xfail tests will XPASS (and fail, since they are strict) once it is.

## 2026-10-06: ZX target fix (no training)
- **Commit:** d92b05a (parent 2836225). Training code (`qec_run.py`, `gnn_models.py`, `train.py`) is unchanged.
- **Fix:** `ReferenceZXPropagator` and `SpiderTraceAdapter` now inject each fault right after its own noise instruction in the flattened circuit (`ZXPropagator.position`, from the error location's stack frames), not at the start of its tick layer. `validate_adapter` now compares the two propagators on every DEM error plus random positions, and `validate_divergence_rate` compares them on the same seeded shots instead of the old fixed baseline (0.856), which had been measured with the buggy reference.
- **Check:** `pytest tests/test_zx_target_consistency.py` gives 9 passed, with the xfail markers removed. The adapter matches the reference on all 219 (d=3) and 1,677 (d=5) DEM faults.

| Config | ZX match before | ZX match after | Raw match | Wrong ZX shots before | Wrong ZX shots after |
|---|---|---|---|---|---|
| d=3, p=0.003 | 0.9776 | 1.0000 | 0.9958 | 0.0920 | 0.0000 |
| d=5, p=0.003 (Runs 3, 4) | 0.9514 | 1.0000 | 0.9772 | 0.3448 | 0.0000 |
| d=3, p=0.01 (Runs 1, 2) | 0.9274 | 1.0000 | 0.9818 | 0.2744 | 0.0000 |

- DEM walk: label-inconsistent ZX frames 9 to 0 (d=3) and 47 to 0 (d=5). Frames differing from the exact-position reference 99 to 0 and 811 to 0.
- "Before" values are recomputed with a replica of the old injection and equal the consistency-check entry above.
- **Not rerun:** Runs 1 to 4 and `results/diagnostic/` keep the old targets, so their GNN-ZX numbers are not comparable with runs from this commit on.

## Run 5: first look after ZX fix (d=3, 1 seed)
- **Status:** first look, not a result. One seed, at d=3, where the arms sit near the MWPM ceiling.
- **Date:** 2026-10-06
- **Commit:** 224b571 (clean). It contains the ZX target fix (d92b05a).
- **Where:** Colab, GPU
- **Command:** `python qec_run.py --d 3 --p 0.01 --shots 50000 --seeds 0 --lambdas 0.01 0.1 0.5 1.0 --epochs 100 --device cuda --outdir /content/drive/MyDrive/SpiderTrace_results/fixed_d3 --aux-data-qubits-only --aux-class-weight inv-freq`
- **Setup:** d=3, p=0.01, 50k shots, seed 0, λ ∈ {0.01, 0.1, 0.5, 1.0}, 100 epochs, split_seed 12345 (7,500 test shots), both aux flags, fixed ZX targets
- **JSON:** results/run_20261006T102236Z_d3_p0.01.json

| Arm | Test LER | Selected λ |
|---|---|---|
| MWPM | 0.0545 | n/a |
| GNN-A | 0.0592 | n/a |
| GNN-Raw | 0.0597 | 0.1 |
| GNN-ZX | 0.0589 | 0.1 |

- **McNemar (ZX vs Raw):** ZX right & Raw wrong = 56, Raw right & ZX wrong = 50, p = 0.627.
- **McNemar (ZX vs MWPM):** ZX right & MWPM wrong = 82, MWPM right & ZX wrong = 115, p = 0.022.
- **Aux diagnostics:**

| Arm | Non-I target density | Non-I recall | collapsed_to_identity |
|---|---|---|---|
| GNN-Raw | 0.0984 | 0.7552 | false |
| GNN-ZX | 0.1222 | 0.7169 | false |

- **Notes:**
  - Within the ZX arm, test LER ranges from 0.0589 to 0.0677 across λ, so model-to-model noise is larger than the differences between arms.
  - First look only: one seed, d=3 ceiling. Not a result.
