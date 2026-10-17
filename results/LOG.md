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

## Run 6: data scaling, GNN-A at 35k shots (100 epochs)
- **Date:** 2026-10-17
- **Commit:** 8ac71cb (clean)
- **Where:** Colab, GPU
- **Command:** `python qec_run.py --d 5 --p 0.003 --shots 50000 --split-seed 12345 --train-shots 35000 --arms A --seeds 0 --epochs 100 --device cuda --expect-mwpm-errors 36 --outdir /content/drive/MyDrive/SpiderTrace_results/data_scaling`
- **Setup:** GNN-A only, d=5, p=0.003, seed 0, up to 100 epochs. Base split of 50k shots with split_seed 12345: 35,000 train (the base train split, no extra shots), 7,500 val, 7,500 test. This is the same test set as Runs 3 to 5: `--expect-mwpm-errors 36` passed.
- **JSON:** results/run_20261017T045025Z_d5_p0.003.json

| Arm | Test LER | Test errors | Val LER |
|---|---|---|---|
| MWPM | 0.0048 | 36 | n/a |
| GNN-A | 0.0124 | 93 | 0.0119 |

- **Statistical uncertainty per LER:** about ±0.0013 (7,500 test shots)
- **McNemar (A vs MWPM):** A right & MWPM wrong = 13, MWPM right & A wrong = 70, p = 1.3e-10. MWPM is significantly better.
- **Training:** epochs_run 100 (the cap), best_epoch 98, wall time 1,130 s (18.8 min, 11.3 s per epoch).
- **Loss:**
  - Final epoch (100): train 0.0369, val 0.0385.
  - Best epoch (98, the restored weights): train 0.0350, val 0.0353.
- **Notes:**
  - The best epoch is 98 of 100, so the model was still improving at the cap. Run 7 lifts the cap.
  - Loss spikes: train loss jumps from 0.0392 to 0.0907 at epoch 77, and val loss jumps at epochs 31 (0.0971 to 0.1296), 60 (0.0468 to 0.0645) and 94 (0.0388 to 0.0542).

## Run 7: data scaling, GNN-A at 35k shots (up to 300 epochs)
- **Date:** 2026-10-17
- **Commit:** 8ac71cb (clean)
- **Where:** Colab, GPU (same host as Run 6)
- **Command:** `python qec_run.py --d 5 --p 0.003 --shots 50000 --split-seed 12345 --train-shots 35000 --arms A --seeds 0 --epochs 300 --device cuda --expect-mwpm-errors 36 --outdir /content/drive/MyDrive/SpiderTrace_results/data_scaling`
- **Setup:** as Run 6, but with up to 300 epochs. `--expect-mwpm-errors 36` passed.
- **JSON:** results/run_20261017T051442Z_d5_p0.003.json

| Arm | Test LER | Test errors | Val LER |
|---|---|---|---|
| MWPM | 0.0048 | 36 | n/a |
| GNN-A | 0.0127 | 95 | 0.0121 |

- **Statistical uncertainty per LER:** about ±0.0013 (7,500 test shots)
- **McNemar (A vs MWPM):** A right & MWPM wrong = 12, MWPM right & A wrong = 71, p = 2.4e-11. MWPM is significantly better.
- **Training:** epochs_run 116 (early stop, 10 epochs without val-loss improvement), best_epoch 106, wall time 1,310 s (21.8 min, 11.3 s per epoch).
- **Loss:**
  - Final epoch (116): train 0.0342, val 0.0395.
  - Best epoch (106, the restored weights): train 0.0341, val 0.0347.
- **Notes:**
  - Run 7 is not a continuation of Run 6. Same commit, seed and host, but train loss already differs at epoch 1 (4.2572 vs 4.2598), most likely from nondeterministic GPU kernels. The two runs differ in the epoch cap and in run-to-run noise.
  - Loss spikes: val loss jumps at epochs 40 (0.0770 to 0.1015) and 48 (0.0657 to 0.0861), and train loss rises from 0.0342 to 0.0434 at epoch 109.

## Interpretation (Runs 6 and 7)
- **Allowing 300 epochs did not help.** The model early-stopped at epoch 116 (best epoch 106) with test LER 0.0127, against 0.0124 at 100 epochs. That is 95 vs 93 errors, within noise. Both lie inside the range of earlier GNN-A results at 35k shots (0.0115 in Run 3, 0.0132 in Run 4).
- **The model is not overfitting.** At the restored (best) epoch, train and val loss are nearly equal, about 0.035: 0.0350 vs 0.0353 in Run 6 and 0.0341 vs 0.0347 in Run 7. The model cannot fit its training data better. This points to a capacity or optimization limit (78k parameters, a single linear head, constant lr 1e-3 with visible loss spikes), not to data shortage.
- **Caveat:** train loss is averaged over each epoch with dropout 0.1 active, while val loss is measured in eval mode. Equal train and val loss is therefore approximate evidence of no generalization gap, not proof.
- **Pending:** Run 8 (300k training shots, Kaggle) tests whether more data helps.
- **Planned next if Run 8 is flat:** lr decay and an MLP head at 35k.
- **Update after Run 8:** more training data clearly helped. On the same test shots, errors fell from 93 (Run 6, 35k) to 65 (Run 8, 300k), with a paired McNemar p = 0.00062 (see Run 8). The earlier reading "not data shortage" was wrong. Equal train and val loss showed that the model was not memorizing its training set, not that the data was sufficient. Run 7 showed that more steps on the same 35k did not help, so the gain comes from the extra data.
  - Caveat: the step counts are not matched. Run 7 stopped on its own after its best epoch 106 (about 14.5k gradient steps), while Run 8's best epoch 37 is at about 43k steps. The evidence is that the 35k run stopped improving by its own criterion, not a comparison at equal steps.

## Run 8: data scaling, GNN-A at 300k shots (100 epochs)
- **Date:** 2026-10-17
- **Commit:** 8ac71cb (clean). This predates the backbone options (5307ef0), so Run 8 uses the original unnormalized backbone: no norm, add-mean-max readout, the same model as Runs 6 and 7.
- **Where:** Kaggle, GPU
- **Command:** `python qec_run.py --d 5 --p 0.003 --shots 50000 --split-seed 12345 --train-shots 300000 --arms A --seeds 0 --epochs 100 --device cuda --expect-mwpm-errors 36 --outdir /kaggle/working/results`
- **Setup:** GNN-A only, d=5, p=0.003, seed 0, up to 100 epochs. 300,000 train shots: the 35,000 base train shots plus 265,000 extra shots from extra_train_seed 12346. Val (7,500) and test (7,500) are the base split with split_seed 12345, the same test set as Runs 3 to 7: `--expect-mwpm-errors 36` passed, and MWPM gets exactly the same test shots wrong as in Run 6.
- **JSON:** results/run_20261017T061603Z_d5_p0.003.json

| Arm | Test LER | Test errors | Val LER |
|---|---|---|---|
| MWPM | 0.0048 | 36 | n/a |
| GNN-A | 0.0087 | 65 | 0.0072 |

- **Statistical uncertainty per LER:** about ±0.0011 (7,500 test shots)
- **McNemar (A vs MWPM):** A right & MWPM wrong = 8, MWPM right & A wrong = 37, p = 1.5e-5. MWPM is still significantly better.
- **Training:** epochs_run 47 (early stop, 10 epochs without val-loss improvement), best_epoch 37, wall time 3,523 s (58.7 min, 75.0 s per epoch).
- **Loss:**
  - Final epoch (47): train 0.0230, val 0.0237.
  - Best epoch (37, the restored weights): train 0.0247, val 0.0209.
- **Paired comparison with Run 6** (35k train shots, same model, seed and test shots), from test_wrong_idx in both JSONs:

| | Shots |
|---|---|
| Wrong in Run 6 | 93 |
| Wrong in Run 8 | 65 |
| Fixed by Run 8 (wrong in Run 6, right in Run 8) | 46 |
| Newly wrong in Run 8 (right in Run 6, wrong in Run 8) | 18 |
| Wrong in both | 47 |

  - Exact binomial McNemar: p = 0.00062 (18 vs 46 of 64 discordant shots).
  - For scale, Run 6 vs Run 7 (same data, differing only by the epoch cap and GPU nondeterminism) gives 19 fixed vs 21 newly wrong. On the same data, two runs disagree on about 40 shots, split evenly; Run 8's split of 46 vs 18 is far from even.
- **Notes:**
  - Learning started at epoch 3 instead of about 13: val loss first drops below 0.40 at epoch 3 (0.1483), against epoch 16 in Run 6 and epoch 13 in Run 7. Per gradient step the onset is not earlier: 3 epochs at 300k are about 3.5k steps (1,172 per epoch), against about 2.2k (Run 6) and 1.8k (Run 7) at 35k (137 per epoch).
  - No loss spikes: no epoch-to-epoch jump above 1.3x in train or val loss, unlike Runs 6 and 7.
  - At the best epoch val loss (0.0209) is below train loss (0.0247). Train loss is measured with dropout active, so this is not a sign of a problem.
  - GNN-A is now 1.8x MWPM on the same shots (65 vs 36 errors), down from 2.6x in Run 6 (93 vs 36).

## Run 9: normalized backbone at 35k shots (100 epochs)
- **Date:** 2026-10-17
- **Commit:** 4f2e17f (clean). It contains the backbone options (5307ef0).
- **Where:** Colab, GPU
- **Command:** `python qec_run.py --d 5 --p 0.003 --shots 50000 --split-seed 12345 --train-shots 35000 --arms A --seeds 0 --epochs 100 --device cuda --expect-mwpm-errors 36 --norm layer --readout mean-max --outdir /content/drive/MyDrive/SpiderTrace_results/data_scaling`
- **Setup:** as Run 6 (GNN-A only, d=5, p=0.003, seed 0, 35,000 train shots, up to 100 epochs, same test set: `--expect-mwpm-errors 36` passed, and MWPM gets exactly the same test shots wrong as in Run 6). The only change is the backbone: LayerNorm after each GINEConv (before the ReLU) and a mean+max readout instead of add+mean+max. 78,725 parameters, against 78,053 in Run 6.
- **JSON:** results/run_20261017T070707Z_d5_p0.003.json

| Arm | Test LER | Test errors | Val LER |
|---|---|---|---|
| MWPM | 0.0048 | 36 | n/a |
| GNN-A | 0.0147 | 110 | 0.0133 |

- **Statistical uncertainty per LER:** about ±0.0014 (7,500 test shots)
- **McNemar (A vs MWPM):** A right & MWPM wrong = 13, MWPM right & A wrong = 87, p = 1.3e-14. MWPM is significantly better.
- **Training:** epochs_run 100 (the cap), best_epoch 100, wall time 1,120 s (18.7 min, 11.2 s per epoch).
- **Loss:**
  - Final epoch (100), which is also the best epoch: train 0.0383, val 0.0379.
  - Epoch 1: train 0.433, val 0.439 (Run 6: train 4.257, val 0.454).
- **Paired comparison with Run 6** (same data, seed and test shots; old backbone), from test_wrong_idx in both JSONs:

| | Shots |
|---|---|
| Wrong in Run 6 | 93 |
| Wrong in Run 9 | 110 |
| Fixed by Run 9 (wrong in Run 6, right in Run 9) | 10 |
| Newly wrong in Run 9 (right in Run 6, wrong in Run 9) | 27 |
| Wrong in both | 83 |

  - Exact binomial McNemar: p = 0.0076 (10 vs 27 of 37 discordant shots).
- **Notes:**
  - Val loss stays between 0.430 and 0.442 for epochs 1 to 16 and first drops below 0.40 at epoch 18 (Run 6: epoch 16).
  - After that it descends slowly, between 0.169 and 0.099 over epochs 22 to 55, then from 0.083 at epoch 60 to 0.054 at epoch 64. At epochs 40, 55 and 60, Run 9's val loss is 0.126, 0.104 and 0.083, against 0.087, 0.051 and 0.065 in Run 6.
  - Loss spikes: val loss jumps at epochs 58 (0.0853 to 0.1109) and 89 (0.0404 to 0.0759). No train-loss jump above 1.3x.

## Interpretation (Run 9)
- **The hypothesis** was that the unnormalized backbone's huge initial logits cause the early plateau and limit learning. Initial BCE did drop, from about 126 to 0.67 in the local check at initialization (2026-10-17, local CPU, 256 locally drawn d=5 shots; not from a run JSON). In Run 9 the epoch-1 train loss is 0.433 against 4.257 in Run 6. But the plateau remained: val loss stayed near the trivial 0.44 until about epoch 16, then a second slow phase around 0.10 to 0.13 lasted until about epoch 60.
- **So the early plateau is not caused by initial scale.**
- **At equal budget (35k shots, 100 epochs) Run 9 is significantly worse than Run 6:** 110 vs 93 errors, paired p = 0.0076. best_epoch was 100, so it was still improving. It learns more slowly; it has not been shown to be worse at convergence.
- **Caveats:** one seed per backbone. Run 9 changed two things at once (LayerNorm and readout) and kept lr 1e-3, which was not retuned for the normalized backbone.
- **Decision:** keep the default backbone for now. The `--norm` and `--readout` options stay in the code, off by default.
- **Next:** Run 10, 1M training shots, default backbone, on Kaggle.
