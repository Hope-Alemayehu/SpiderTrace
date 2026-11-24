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

## Run 10: data scaling, GNN-A at 600k shots (100 epochs)
- **Date:** 2026-10-19
- **Commit:** not recorded in the JSON. Provenance has `git_available: false` and `git_commit`, `git_dirty` and `git_dirty_files` all null: the script was run by full path (`/kaggle/working/SpiderTrace/qec_run.py`) from `/kaggle/working`, outside the repo, so git was not reachable. Commit recovered from the notebook's clone output: [TBD: commit hash]. Whether the tree was dirty is unknown. The provenance config contains `norm` and `readout`, so the code was at 5307ef0 or later.
- **Where:** Kaggle, GPU
- **Command:** `python /kaggle/working/SpiderTrace/qec_run.py --d 5 --p 0.003 --shots 50000 --split-seed 12345 --train-shots 600000 --arms A --seeds 0 --epochs 100 --device cuda --expect-mwpm-errors 36 --outdir /kaggle/working/results`
- **Setup:** GNN-A only, default backbone (norm none, add-mean-max readout, 78,053 parameters), d=5, p=0.003, seed 0, up to 100 epochs. 600,000 train shots: the 35,000 base train shots plus 565,000 extra shots from extra_train_seed 12346. Val (7,500) and test (7,500) are the base split with split_seed 12345, the same test set as Runs 3 to 9: `--expect-mwpm-errors 36` passed, and MWPM gets exactly the same test shots wrong as in Runs 6 and 8. Run 9's interpretation planned 1M train shots; this run used 600k.
- **JSON:** results/run_20261019T064106Z_d5_p0.003.json

| Arm | Test LER | Test errors | Val LER |
|---|---|---|---|
| MWPM | 0.0048 | 36 | n/a |
| GNN-A | 0.0071 | 53 | 0.0064 |

- **Statistical uncertainty per LER:** about ±0.0010 (7,500 test shots)
- **McNemar (A vs MWPM):** A right & MWPM wrong = 13, MWPM right & A wrong = 30, p = 0.014. MWPM is still significantly better.
- **Training:** epochs_run 55 (early stop, 10 epochs without val-loss improvement), best_epoch 45, wall time 8,001 s (133.4 min, 145.5 s per epoch).
- **Loss:**
  - Final epoch (55): train 0.0197, val 0.0214.
  - Best epoch (45, the restored weights): train 0.0208, val 0.0180.
- **Paired comparisons** (same model, seed and test shots), from test_wrong_idx in the JSONs:

| | vs Run 8 (300k) | vs Run 6 (35k) |
|---|---|---|
| Wrong in the earlier run | 65 | 93 |
| Wrong in Run 10 | 53 | 53 |
| Fixed by Run 10 (wrong before, right in Run 10) | 30 | 58 |
| Newly wrong in Run 10 (right before, wrong in Run 10) | 18 | 18 |
| Wrong in both | 35 | 35 |
| Exact binomial McNemar p | 0.11 | 4.7e-6 |

  - Against Run 8 the difference is not significant: 30 vs 18 of 48 discordant shots. For scale, Runs 6 and 7 (same data) disagree on 40 shots, split 19 vs 21. The split here leans toward Run 10 but not enough to rule out chance.
  - Against Run 6 the improvement is clear.
- **Notes:**
  - Val loss is 0.170 after epoch 1, already well below the trivial 0.44. One epoch at 600k is 2,344 gradient steps; the best epoch 45 is at about 105k steps.
  - No loss spikes: no epoch-to-epoch jump above 1.3x in train or val loss.
  - At the best epoch val loss (0.0180) is below train loss (0.0208), as in Run 8; train loss is measured with dropout active.

## Data scaling summary (GNN-A, default backbone)
All four runs use the same model, seed 0, up to 100 epochs, and the same 7,500 test shots, on which MWPM makes 36 errors. Run 11 is also scored on the big test set (200,000 fresh shots, seed 12347), on which MWPM makes 664 errors.

| Run | Train shots | Test errors | Test LER | Ratio to MWPM (36 errors) | Paired p vs previous row | Big test errors (LER) | Big-test ratio to MWPM (664 errors) |
|---|---|---|---|---|---|---|---|
| Run 6 | 35,000 | 93 | 0.0124 | 2.58 | n/a | n/a | n/a |
| Run 8 | 300,000 | 65 | 0.0087 | 1.81 | 0.00062 | n/a | n/a |
| Run 10 | 600,000 | 53 | 0.0071 | 1.47 | 0.11 | n/a | n/a |
| Run 11 | 1,000,000 | 37 | 0.0049 | 1.03 | 0.023 | 1,021 (0.0051) | 1.54 |

- Single-seed results: each row is one training run.
- The ratio to MWPM on the 7,500-shot set is noisy. MWPM makes only 36 errors there, so the ratio's denominator alone carries a relative uncertainty of about 1/sqrt(36), roughly 17%.
- The 7,500-shot ratios also understate the gap: 36 errors is high for MWPM. At its big-test LER (0.00332) it would make about 25 errors on 7,500 shots. For Run 11 the ratio is 1.03 on the 7,500-shot set and 1.54 on the big test.
- On the 7,500-shot set, the steps from 35k to 300k (p = 0.00062) and from 600k to 1M (p = 0.023) are significant. The step from 300k to 600k is in the same direction but not significant.
- From Run 11 on, the big-test numbers are the headline.

## Run 11: data scaling, GNN-A at 1M shots, with the big test set (100 epochs)
- **Date:** 2026-10-19
- **Commit:** 70ce245 (clean)
- **Where:** Kaggle, GPU
- **Command:** `python qec_run.py --d 5 --p 0.003 --shots 50000 --split-seed 12345 --train-shots 1000000 --arms A --seeds 0 --epochs 100 --device cuda --expect-mwpm-errors 36 --big-test-shots 200000 --outdir /kaggle/working/results`
- **Setup:** GNN-A only, default backbone (norm none, add-mean-max readout, 78,053 parameters), d=5, p=0.003, seed 0, up to 100 epochs. 1,000,000 train shots: the 35,000 base train shots plus 965,000 extra shots from extra_train_seed 12346. Val (7,500) and test (7,500) are the base split with split_seed 12345, the same test set as Runs 3 to 10: `--expect-mwpm-errors 36` passed, and MWPM gets exactly the same test shots wrong as in Run 10. Not resumed (no checkpoint record in the JSON).
- **Big test set:** 200,000 fresh shots from seed 12347, scored after training and never used for training, selection or early stopping. Built in 38.2 s; MWPM decoded it in 2.5 s and GNN-A in 27.8 s.
- **JSON:** results/run_20261019T100602Z_d5_p0.003.json

**Primary test set (7,500 shots)**

| Arm | Test LER | Test errors | Val LER |
|---|---|---|---|
| MWPM | 0.0048 | 36 | n/a |
| GNN-A | 0.0049 | 37 | 0.0057 |

- **Statistical uncertainty per LER:** about ±0.0008 (7,500 test shots)
- **McNemar (A vs MWPM):** A right & MWPM wrong = 11, MWPM right & A wrong = 12, p = 1.0. No detectable difference on this set.
- **Paired comparison with Run 10** (600k train shots, same model, seed and test shots), from test_wrong_idx in both JSONs:

| | Shots |
|---|---|
| Wrong in Run 10 | 53 |
| Wrong in Run 11 | 37 |
| Fixed by Run 11 (wrong in Run 10, right in Run 11) | 30 |
| Newly wrong in Run 11 (right in Run 10, wrong in Run 11) | 14 |
| Wrong in both | 23 |

  - Exact binomial McNemar: p = 0.023 (14 vs 30 of 44 discordant shots).

**Big test set (200,000 shots, seed 12347)**

| Arm | Test LER | Test errors |
|---|---|---|
| MWPM | 0.00332 (±0.00013) | 664 |
| GNN-A | 0.00511 (±0.00016) | 1,021 |

- **Ratio GNN-A / MWPM:** 1.54 (1,021 vs 664 errors).
- **McNemar (A vs MWPM):** A right & MWPM wrong = 184, MWPM right & A wrong = 541, p = 1.3e-41. MWPM is clearly better. Both are wrong on 480 shots.

- **Training:** epochs_run 41 (early stop, 10 epochs without val-loss improvement), best_epoch 31, wall time 9,913 s (165.2 min, 241.8 s per epoch, 3,907 gradient steps per epoch).
- **Loss:**
  - Final epoch (41): train 0.0233, val 0.0242.
  - Best epoch (31, the restored weights): train 0.0184, val 0.0161.
- **Notes:**
  - Instability: at epoch 32 both losses jumped, train from 0.0184 to 0.0207 and val from 0.0161 to 0.0239, and the run did not recover. Val loss came back to 0.0172 at epoch 35 but never below 0.0161 again, and train loss never got back below 0.0204. Early stopping therefore restored epoch 31. An earlier jump at epoch 4 (val 0.0425 to 0.0624) did recover.
  - Likely cause, not confirmed: the constant lr of 1e-3 with no gradient clipping.

## Interpretation (Run 11)
- **The primary set shows parity, the big test does not.** On the 7,500-shot set GNN-A and MWPM are level (37 vs 36 errors, p = 1.0). On the big test MWPM is clearly better (1,021 vs 664 errors, about 1.54x, p = 1.3e-41).
- **The 7,500-shot set is too small to resolve this.** GNN-A's 37 errors match its big-test rate (0.00511 × 7,500 is about 38). MWPM's 36 is well above its big-test rate (0.00332 × 7,500 is about 25). The apparent parity comes from MWPM doing unusually badly on this particular set.
- **From now on the big test is the headline number.** The 7,500-shot set stays the primary split for continuity and paired comparisons with earlier runs.
- **More data still helped:** 53 to 37 errors from 600k to 1M on the primary set, paired p = 0.023.
- **Next:** Run 12, same as Run 11 plus lr decay and gradient clipping.

## Phase 3, S1 (C0 vs C2)
- **Date:** 2026-11-16
- **Commit:** 5110582 (clean), branch arch-exploration. Both configs ran back to back in one Kaggle session (same host), torch 2.11.0+cu130.
- **Where:** Kaggle, GPU
- **Command (C0):** `python qec_run.py --d 5 --p 0.003 --shots 50000 --split-seed 12345 --train-shots 300000 --arms A --seeds 0 1 --epochs 100 --patience 10 --lr 1e-3 --lr-schedule constant --big-test-shots 200000 --expect-mwpm-errors 36 --device cuda --outdir /kaggle/working/results/C0`
- **Command (C2):** the same plus `--readout fired`, with `--outdir /kaggle/working/results/C2`.
- **Setup:** first session of the Phase 3 architecture ablation. GNN-A only, d=5, p=0.003, seeds 0 and 1, up to 100 epochs, patience 10, constant lr 1e-3, no gradient clipping. 300,000 train shots (the 35,000 base train shots plus 265,000 extra shots from extra_train_seed 12346), as in Run 8. Same primary test set as Runs 3 to 11 (`--expect-mwpm-errors 36` passed) and same big test set as Run 11 (200,000 shots, seed 12347, 664 MWPM errors). Not resumed.
  - C0: default backbone (norm none, add-mean-max readout, linear head), 78,053 parameters.
  - C2: readout is mean and max over fired detectors only; a shot with no fired detector gets a learned "empty" vector instead. The boundary node is never fired, so it is never pooled. 78,149 parameters.
- **JSON:** C0 results/run_20261116T154357Z_d5_p0.003.json, C2 results/run_20261116T183801Z_d5_p0.003.json
- **Analysis:** analysis/phase3_results/phase3_compare.md (from analysis/phase3_compare.py) and analysis/phase3_results/s1_details.json (from analysis/phase3_s1_details.py).

**Big test set (200,000 shots, seed 12347)**

| Config | Seed | Errors | LER | Ratio to MWPM (664 errors) | Fixed vs C0 | Newly wrong vs C0 | McNemar p vs C0 |
|---|---|---|---|---|---|---|---|
| MWPM | | 664 | 0.003320 (±0.00013) | 1 | | | |
| C0 | 0 | 1,348 | 0.006740 (±0.00018) | 2.03 | | | |
| C0 | 1 | 1,429 | 0.007145 (±0.00019) | 2.15 | | | |
| C2 | 0 | 1,477 | 0.007385 (±0.00019) | 2.22 | 321 | 450 | 3.8e-6 |
| C2 | 1 | 1,379 | 0.006895 (±0.00019) | 2.08 | 444 | 394 | 0.090 |

- **Pooled over seeds:** 765 fixed vs 844 newly wrong, p = 0.052. Both seeds are scored on the same shots, so the pooled test overstates significance and is not used for the decision.
- **McNemar vs MWPM** (A right & MWPM wrong / MWPM right & A wrong): C0 seed 0 184 / 868, p = 1.1e-106; C0 seed 1 189 / 954, p = 2.7e-123; C2 seed 0 178 / 991, p = 3.9e-137; C2 seed 1 204 / 919, p = 9.3e-109.
- **Decision** (rule fixed before the runs: fewer errors than C0 and p < 0.05 at every seed): **C2 does not beat C0.** At seed 0 it is significantly worse (1,477 vs 1,348 errors, p = 3.8e-6). At seed 1 it has fewer errors (1,379 vs 1,429), but not significantly (p = 0.090).
- **Seed-to-seed spread within one config** (big test, from test_wrong_idx):
  - C0 seed 0 vs seed 1: 1,348 vs 1,429 errors, 332 vs 413 of 745 discordant shots, p = 0.0034.
  - C2 seed 0 vs seed 1: 1,477 vs 1,379 errors, 446 vs 348 of 794 discordant shots, p = 0.00057.
  - Two trainings of the same config differ about as much as C2 differs from C0.

**Primary test set (7,500 shots)**

| Config | Seed | Test errors | Test LER | Val LER | McNemar vs MWPM (A right & MWPM wrong / MWPM right & A wrong) | McNemar vs C0 (fixed / newly wrong) |
|---|---|---|---|---|---|---|
| MWPM | | 36 | 0.0048 | n/a | | |
| C0 | 0 | 58 | 0.0077 | 0.0075 | 10 / 32, p = 0.00094 | |
| C0 | 1 | 54 | 0.0072 | 0.0067 | 13 / 31, p = 0.0096 | |
| C2 | 0 | 64 | 0.0085 | 0.0075 | 7 / 35, p = 1.5e-5 | 12 / 18, p = 0.36 |
| C2 | 1 | 60 | 0.0080 | 0.0069 | 9 / 33, p = 0.00027 | 17 / 23, p = 0.43 |

- **Statistical uncertainty per LER:** about ±0.0010 (7,500 test shots)
- **Comparison with Run 8** (same settings at seed 0, commit 8ac71cb, torch 2.11.0+cu128): Run 8 made 65 primary errors, C0 seed 0 makes 58; 21 fixed vs 14 newly wrong, p = 0.31.

**Training**

| Config | Seed | epochs_run | best_epoch | Wall time | Val loss, epoch 1 | Val loss, epoch 3 | Val loss, epoch 10 | Best val loss (train loss at that epoch) |
|---|---|---|---|---|---|---|---|---|
| C0 | 0 | 69 | 59 | 5,204 s (86.7 min, 75.4 s per epoch) | 0.426 | 0.143 | 0.0401 | 0.0183 (0.0219) |
| C0 | 1 | 71 | 61 | 5,269 s (87.8 min, 74.2 s per epoch) | 0.390 | 0.104 | 0.0393 | 0.0184 (0.0206) |
| C2 | 0 | 48 | 38 | 3,534 s (58.9 min, 73.6 s per epoch) | 0.106 | 0.0708 | 0.0379 | 0.0204 (0.0233) |
| C2 | 1 | 92 | 82 | 6,723 s (112.0 min, 73.1 s per epoch) | 0.134 | 0.0612 | 0.0441 | 0.0177 (0.0197) |

- All four models stopped early (10 epochs without val-loss improvement); none reached the 100-epoch cap.
- **Loss spikes** (epoch-to-epoch jump above 1.3x in train or val loss):
  - C0 seed 0: epoch 15, train 0.0390 to 0.0666 (1.71x) and val 0.0313 to 0.0987 (3.15x); epoch 29, val 0.0270 to 0.0369 (1.37x).
  - C0 seed 1: epoch 35, val 0.0235 to 0.0345 (1.47x). No train-loss spike.
  - C2 seed 0: epoch 34, val 0.0242 to 0.0324 (1.34x); epoch 40, train 0.0230 to 0.0408 (1.77x) and val 0.0221 to 0.0457 (2.06x).
  - C2 seed 1: epoch 9, train 0.0403 to 0.1024 (2.54x) and val 0.0359 to 0.0527 (1.47x); epoch 85, train 0.0208 to 0.0362 (1.74x) and val 0.0180 to 0.0284 (1.58x).
  - Both C2 runs ended right after a spike: best epochs 38 and 82, spikes at 40 and 85, and no later epoch improved on the best val loss. In both C0 runs the last spike (epochs 29 and 35) came more than 20 epochs before the best epoch.
- **Faster start for C2:**
  - Epoch-1 val loss is 0.106 and 0.134 for C2, against 0.426 and 0.390 for C0. C0 starts near the trivial 0.44.
  - First epoch with val loss below 0.05: C2 epochs 7 and 4, C0 epochs 10 and 9.
  - By epoch 10 the curves are level (C2 0.0379 and 0.0441, C0 0.0401 and 0.0393).
- **Session time:** C0 started at 12:46 UTC and C2's JSON was written at 18:38 UTC, 5.9 h for both configs. The big test was built in 39.6 s (C0) and 35.9 s (C2). MWPM decoded it in 7.3 s and 9.9 s, and GNN-A scored it in about 28 s per seed.

**Zero-fired shots**
- Big test: 15,307 of 200,000 shots (7.65%) have no fired detector. None of them has a true flip, and no model predicts a flip on any of them (0 of 15,307 for both seeds of both configs).
- Primary test set: 567 shots, also 0 true flips and 0 predicted flips for all four models.
- For C2 the prediction on these shots is a single learned constant (the empty vector through the head); for C0 it comes from the graph with every fired bit 0.

## Interpretation (Phase 3, S1)
- **Data:**
  - Fired-only pooling did not help at 300k shots. It is significantly worse at seed 0, better but not significantly at seed 1, and fails the decision rule.
  - The spread between two seeds of the same config (C0: 1,348 vs 1,429 errors, p = 0.0034) is as large as the C2 vs C0 difference. A per-seed McNemar test compares two trained models, not two architectures. With two seeds, an architecture effect smaller than this spread cannot be told apart from training noise.
  - C2 learns much faster in the first epochs, but the gap is gone by epoch 10.
- **Hypotheses, not tested here:**
  - C2's faster start comes from removing the add pool over 121 mostly unfired nodes. At initialization C2's logit is about -1, against about +147 for C0 (local check, 2026-11-16, not from a run JSON).
  - C2's poor seed-0 result is partly training instability rather than the readout itself, since the run ended right after its largest spike. The data cannot separate the two.
- **Phase 1 consistency:** C2 never pools the boundary hub, yet it did not lose clearly. This does not contradict the Phase 1 finding that the label information sits on the hub's edges. The hub is one hop from every detector that has a boundary edge, so its information can still reach the fired nodes through message passing. S1 does not test the hub directly.
- **Decision:** C2 does not enter the combined config. The `--readout fired` option stays in the code, off by default.
- **Next:** S2. See the Phase 3 plan; the spikes under constant lr 1e-3 (all four models, one to two each) bear on what to run.

## Phase 3, S2 (training recipe)
- **Date:** 2026-11-17
- **Where:** Kaggle, GPU, one session, torch 2.11.0+cu130. Three runs back to back: the C0 rerun started at 02:38 UTC, and the C0_recipe JSON was written at 09:52 UTC (7.2 h).
- **Runs:**
  - **C0 rerun:** commit 5110582 (clean). Identical command to the S1 C0, outdir `/kaggle/working/results/C0`. The JSON was not downloaded from the Kaggle session. It was recovered from the notebook output, where a `!cat` cell had printed it in full. Its MWPM wrong-shot lists match the S1 runs exactly on both test sets.
  - **C2 rerun (partial):** identical command to the S1 C2 (`--readout fired`, outdir `/kaggle/working/results/C2`). The commit is 5110582 per the session plan; the log does not print it. The JSON (run_20261117T072706Z) was deleted on Kaggle. Only the printed log survives: per-epoch losses rounded to 4 decimals, the primary-set LER mean over seeds, and big-test errors for seed 0. No test_wrong_idx, so no paired statistics.
  - **C0_recipe:** commit 9b2a066 (clean). The same command as C0 plus `--lr-schedule cosine --lr-min 1e-5 --grad-clip 1.0`, outdir `/kaggle/working/results/C0_recipe`. 9b2a066 only adds analysis and result files on top of 5110582, so the training code is identical; only the flags differ.
- **Command (C0_recipe):** `python qec_run.py --d 5 --p 0.003 --shots 50000 --split-seed 12345 --train-shots 300000 --arms A --seeds 0 1 --epochs 100 --patience 10 --lr 1e-3 --lr-schedule cosine --lr-min 1e-5 --grad-clip 1.0 --big-test-shots 200000 --expect-mwpm-errors 36 --device cuda --outdir /kaggle/working/results/C0_recipe`
- **Setup:** as S1. GNN-A, default backbone (78,053 parameters), d=5, p=0.003, seeds 0 and 1, 300,000 train shots, up to 100 epochs, patience 10, the same primary test set (`--expect-mwpm-errors 36` passed) and the same big test (200,000 shots, seed 12347, 664 MWPM errors). None resumed.
- **Files:**
  - C0 rerun: results/run_20261117T044139Z_d5_p0.003.json (recovered from the notebook output)
  - C0_recipe: results/run_20261117T095238Z_d5_p0.003.json
  - C2 rerun: analysis/kaggle_logs/c2_rerun_log_20261117T072706Z.txt (log only)
- **Analysis:** analysis/phase3_results/s2_details.json (from analysis/phase3_s2_details.py). The S1 runs are referred to below as "C0, Nov 16" and "C2, Nov 16".

**Big test set (200,000 shots, seed 12347)**

| Run | Seed | Errors | LER | Ratio to MWPM (664 errors) | McNemar vs MWPM (A right & MWPM wrong / MWPM right & A wrong) |
|---|---|---|---|---|---|
| C0, Nov 16 | 0 | 1,348 | 0.006740 | 2.03 | 184 / 868, p = 1.1e-106 |
| C0, Nov 16 | 1 | 1,429 | 0.007145 | 2.15 | 189 / 954, p = 2.7e-123 |
| C0 rerun | 0 | 1,647 | 0.008235 | 2.48 | 182 / 1,165, p = 1.1e-175 |
| C0 rerun | 1 | 1,658 | 0.008290 | 2.50 | 191 / 1,185, p = 2.3e-175 |
| C0_recipe | 0 | 1,479 | 0.007395 | 2.23 | 192 / 1,007, p = 1.0e-133 |
| C0_recipe | 1 | 1,206 | 0.006030 | 1.82 | 188 / 730, p = 5.2e-76 |
| C2 rerun (partial) | 0 | 1,425 | 0.00713 | 2.15 | p = 6.3e-123 (counts not printed) |

- Statistical uncertainty per LER: about ±0.00018 to ±0.00020 (200,000 shots).
- The C2 rerun log prints the big test for seed 0 only.

**Paired McNemar, C0_recipe vs constant-lr C0, same seed** (fixed = constant-lr C0 wrong and C0_recipe right; newly wrong = the reverse)

| Reference | Seed | Set | Fixed | Newly wrong | Wrong in both | p |
|---|---|---|---|---|---|---|
| C0, Nov 16 | 0 | big | 314 | 445 | 1,034 | 2.2e-6 |
| C0, Nov 16 | 1 | big | 473 | 250 | 956 | 8.6e-17 |
| C0 rerun | 0 | big | 434 | 266 | 1,213 | 2.3e-10 |
| C0 rerun | 1 | big | 720 | 268 | 938 | 2.2e-48 |
| C0, Nov 16 | 0 | primary | 15 | 22 | 43 | 0.32 |
| C0, Nov 16 | 1 | primary | 19 | 12 | 35 | 0.28 |
| C0 rerun | 0 | primary | 14 | 15 | 50 | 1.0 |
| C0 rerun | 1 | primary | 31 | 13 | 34 | 0.0096 |

- Against the Nov 16 C0, C0_recipe is significantly worse at seed 0 and significantly better at seed 1 on the big test.
- Against the C0 rerun it is significantly better at both seeds.
- On the primary set, only seed 1 against the C0 rerun is significant.

**Primary test set (7,500 shots, MWPM 36 errors)**

| Run | Seed | Test errors | Test LER | Val LER | McNemar vs MWPM (A right & MWPM wrong / MWPM right & A wrong) |
|---|---|---|---|---|---|
| C0, Nov 16 | 0 | 58 | 0.0077 | 0.0075 | 10 / 32, p = 0.00094 |
| C0, Nov 16 | 1 | 54 | 0.0072 | 0.0067 | 13 / 31, p = 0.0096 |
| C0 rerun | 0 | 64 | 0.0085 | 0.0085 | 11 / 39, p = 9.0e-5 |
| C0 rerun | 1 | 65 | 0.0087 | 0.0084 | 12 / 41, p = 8.2e-5 |
| C0_recipe | 0 | 65 | 0.0087 | 0.0076 | 8 / 37, p = 1.5e-5 |
| C0_recipe | 1 | 47 | 0.0063 | 0.0063 | 10 / 21, p = 0.071 |

- Statistical uncertainty per LER: about ±0.0010 (7,500 test shots).
- C2 rerun (partial): test LER 0.0077 ± 0.0007 over 2 seeds as printed (C2, Nov 16: 0.0085 and 0.0080). McNemar vs MWPM at seed 0: p = 1.1e-4 (counts not printed).

**GPU run-to-run nondeterminism (same commit, same command, same seeds)**
- **C0 rerun vs C0, Nov 16** (fixed = Nov 16 wrong and rerun right):
  - Seed 0: 1,647 vs 1,348 big-test errors; 301 fixed vs 600 newly wrong, p = 1.2e-23. Primary set: 64 vs 58; 20 vs 26, p = 0.46.
  - Seed 1: 1,658 vs 1,429 big-test errors; 342 fixed vs 571 newly wrong, p = 3.3e-14. Primary set: 65 vs 54; 16 vs 27, p = 0.13.
- **C2 rerun vs C2, Nov 16** (partial, no paired statistics): seed 0 has 1,425 vs 1,477 big-test errors. Epochs run are 59 and 74 against 48 and 92.
- The data, the seed and the code are identical, so the differences come from nondeterministic GPU kernels. The reruns diverge from the first epoch: epoch-1 val loss is 0.4300 vs 0.4263 at seed 0 and 0.3756 vs 0.3902 at seed 1. On the big test, the same-seed reruns differ by 229 to 299 errors. That is more than the C2 vs C0 difference in S1 (129 errors at seed 0, 50 at seed 1).
- **Seed spread within one run** (big test, seed 0 vs seed 1):

| Run | Seed 0 errors | Seed 1 errors | Only seed 0 wrong | Only seed 1 wrong | p |
|---|---|---|---|---|---|
| C0, Nov 16 | 1,348 | 1,429 | 332 | 413 | 0.0034 |
| C0 rerun | 1,647 | 1,658 | 364 | 375 | 0.71 |
| C0_recipe | 1,479 | 1,206 | 532 | 259 | 1.5e-22 |

**Training**

| Run | Seed | epochs_run | best_epoch | Wall time | Epoch-1 val loss | Best val loss (train loss at that epoch) | Val spikes (>1.3x) |
|---|---|---|---|---|---|---|---|
| C0, Nov 16 | 0 | 69 | 59 | 5,204 s | 0.426 | 0.0183 (0.0219) | epoch 15 (3.15x), epoch 29 (1.37x) |
| C0, Nov 16 | 1 | 71 | 61 | 5,269 s | 0.390 | 0.0184 (0.0206) | epoch 35 (1.47x) |
| C0 rerun | 0 | 48 | 38 | 3,538 s | 0.430 | 0.0210 (0.0263) | none |
| C0 rerun | 1 | 50 | 40 | 3,691 s | 0.376 | 0.0215 (0.0249) | none |
| C0_recipe | 0 | 48 | 38 | 3,589 s | 0.104 | 0.0189 (0.0219) | none |
| C0_recipe | 1 | 66 | 56 | 4,951 s | 0.110 | 0.0168 (0.0175) | none |
| C2, Nov 16 | 0 | 48 | 38 | 3,534 s | 0.106 | 0.0204 (0.0233) | epoch 34 (1.34x), epoch 40 (2.06x) |
| C2, Nov 16 | 1 | 92 | 82 | 6,723 s | 0.134 | 0.0177 (0.0197) | epoch 9 (1.47x), epoch 85 (1.58x) |
| C2 rerun (log) | 0 | 59 | 49 | not printed | 0.1199 | 0.0191 | epoch 10 (1.49x), epoch 34 (1.30x) |
| C2 rerun (log) | 1 | 74 | 64 | not printed | 0.1188 | 0.0177 | epoch 9 (2.01x) |

- All runs stopped early (10 epochs without val-loss improvement). For the C2 rerun, best_epoch is the epoch with the lowest printed val loss.
- **Train-loss spikes (>1.3x):**
  - C0 rerun and C0_recipe: none.
  - C2 rerun: seed 0 epoch 7 (0.0494 to 0.0759, 1.54x); seed 1 epoch 9 (0.0395 to 0.0584, 1.48x) and epoch 69 (0.0197 to 0.0310, 1.57x; val 1.28x, below the threshold).
  - The Nov 16 runs are listed in the S1 entry.
- The C2 rerun's seed-0 epoch-34 val spike is 0.0233 to 0.0304 in the printed 4-decimal values, a ratio of 1.305. It is at the threshold within rounding.
- **Gradient norms** (measured before clipping; 1,172 steps per epoch):
  - C0_recipe seed 0: epoch 1 mean 32.5, max 9,355, 91.4% of steps clipped. From epoch 2 on, the mean of the per-epoch means is 0.48, the largest norm 13.8, and 7.1% of steps are clipped. The last epoch has 0.9% clipped. Over the whole run, 8.8% of steps are clipped.
  - C0_recipe seed 1: epoch 1 mean 19.9, max 4,419, 93.3% clipped. From epoch 2 on: mean 0.39, largest norm 35.0, 4.1% clipped. The last epoch has 1.0% clipped. Over the whole run, 5.4% clipped.
  - C0 rerun, C2 rerun and the Nov 16 runs: not recorded (no --grad-clip).
- **lr at early stop (C0_recipe):**
  - Cosine is over the 100-epoch cap, from 1e-3 towards lr_min 1e-5.
  - Seed 0 stopped at epoch 48 with lr 5.52e-4 (55% of the initial lr); the best epoch 38 ran at 7.02e-4.
  - Seed 1 stopped at epoch 66 with lr 2.80e-4 (28%); the best epoch 56 ran at 4.28e-4.
  - The recorded values match the cosine formula. Neither run got near the low-lr end of the schedule, because early stopping ended both before epoch 70.

**Zero-fired shots:** 15,307 big-test shots; 0 true flips and 0 predicted flips for every seed of the C0 rerun and C0_recipe, as in S1.

## Interpretation (Phase 3, S2)
- **What the recipe changed (data):**
  - No loss spikes in either C0_recipe seed.
  - The first epoch is much better: val loss 0.104 and 0.110 against 0.38 to 0.43 for constant-lr C0.
  - In epoch 1 clipping was active on more than 90% of steps, with gradient norms up to 9,355. After that it clipped 4 to 7% of steps. So the clip mainly tames the very large early gradients. That is consistent with the large initial logits of the unnormalized backbone (about +147 at initialization, local check from Phase 2).
- **What it did not show (data):**
  - The C0 rerun at constant lr also had no spikes, so a run without spikes is not by itself evidence for the recipe. Spikes vary from run to run.
  - The recipe did not narrow the seed spread: 1,479 vs 1,206 errors, p = 1.5e-22, the widest of the three C0 variants.
  - C0_recipe is better than the C0 rerun at both seeds, but against the Nov 16 C0 it is worse at seed 0 and better at seed 1.
  - Cosine never reached its low-lr phase, so the recipe tested here is in practice "clip plus a mild lr decay".
- **Nondeterminism and the decision rule (data, then consequence):**
  - Rerunning the same commit, command and seed on Kaggle changed big-test errors by 229 to 299, with per-seed McNemar p = 1.2e-23 and 3.3e-14.
  - A per-seed McNemar test on one pair of runs therefore measures two particular trained models. It cannot attribute a difference to the config, and with differences this large it will find "significant" effects in either direction. The S1 C2 vs C0 comparison (129 and 50 errors) is within this run-to-run noise. The S1 decision on C2 stands as "no evidence of a benefit", not as evidence of harm.
  - The rule "fewer errors and p < 0.05 at every seed" is not a valid test of a config effect under this noise.
- **Hypothesis, not tested:** much of the run-to-run spread enters through early stopping. The reruns stopped at different points (C0 rerun at epochs 48 and 50 against 69 and 71) and are scored at different best epochs. A schedule that reaches a low lr before stopping, or a fixed epoch budget without early stopping, might reduce it.
- **Next:** decide on a new comparison protocol before more ablation sessions. For example: several independent runs per config, comparing big-test error counts across runs rather than shots within one pair, or deterministic GPU training if it can be made to work with the PyG scatter ops.

## Phase 3, S3 (recipe v2: C0 and C4)
- **Date:** 2026-11-22 to 2026-11-23
- **Where:** Kaggle, GPU, torch 2.11.0+cu130. Two runs back to back: C0 started at 16:37 UTC and its JSON was written at 20:27 UTC; C4 started at 20:27 UTC and its JSON was written at 00:35 UTC.
- **Runs:**
  - **C0, recipe v2:** commit 76da5a7 (clean). Default backbone (norm none, no residual, 78,053 parameters). Outdir `/kaggle/working/results/C0_recipe_v2`.
  - **C4, recipe v2:** commit 76da5a7 (clean). Same command plus `--residual --norm layer` (78,821 parameters). Outdir `/kaggle/working/results/C4_recipe_v2`. C4 changes two things at once, the residual connections and LayerNorm.
  - 76da5a7 only adds analysis and result files on top of 9b2a066 (and so on top of 5110582), so the training code is the same as in S1 and S2.
- **Recipe v2:** cosine over a 60-epoch cap, from lr 1e-3 to lr_min 1e-5, `--grad-clip 1.0`, patience 60. Patience equals the cap, so early stopping cannot fire: every run trains all 60 epochs and the cosine schedule runs to its end. As before, the model is scored at the epoch with the lowest val loss (best_epoch).
- **Command (C4):** `python qec_run.py --d 5 --p 0.003 --shots 50000 --split-seed 12345 --train-shots 300000 --arms A --seeds 0 1 2 --epochs 60 --patience 60 --lr 1e-3 --lr-schedule cosine --lr-min 1e-5 --grad-clip 1.0 --residual --norm layer --big-test-shots 200000 --expect-mwpm-errors 36 --device cuda --outdir /kaggle/working/results/C4_recipe_v2`. The C0 command is the same without `--residual --norm layer`, with outdir `C0_recipe_v2`.
- **Setup:** GNN-A, d=5, p=0.003, seeds 0, 1 and 2, 300,000 train shots. Same primary test set (`--expect-mwpm-errors 36` passed) and same big test (200,000 shots, seed 12347, 664 MWPM errors). The MWPM wrong-shot lists match S2 and Run 11 exactly on both test sets. Neither run was resumed.
- **Runs per config:** three seeds, so three independent training runs per config. Under the S2 finding, "seed" is a label for a run rather than a controlled variable.
- **Files:**
  - C0, recipe v2: results/run_20261122T202714Z_d5_p0.003.json
  - C4, recipe v2: results/run_20261123T003558Z_d5_p0.003.json
- **Analysis:** analysis/phase3_results/s3_details.json (from analysis/phase3_s3_details.py). "C0_recipe" below is the S2 run (results/run_20261117T095238Z_d5_p0.003.json).

**Big test set (200,000 shots, seed 12347)**

| Run | Seed | Errors | LER | Ratio to MWPM (664 errors) | McNemar vs MWPM (A right & MWPM wrong / MWPM right & A wrong) |
|---|---|---|---|---|---|
| C0, recipe v2 | 0 | 1,272 | 0.006360 | 1.92 | 184 / 792, p = 2.0e-90 |
| C0, recipe v2 | 1 | 1,249 | 0.006245 | 1.88 | 195 / 780, p = 2.1e-83 |
| C0, recipe v2 | 2 | 1,184 | 0.005920 | 1.78 | 191 / 711, p = 4.6e-71 |
| C4, recipe v2 | 0 | 968 | 0.004840 | 1.46 | 176 / 480, p = 1.8e-33 |
| C4, recipe v2 | 1 | 972 | 0.004860 | 1.46 | 193 / 501, p = 2.0e-32 |
| C4, recipe v2 | 2 | 986 | 0.004930 | 1.48 | 175 / 497, p = 1.3e-36 |

| Run | Mean errors ± SD across 3 runs | Range | Mean ratio to MWPM |
|---|---|---|---|
| C0, recipe v2 | 1,235 ± 45.6 | 1,184 to 1,272 (88) | 1.86 |
| C4, recipe v2 | 975.3 ± 9.5 | 968 to 986 (18) | 1.47 |

- SD is the sample SD (n-1) over the three runs. Statistical uncertainty per LER: about ±0.00016 to ±0.00018 (200,000 shots).

**Primary test set (7,500 shots, MWPM 36 errors)**

| Run | Seed | Test errors | Test LER | Ratio to MWPM | Val LER | McNemar vs MWPM (A right & MWPM wrong / MWPM right & A wrong) |
|---|---|---|---|---|---|---|
| C0, recipe v2 | 0 | 54 | 0.0072 | 1.50 | 0.0076 | 8 / 26, p = 0.0029 |
| C0, recipe v2 | 1 | 54 | 0.0072 | 1.50 | 0.0068 | 11 / 29, p = 0.0064 |
| C0, recipe v2 | 2 | 51 | 0.0068 | 1.42 | 0.0067 | 11 / 26, p = 0.020 |
| C4, recipe v2 | 0 | 41 | 0.0055 | 1.14 | 0.0055 | 9 / 14, p = 0.40 |
| C4, recipe v2 | 1 | 36 | 0.0048 | 1.00 | 0.0051 | 15 / 15, p = 1.0 |
| C4, recipe v2 | 2 | 38 | 0.0051 | 1.06 | 0.0059 | 8 / 10, p = 0.81 |

- Across runs: C0 53.0 ± 1.7 errors (51 to 54), mean ratio 1.47; C4 38.3 ± 2.5 errors (36 to 41), mean ratio 1.06.
- Statistical uncertainty per LER: about ±0.0008 to ±0.0010 (7,500 test shots).

**C4 vs C0 under recipe v2**
- **Decision rule (from S3 on):** config B beats config A only if every B run beats every A run on the big test (3 runs each), and the mean gap is larger than the run-to-run range within either config. Per-seed McNemar is reported but is not decisive.
- **Applied to C4 vs C0: C4 beats C0.**
  - Every C4 run beats every C0 run: the worst C4 run has 986 errors and the best C0 run has 1,184.
  - The mean gap is 259.7 errors. The run-to-run range is 88 for C0 and 18 for C4, so the gap is larger than both.
- **Same seed, paired McNemar** (reported, not decisive; fixed = C0 wrong and C4 right; newly wrong = the reverse):

| Seed | Set | Fixed | Newly wrong | Wrong in both | p |
|---|---|---|---|---|---|
| 0 | big | 554 | 250 | 718 | 2.8e-27 |
| 1 | big | 557 | 280 | 692 | 6.0e-22 |
| 2 | big | 471 | 273 | 713 | 3.7e-13 |
| 0 | primary | 23 | 10 | 31 | 0.035 |
| 1 | primary | 30 | 12 | 24 | 0.0079 |
| 2 | primary | 23 | 10 | 28 | 0.035 |

- **Cross-run comparison** (the three C4 runs against the three C0 runs, as independent samples; seeds are not matched):
  - Big test: every C4 run has fewer errors than every C0 run (C4 at most 986, C0 at least 1,184). The mean difference is 259.7 errors. Exact Mann-Whitney U = 0: one-sided p = 0.05, two-sided p = 0.10, which are the smallest values possible with 3 runs against 3. Welch t-test, two-sided: p = 0.0080.
  - Big test, all 9 cross pairs (each C4 run against each C0 run, paired McNemar): C4 has fewer errors in all 9, with p between 2.8e-27 and 3.7e-13.
  - Primary set: every C4 run is below every C0 run (C4 at most 41, C0 at least 51); mean difference 14.7 errors; Mann-Whitney one-sided p = 0.05, two-sided p = 0.10; Welch two-sided p = 0.0019. In the 9 cross pairs C4 has fewer errors in all 9, with p < 0.05 in 7 (the other two: 0.053 and 0.13).

**C0 recipe v2 vs C0_recipe (S2)**
- **What differs:** the epoch cap (60 vs 100) and patience (60 vs 10). In S2, early stopping ended the runs at epochs 48 and 66, with lr still at 55% and 28% of the initial value. In recipe v2 every run reached epoch 60 with lr 1.07e-5 (1.1% of the initial value). Same commit lineage, flags otherwise identical.
- **Same seed, paired McNemar** (fixed = C0_recipe wrong and C0 recipe v2 right):

| Seed | Set | C0_recipe errors | Recipe v2 errors | Fixed | Newly wrong | Wrong in both | p |
|---|---|---|---|---|---|---|---|
| 0 | big | 1,479 | 1,272 | 476 | 269 | 1,003 | 3.1e-14 |
| 1 | big | 1,206 | 1,249 | 320 | 363 | 886 | 0.11 |
| 0 | primary | 65 | 54 | 24 | 13 | 41 | 0.099 |
| 1 | primary | 47 | 54 | 9 | 16 | 38 | 0.23 |

- **Across runs, big test:** C0_recipe 1,342.5 ± 193.0 (2 runs, range 273); C0 recipe v2 1,235 ± 45.6 (3 runs, range 88).
- **Run-to-run disagreement on the big test** (shots wrong in only one of two runs of the same config):

| Run | Pair | Only first wrong | Only second wrong | p |
|---|---|---|---|---|
| C0_recipe | seed 0 vs 1 | 532 | 259 | 1.5e-22 |
| C0, recipe v2 | seed 0 vs 1 | 366 | 343 | 0.41 |
| C0, recipe v2 | seed 0 vs 2 | 362 | 274 | 5.5e-4 |
| C0, recipe v2 | seed 1 vs 2 | 377 | 312 | 0.015 |
| C4, recipe v2 | seed 0 vs 1 | 216 | 220 | 0.89 |
| C4, recipe v2 | seed 0 vs 2 | 201 | 219 | 0.41 |
| C4, recipe v2 | seed 1 vs 2 | 238 | 252 | 0.56 |

- **Nondeterminism at epoch 1:** C0 recipe v2 and C0_recipe have the same seeds and the same lr in epoch 1 (1e-3). The largest epoch-1 gradient norm is the same in both (9,354.8 at seed 0; 4,419.2 and 4,419.3 at seed 1), yet epoch-1 val loss is 0.124 vs 0.104 at seed 0 and 0.121 vs 0.110 at seed 1. The runs diverge within the first epoch, as in S2.

**Training**

| Run | Seed | epochs_run | best_epoch | Wall time | Epoch-1 val loss | Best val loss (train loss at that epoch) | lr at best epoch | Final lr (epoch 60) | Val spikes (>1.3x) |
|---|---|---|---|---|---|---|---|---|---|
| C0, recipe v2 | 0 | 60 | 56 | 4,480 s | 0.124 | 0.0187 (0.0179) | 2.69e-5 | 1.07e-5 | none |
| C0, recipe v2 | 1 | 60 | 46 | 4,523 s | 0.121 | 0.0172 (0.0179) | 1.55e-4 | 1.07e-5 | none |
| C0, recipe v2 | 2 | 60 | 52 | 4,590 s | 0.115 | 0.0182 (0.0162) | 6.40e-5 | 1.07e-5 | epoch 6 (0.0422 to 0.1018, 2.41x) |
| C4, recipe v2 | 0 | 60 | 45 | 4,896 s | 0.164 | 0.0156 (0.0130) | 1.74e-4 | 1.07e-5 | epoch 18 (0.0175 to 0.0245, 1.40x), epoch 31 (0.0163 to 0.0223, 1.37x) |
| C4, recipe v2 | 1 | 60 | 42 | 4,885 s | 0.170 | 0.0143 (0.0126) | 2.35e-4 | 1.07e-5 | none |
| C4, recipe v2 | 2 | 60 | 55 | 4,892 s | 0.149 | 0.0151 (0.0110) | 3.42e-5 | 1.07e-5 | none |

- **Train-loss spikes (>1.3x):** none in any of the six runs.
- **lr:** the recorded lr matches the per-epoch cosine formula with T_max = 60 in every run. All best epochs (42 to 56) fall in the second half of the schedule, at lr between 2.7e-5 and 2.4e-4.
- **Gradient norms** (measured before clipping; 1,172 steps per epoch):

| Run | Seed | Epoch 1: mean, max, clipped | From epoch 2: mean of epoch means, max, clipped | Last epoch clipped |
|---|---|---|---|---|
| C0, recipe v2 | 0 | 32.0, 9,355, 82.5% | 0.44, 18.9, 5.1% | 0.2% |
| C0, recipe v2 | 1 | 19.3, 4,419, 92.3% | 0.36, 15.8, 3.1% | 0.6% |
| C0, recipe v2 | 2 | 17.0, 2,665, 92.6% | 0.48, 32.8, 6.7% | 0.9% |
| C4, recipe v2 | 0 | 353.5, 2,239, 100% | 0.96, 572, 7.2% | 0.6% |
| C4, recipe v2 | 1 | 249.8, 2,386, 100% | 0.64, 208, 6.0% | 0.2% |
| C4, recipe v2 | 2 | 269.5, 2,078, 100% | 0.58, 123, 5.7% | 0.4% |

**Zero-fired shots:** 15,307 big-test shots; 0 true flips and 0 predicted flips for every run.

**Context: C4 at 300k against the 1M baseline.** Run 11 (1,000,000 train shots, default backbone, constant lr, no clipping, seed 0, a single run) has 1,021 big-test errors (ratio 1.54). The three C4 runs have 968, 972 and 986. Paired McNemar against Run 11 (fixed = Run 11 wrong and C4 right): 395 vs 342 (p = 0.055), 410 vs 361 (p = 0.084), 410 vs 375 (p = 0.22). The three C0 recipe v2 runs are all worse than Run 11 (p between 8.8e-19 and 5.7e-9). Run 11 is one run, and it differs from C4 in data, recipe and architecture.

## Interpretation (Phase 3, S3)
- **C4 vs C0 (data):**
  - C4 passes the decision rule: every C4 run beats every C0 run on the big test, and the mean gap (259.7 errors) is larger than the range within either config (88 and 18).
  - Under recipe v2, all three C4 runs beat all three C0 runs on both test sets. On the big test the gap is 259.7 errors on average (1.47x MWPM against 1.86x), and the closest pair is still 198 errors apart (986 vs 1,184).
  - Every one of the 9 cross pairs on the big test is significant in favour of C4. The same-seed McNemar agrees at every seed on both sets.
  - With 3 runs per config, the exact rank test cannot go below one-sided p = 0.05, and it reaches that bound.
  - On the primary set C4 is level with MWPM (36 to 41 errors against 36; no seed significant). On the big test it is still clearly worse than MWPM (p ≤ 2.0e-32 at every seed).
- **C4 vs C0 (what it does not show):**
  - C4 changes two things (residual connections and LayerNorm). These runs cannot say which one matters, or whether both are needed.
  - The C4 runs are not more stable in training: epoch-1 val loss is worse (0.149 to 0.170 against 0.115 to 0.124), every epoch-1 step was clipped, and later gradient norms reach 123 to 572 against 16 to 33 for C0. One C4 run has two val spikes.
- **Recipe v2 vs the S2 recipe (data):**
  - At seed 0 recipe v2 is significantly better (1,272 vs 1,479, p = 3.1e-14). At seed 1 it is slightly worse, not significantly (1,249 vs 1,206, p = 0.11).
  - The spread between runs is much smaller: range 88 errors over 3 runs against 273 over 2, and the seed-0 vs seed-1 disagreement is balanced (366 vs 343, p = 0.41) against 532 vs 259 (p = 1.5e-22) in S2. C4 under recipe v2 has the smallest spread so far (range 18; no seed pair significant).
  - The full schedule cannot be separated from removing early stopping: recipe v2 changes both at once.
- **Nondeterminism (data):** epoch 1 still diverges between same-seed runs (0.124 vs 0.104 val loss at seed 0). Recipe v2 did not make training deterministic, and no same-seed rerun under recipe v2 exists yet, so its run-to-run noise has not been measured directly.
- **Against the noise scale from S2 (consequence):** the C4 gap (259.7 errors on average) is about the size of the S2 same-seed rerun differences (229 to 299 errors). Those were measured under constant lr with early stopping, so they are not the noise scale for recipe v2. Within recipe v2 the run-to-run ranges (88 for C0, 18 for C4) are well below the gap. The decision rule compares against these within-config ranges, not against the S2 numbers.
  - The range term of the rule uses only 3 runs per config. With so few runs, the range can understate the true run-to-run spread.
- **Hypotheses, not tested:**
  - Training to a low lr without early stopping shrinks the run-to-run spread. This is consistent with the S2 hypothesis and with the ranges above, but different seeds also differ by nondeterminism, so a same-seed rerun under recipe v2 is needed to measure it.
  - The C4 gain comes from LayerNorm controlling the large activations of the unnormalized backbone, rather than from the residual connections. The gradient norms above do not support a simple version of this (C4's are larger), so it needs the two parts ablated separately.
  - C4 at 300k matching Run 11 at 1M suggests the architecture and recipe are worth at least as much as a 3.3x data increase here. Run 11 is one run under the old recipe, so this is a lead, not a result.
- **Next:** adopt recipe v2 and C4 as the new baseline; ablate residual and LayerNorm separately (3 runs each); one same-seed rerun under recipe v2 to measure its nondeterminism; then C4 at 1M shots.

## 2026-11-23: canonical stored eval set and the correlated-matching baseline (no training)
- **Why:** stim's seeded sampling is not portable across platforms. The same seed and stim version gave different big-test shots on Windows (MWPM 690 errors) and on Kaggle Linux (664). The canonical big test is therefore stored once and loaded with `--eval-set`.
- **File:** data/eval_d5_p0.003_seed12347_200k.npz (2,325,806 bytes). sha256 `4b9504da61a4404c3a07bed770cb33c6568a8c014f45d20562e6d5fabc292772`.
- **Provenance (file header):**
  - Exported with `export_eval_set.py` on Kaggle Linux (Linux-6.18.55+-x86_64-with-glibc2.39), 2026-11-23 12:41 UTC.
  - Commit 73a2e41 (clean), stim 1.16.0, pymatching 2.4.0.
  - d=5, p=0.003, rounds 5, 200,000 shots, sampler seed 12347, sampled in chunks of 10,000, exactly as `--big-test-shots 200000`. 120 detectors, 1,677 DEM errors.
- **Same shots as the earlier big tests:** MWPM's wrong-shot list on the stored shots is identical to the big-test MWPM list in the Phase 3 S3 JSONs (C0 and C4, recipe v2). Paired comparisons between this file and runs since Run 11 are valid.
- **MWPM:** 664 errors (LER 0.00332), stored in the file and reproduced on Windows with identical wrong shots.
- **Correlated matching:** 475 errors (LER 0.002375), 0.715x MWPM. This is PyMatching 2.4.0's two-pass correlated matching on the decomposed DEM, with `enable_correlations=True` set on both the Matching and the decode. Paired with MWPM: it fixes 350 of MWPM's errors and adds 161 new ones, p = 4e-17. Decoded on Windows. Future Kaggle runs record it themselves in the `correlated_matching` block.
- **C4 and C0 (recipe v2, S3) against correlated matching** (paired, same shots):

| Run | Seed | Errors | Ratio to correlated matching (475) | GNN right & corr wrong / corr right & GNN wrong | p |
|---|---|---|---|---|---|
| C4 | 0 | 968 | 2.04 | 186 / 679 | 1.3e-66 |
| C4 | 1 | 972 | 2.05 | 197 / 694 | 1.5e-65 |
| C4 | 2 | 986 | 2.08 | 175 / 686 | 3.6e-72 |
| C0 | 0 | 1,272 | 2.68 | 192 / 989 | 1.1e-129 |
| C0 | 1 | 1,249 | 2.63 | 199 / 973 | 9.2e-123 |
| C0 | 2 | 1,184 | 2.49 | 198 / 907 | 7.8e-109 |

- **Loader check on Windows:** `--eval-set` loading first failed on this Windows machine at the strict DEM fingerprint check. The file records sha256 `a48a3ea9...` of `str(dem)` on Kaggle Linux; Windows builds `cf79d2f6...` from the same commit and stim version. Every other check passes here:
  - the stored detector and observable bits are exactly the effect of the stored errors under the Windows DEM, so all 1,677 mechanisms have the same order and targets;
  - MWPM reproduces 664 errors with identical wrong shots.
  - So the difference is in the printed probability digits (`str(dem)` prints full float precision, for example `0.0075628575999998`), most likely last-digit rounding differences between platforms. This is not confirmed: the file stores only the hash.
  - **Loader change:** a fingerprint mismatch is now a warning, not a failure, but only when the bit-consistency and MWPM checks both pass. If either fails, loading still fails, and the error says the fingerprint also differs. The loader prints each check's result, and runs record them in `big_test.eval_set.checks`. On Windows the canonical file now loads with `dem_fingerprint FAIL` (warning) and the three other checks passing.
- **From now on:** every run uses `--eval-set data/eval_d5_p0.003_seed12347_200k.npz` in place of `--big-test-shots 200000`.

## Phase 3, S4 (auxiliary targets and fresh training data, on C4)
- **Date:** 2026-11-23 to 2026-11-24
- **Where:** Kaggle, GPU, torch 2.11.0+cu130, stim 1.16.0. Two separate sessions.
- **Runs:**
  - **aux_C4_v2:** commit 986a728 (clean), started 10:21 UTC. Raw and ZX arms, seeds 0 1 2. 986a728 only adds analysis and result files on top of 76da5a7 (the S3 C4 commit), so the training code is the S3 code.
  - **C4_fresh_v2:** commit ed15ffa (clean), started 19:41 UTC. A arm, seeds 0 1 2, `--fresh-train`.
- **Commands:**
  - aux: `python qec_run.py --d 5 --p 0.003 --shots 50000 --split-seed 12345 --train-shots 300000 --arms Raw ZX --seeds 0 1 2 --lambdas 0.1 --aux-data-qubits-only --aux-class-weight inv-freq --residual --norm layer --epochs 60 --patience 60 --lr 1e-3 --lr-schedule cosine --lr-min 1e-5 --grad-clip 1.0 --big-test-shots 200000 --expect-mwpm-errors 36 --device cuda --outdir /kaggle/working/results/aux_C4_v2`
  - fresh: `python qec_run.py --d 5 --p 0.003 --shots 50000 --split-seed 12345 --train-shots 300000 --fresh-train --arms A --seeds 0 1 2 --residual --norm layer --epochs 60 --patience 60 --lr 1e-3 --lr-schedule cosine --lr-min 1e-5 --grad-clip 1.0 --eval-set data/eval_d5_p0.003_seed12347_200k.npz --expect-mwpm-errors 36 --device cuda --outdir /kaggle/working/results/C4_fresh_v2`
- **Setup:** both on the C4 backbone (residual + LayerNorm) with recipe v2, as S3 C4.
  - **aux:** λ fixed at 0.1 (no grid), aux head on the 25 data qubits, inverse-frequency class weights from the training split. Raw weights I/X/Y/Z 0.26 / 10.98 / 17.19 / 18.92, ZX 0.27 / 7.59 / 17.05 / 11.59. 107,721 parameters with the aux head (78,821 without; the head is used in training only). Same fixed 300,000 training shots as S3 (35,000 base + 265,000 from seed 12346).
  - **fresh:** 300,000 new shots every epoch, epoch e from sampler seed 13,345,000 + e (13,345,001 to 13,345,060), recorded in `provenance.fresh_train`. No fixed training set.
- **Test sets:** the same primary set (`--expect-mwpm-errors 36` passed) and the same 200,000 big-test shots. The aux run sampled its big test on Kaggle (seed 12347); its MWPM wrong-shot list equals the stored eval set's. The fresh run loaded the stored eval set with all four checks passing, including the strict DEM fingerprint (Kaggle Linux). Its JSON records correlated matching as 475 errors, the same wrong shots as the canonical value.
- **Compared against:** the S3 C4 runs (results/run_20261123T003558Z_d5_p0.003.json), "C4" below. Same code, recipe, backbone and test shots.
- **Files:**
  - aux: results/run_20261123T184608Z_d5_p0.003.json
  - fresh: results/run_20261123T232850Z_d5_p0.003.json, with results/run_20261123T232850Z_d5_p0.003_weights/ (3 files, 327,787 bytes each) and results/run_20261123T232850Z_d5_p0.003_bigtest_probs.npz (2,024,720 bytes)

**Big test set (200,000 shots, MWPM 664 errors, correlated matching 475)**

| Run | Seed 0 | Seed 1 | Seed 2 | Mean | Range | Ratio to MWPM | Ratio to corr. matching |
|---|---|---|---|---|---|---|---|
| C4 (S3) | 968 | 972 | 986 | 975.3 | 18 | 1.47 | 2.05 |
| C4 + aux raw | 1,133 | 1,165 | 1,060 | 1,119.3 | 105 | 1.69 | 2.36 |
| C4 + aux ZX | 1,036 | 1,161 | 1,112 | 1,103.0 | 125 | 1.66 | 2.32 |
| C4 + fresh data | 921 | 823 | 848 | 864.0 | 98 | 1.30 | 1.82 |

- Paired against MWPM, every run of every arm is significantly worse (fresh: 168 / 425, 176 / 335, 159 / 343; p from 1.8e-12 to 1.2e-26). The same holds against correlated matching (fresh: p from 1.8e-38 to 3.3e-56).
- Zero-fired shots: 15,307; 0 true and 0 predicted flips for every run, as before.

**Decision rule** (every B run beats every A run on the big test, and the mean gap exceeds the range within either config)
- **Fresh data beats C4.** The worst fresh run (921) is below the best C4 run (968). The mean gap of 111.3 exceeds both ranges (98 and 18). It passes, but the gap only just exceeds the fresh runs' own range.
- **C4 beats both aux arms.** The worst C4 run (986) is below the best Raw (1,060) and ZX (1,036) runs. Gaps 144.0 (Raw) and 127.7 (ZX), above all ranges. The auxiliary target makes the model worse.
- **ZX vs Raw: no difference.** Gap 16.3 errors, ranges 105 and 125; neither config's runs all beat the other's.
- **Same-seed paired McNemar** (reported, not decisive; fixed = first wrong & second right):

| Second vs first | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| fresh vs C4 | 285 / 238, p = 0.044 | 284 / 135, p = 2.7e-13 | 299 / 161, p = 1.2e-10 |
| ZX vs C4 | 264 / 332, p = 0.006 | 267 / 456, p = 2.0e-12 | 286 / 412, p = 2.1e-6 |
| Raw vs C4 | 238 / 403, p = 7.4e-11 | 252 / 445, p = 2.5e-13 | 233 / 307, p = 0.0017 |
| ZX vs Raw | 380 / 283, p = 1.9e-4 | 325 / 321, p = 0.91 | 306 / 358, p = 0.048 |

The ZX vs Raw rows go in both directions across seeds, which is what run-to-run noise looks like.

**Primary test set (7,500 shots, MWPM 36 errors, correlated matching 21)**

| Run | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| C4 (S3) | 41 | 36 | 38 |
| C4 + aux raw | 52 | 53 | 45 |
| C4 + aux ZX | 47 | 50 | 49 |
| C4 + fresh data | 40 | 30 | 35 |

- Fresh vs MWPM: 7 / 11 (p = 0.48), 9 / 3 (p = 0.15), 5 / 4 (p = 1.0). Level with MWPM on this set; seed 1 has fewer errors than MWPM, not significantly. As before, this set is too small to resolve the gap the big test shows.
- ZX vs MWPM: p = 0.013, 0.020, 0.019 (worse). ZX vs Raw: p = 0.42, 0.73, 0.54.

**Aux heads (test set, selected λ 0.1; not collapsed)**
- Raw: non-identity recall 0.917 / 0.917 / 0.915. Of the slots it predicts as non-identity, 43.0% / 43.5% / 44.3% have the exact Pauli right. The true non-identity density is 0.051, and it predicts non-identity on 0.109 / 0.107 / 0.105 of slots.
- ZX: recall 0.892 / 0.888 / 0.892, precision 40.4% / 38.5% / 39.2%, true density 0.069, predicted 0.152 / 0.158 / 0.156.
- So the heads learn real structure (precision about 6 to 9x the base rate) and over-predict non-identity about 2x, as the class weights push them to.

**Training**

| Run | Seed | best_epoch (of 60) | Epoch-1 val loss | Best val loss (train loss at that epoch) | Val spikes (>1.3x) | s / epoch |
|---|---|---|---|---|---|---|
| C4 (S3) | 0 / 1 / 2 | 45 / 42 / 55 | 0.164 / 0.170 / 0.149 | 0.0156 (0.0130) / 0.0143 (0.0126) / 0.0151 (0.0110) | 18, 31 / none / none | 81.6 / 81.4 / 81.5 |
| C4 + aux raw | 0 / 1 / 2 | 51 / 48 / 43 | 0.180 / 0.181 / 0.178 | 0.0450 (0.0410) / 0.0450 (0.0407) / 0.0429 (0.0420) | none | 83.0 / 83.2 / 83.2 |
| C4 + aux ZX | 0 / 1 / 2 | 46 / 42 / 48 | 0.198 / 0.173 / 0.207 | 0.0509 (0.0502) / 0.0535 (0.0512) / 0.0519 (0.0501) | none | 83.2 / 83.2 / 83.1 |
| C4 + fresh data | 0 / 1 / 2 | 60 / 58 / 60 | 0.174 / 0.152 / 0.154 | 0.0150 (0.0142) / 0.0136 (0.0139) / 0.0130 (0.0135) | none / 7 / none | 74.8 / 74.7 / 74.7 |

- The aux losses include λ times the aux cross-entropy, so they are not comparable with the A-arm losses.
- **Fresh data:**
  - Train loss at the best epoch is now level with val loss (C4 on the fixed set: train 0.011 to 0.013 below val 0.014 to 0.016).
  - The best epochs sit at the end of the schedule (58 to 60), so the runs were still improving when the cosine schedule ended.
  - Epochs are 8% faster than C4's (batches built from detector bits, no PyG objects or collation).
- **GNN inference on the big test:** 29.7 to 30.8 s per run (fresh), 30.1 to 30.3 s (aux).

**Saved models (fresh run): checked on another machine**
- The weights and probabilities came from the Kaggle output bundle. The individual downloads are identical to the bundle's files. The bundle's JSON has the same content as results/run_20261123T232850Z_d5_p0.003.json; only the whitespace differs.
- Reloading each seed's weights on Windows CPU (`torch.load(..., weights_only=True)`) and predicting the stored eval set reproduces the run's wrong-shot list exactly (921, 823, 848). The probabilities agree with the saved GPU values to within 3e-6.
- **Ensembles of the 3 fresh runs:** averaging the saved probabilities gives 826 errors, majority vote 819. The best single run makes 823. Ensemble vs that run: 79 fixed, 82 newly wrong, p = 0.87.

## Interpretation (Phase 3, S4)
- **Fresh data (data):**
  - Fresh data passes the rule on top of C4: 975 → 864 mean errors, 1.47x → 1.30x MWPM, at slightly lower GPU time per epoch.
  - The fixed 300,000-shot set was limiting the C4 model: its train/val gap closes with fresh shots, and the best epoch moves to the end of the schedule.
  - Run-to-run spread is larger than C4's (range 98 vs 18). The pass is narrow on the range term.
- **Auxiliary targets (data):**
  - Neither auxiliary target helps. Both make the C4 model clearly worse (by 128 and 144 errors), and the ZX-propagated target is no better than the raw one.
  - This is the first post-fix ZX test at d=5, and the core SpiderTrace hypothesis is not supported in this setting.
  - The heads are alive (precision 6 to 9x the base rate), so this is not a collapsed-head artefact.
- **Hypotheses, not tested:**
  - λ = 0.1 may be too large for this backbone. No λ grid was run, so "aux hurts" holds for λ = 0.1 only.
  - The aux head reads 25 per-qubit targets from one pooled graph vector. It may compete with the flip head for the same 288 dimensions. A node-level aux head (targets read from node states) would not.
  - Fresh data plus a longer schedule (best epochs at the end) may improve further.
- **Ensembles (data):** 3-run ensembles land at the best single run's level (819 to 826 vs 823), not below it. They remove the risk of a bad run rather than improving on the best.
- **Next:**
  - Make fresh data part of the baseline recipe.
  - Residual-only and LayerNorm-only with fresh data (3 runs each).
  - Optionally, a longer schedule (for example 100 epochs) with fresh data.
  - Drop the auxiliary target unless a λ grid or a node-level head is tried.

## Phase 3, S5 (35k training graphs: C0, C4, C4 + aux ZX)
- **Date:** 2026-11-24
- **Where:** Kaggle, GPU, torch 2.11.0+cu130, torch_geometric 2.8.1, stim 1.16.0, PyMatching 2.4.0. One session; runs started 05:54 (C4), 06:32 (C4 + ZX) and 07:11 (C0) UTC.
- **Commit:** ba42925 (clean) for all three.
- **Purpose:**
  - Does the C4 backbone (residual + LayerNorm) help at a small fixed training set too, and by how much compared with 300k?
  - Does the auxiliary ZX target help when data is scarce, where extra supervision would matter most?
- **Commands** (common part: `python qec_run.py --d 5 --p 0.003 --shots 50000 --split-seed 12345 --train-shots 35000 --seeds 0 1 2 --epochs 60 --patience 60 --lr 1e-3 --lr-schedule cosine --lr-min 1e-5 --grad-clip 1.0 --eval-set data/eval_d5_p0.003_seed12347_200k.npz --expect-mwpm-errors 36 --device cuda`):
  - C0: `--arms A --outdir /kaggle/working/results/base_v2_35k`
  - C4: `--arms A --residual --norm layer --outdir /kaggle/working/results/resln_v2_35k`
  - C4 + ZX: `--arms ZX --lambdas 0.1 --aux-data-qubits-only --aux-class-weight inv-freq --residual --norm layer --outdir /kaggle/working/results/aux_zx_35k`
- **Setup:**
  - Recipe v2, as S3 and S4. Training set: the 35,000 base shots of the split (seed 12345), no extra shots. Val 7,500, test 7,500.
  - Parameters: C0 78,053, C4 78,821, C4 + ZX 107,721 (aux head used in training only).
  - ZX class weights I/X/Y/Z 0.27 / 7.61 / 17.30 / 11.65, from the 35k training split (S4 at 300k: 0.27 / 7.59 / 17.05 / 11.59).
- **Test sets:** the same primary set (`--expect-mwpm-errors 36` passed; correlated matching 21). All three runs loaded the stored eval set with all four checks passing, sha256 4b9504da...2772; MWPM 664, correlated matching 475.
- **Compared against:** S3 C0 and S3 C4 at 300k (results/run_20261122T202714Z_d5_p0.003.json, results/run_20261123T003558Z_d5_p0.003.json). Same code path, recipe and test shots; only the training-set size differs.
- **Files** (each with a `_weights/` folder of 3 files and a `_bigtest_probs.npz`):
  - C0: results/run_20261124T074644Z_d5_p0.003.json
  - C4: results/run_20261124T063211Z_d5_p0.003.json
  - C4 + ZX: results/run_20261124T071047Z_d5_p0.003.json

**Big test set (200,000 shots, MWPM 664 errors, correlated matching 475)**

| Run | Train graphs | Seed 0 | Seed 1 | Seed 2 | Mean | Range | Ratio to MWPM | Ratio to corr. matching |
|---|---|---|---|---|---|---|---|---|
| C0 | 35k | 2,491 | 2,502 | 2,371 | 2,454.7 | 131 | 3.70 | 5.17 |
| C4 | 35k | 2,191 | 2,245 | 2,150 | 2,195.3 | 95 | 3.31 | 4.62 |
| C4 + aux ZX | 35k | 2,160 | 2,190 | 2,431 | 2,260.3 | 271 | 3.40 | 4.76 |
| C0 (S3) | 300k | 1,272 | 1,249 | 1,184 | 1,235.0 | 88 | 1.86 | 2.60 |
| C4 (S3) | 300k | 968 | 972 | 986 | 975.3 | 18 | 1.47 | 2.05 |

- Paired against MWPM, every 35k run is significantly worse (fixed / newly wrong: C0 192 / 2,019, 182 / 2,020, 188 / 1,895; C4 177 / 1,704, 178 / 1,759, 172 / 1,658; ZX 165 / 1,661, 176 / 1,702, 174 / 1,941; every p below 1e-300).
- Zero-fired shots: 15,307; 0 true and 0 predicted flips for every run, as before.

**Decision rule**
- **C4 beats C0 at 35k.** The worst C4 run (2,245) is below the best C0 run (2,371). The mean gap of 259.3 exceeds both ranges (131 and 95). It passes.
- **Same gap as at 300k:** 259.3 errors at 35k, 259.7 at 300k (S3). As a share of C0's errors: 10.6% at 35k, 21.0% at 300k.
- **Same data effect for both backbones:** going from 35k to 300k removes 1,219.7 errors (C0) and 1,220.0 (C4).
- **C4 + ZX vs C4 at 35k: no decision either way.** ZX's mean is 65.0 errors higher. Two ZX runs (2,160, 2,190) are level with C4's best runs, and the third (2,431) is the worst of the six. The gap is below both ranges (95 and 271). At 300k (S4), C4 beat ZX by 127.7.
- **Same-seed paired McNemar** (reported, not decisive; fixed = first wrong & second right):

| Second vs first | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| C4 vs C0 (35k) | 787 / 487, p = 3.9e-17 | 799 / 542, p = 2.4e-12 | 786 / 565, p = 2.0e-9 |
| ZX vs C4 (35k) | 520 / 489, p = 0.34 | 544 / 489, p = 0.093 | 461 / 742, p = 5.1e-16 |

**Primary test set (7,500 shots, MWPM 36 errors, correlated matching 21)**

| Run | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| C0, 35k | 98 | 96 | 87 |
| C4, 35k | 75 | 84 | 77 |
| C4 + aux ZX, 35k | 75 | 80 | 85 |

- All worse than MWPM: C0 p = 4.7e-12, 2.7e-11, 1.7e-9; C4 p = 1.2e-6, 4.5e-9, 2.8e-7; ZX p = 1.5e-7, 1.0e-7, 4.8e-9.
- C0 at 35k is at the level of Run 6 (same data, old recipe, 100 epochs, 1 run: 93 errors).

**Aux head (ZX, test set, λ 0.1; not collapsed)**
- Non-identity recall 0.826 / 0.826 / 0.822. Precision 27.0% / 28.0% / 27.0%, against a true non-identity density of 0.069 (about 4x the base rate).
- It predicts non-identity on 0.210 / 0.202 / 0.209 of slots, about 3x the true density.
- At 300k (S4): recall 0.888 to 0.892, precision 38.5% to 40.4%, predicted 0.152 to 0.158. The head is weaker with less data, but it still learns.

**Training**

| Run | Seed | best_epoch (of 60) | Epoch-1 val loss | Best val loss (train loss at that epoch) | Val spikes (>1.3x) | s / epoch |
|---|---|---|---|---|---|---|
| C0, 35k | 0 / 1 / 2 | 56 / 55 / 50 | 0.455 / 0.440 / 0.456 | 0.0408 (0.0344) / 0.0355 (0.0331) / 0.0340 (0.0335) | none / 19 / 18 | 11.0 / 11.0 / 11.0 |
| C4, 35k | 0 / 1 / 2 | 50 / 59 / 59 | 0.680 / 0.504 / 0.726 | 0.0333 (0.0308) / 0.0348 (0.0301) / 0.0332 (0.0304) | 16 / 2, 5, 8, 28 / 4, 7, 12 | 11.7 / 11.7 / 11.8 |
| C4 + aux ZX, 35k | 0 / 1 / 2 | 49 / 42 / 59 | 1.475 / 1.446 / 1.035 | 0.0807 (0.0773) / 0.0822 (0.0737) / 0.0856 (0.0765) | none / none / 2 | 11.8 / 11.8 / 11.8 |

- The ZX losses include λ times the aux cross-entropy, so they are not comparable with the A-arm losses.
- All runs ran the full 60 epochs. C4's val loss spikes early (epochs 2 to 16, plus 28 in seed 1), but each run recovers.
- **GNN inference on the big test:** 29.2 to 29.3 s (C0), 30.4 to 30.8 s (C4), 30.5 to 30.9 s (ZX).

## Interpretation (Phase 3, S5)
- **Backbone (data):**
  - C4 beats C0 at 35k as well as at 300k, by the same absolute amount (about 260 errors).
  - Over this range the backbone effect and the data effect add up: each backbone drops by about 1,220 errors from 35k to 300k.
  - In relative terms the backbone matters more with more data (10.6% → 21.0% of C0's errors), because the total shrinks while the gap stays the same.
- **"Less data needed" (C2b): not shown yet.** Two sizes per backbone cannot say how many graphs C0 needs to match C4.
  - Under an assumption, not a measurement: if C0's errors fall linearly in log(training graphs) between its 35k and 300k means, C0 would need about 55,000 graphs to match C4 at 35k, and about 474,000 (an extrapolation) to match C4 at 300k. Both are about 1.6x the data. Values from `gml_final/numbers.md` key `scaling.v2_loglinear_equiv_graphs`.
  - The 100k runs test the log-linear assumption directly.
- **Auxiliary ZX target at low data (data):**
  - No gain at 35k either. The hypothesis that extra supervision pays off when data is scarce is not supported.
  - Unlike at 300k, it is also not clearly harmful: one bad run out of three decides the mean, and the same-seed McNemar goes both ways (seeds 0 and 1 not significant, seed 2 strongly against ZX).
  - The head learns less at 35k (precision 27% vs about 39%), so the extra target is noisier exactly where it was meant to help.
- **Hypotheses, not tested:**
  - λ = 0.1 is still the only weight tried.
  - The early val spikes in C4 at 35k (none at 300k in seeds 1 and 2) may come from the small training set and the full lr of 1e-3 at the start of the cosine schedule.
- **Next:**
  - C0 and C4 at 100k (3 runs each): a third point on the scaling curve, and a direct check of the log-linear assumption.
  - Residual-only and LayerNorm-only at 300k: which half of C4 matters.

## 2026-11-24: smoothing inside the trained models (analysis, no training)
- **Question:** C4 was motivated by the Phase 1 oversmoothing diagnosis, which was measured at initialisation (d=3) and on one trained d=3 default model. Do the trained d=5 models differ the way that motivation predicts?
- **Where:** local Windows CPU, torch 2.11.0. Code at 083700f plus the new, uncommitted `analysis/p4_trained_smoothing.py`.
- **Command:** `python analysis/p4_trained_smoothing.py` (defaults: 4,000 shots, probe on the first 1,000). Wall time 306 s.
- **Models:** the saved weights of S5 C0 and S5 C4 (35k) and of S4 C4 + fresh data (the final model), seeds 0 1 2. Plus C0 and C4 before training, built as `run_arm` builds them (`_seed_all(seed)`, then `build_model`). The CUDA runs built their models on CPU from the same torch seed, so these should be their starting weights; this is not checked.
- **Shots:** the first 4,000 shots of the stored eval set (5.02 fired detectors per shot on average, 318 with none). The file loaded with the strict DEM fingerprint failing and the bit and MWPM checks passing (Windows; the warning path from 4f759ad).
- **Check:** every reloaded trained model gives exactly its run's wrong-shot list on these shots (C0 46 / 45 / 46, C4 39 / 48 / 37, final 21 / 17 / 15).
- **Metrics:** as in Phase 1 (`analysis/p3_oversmoothing.py`), on x_k, the node states layer k passes on (k = 4 is what the readout pools), eval mode. For C4, x_k = x_{k-1} + relu(LN(conv_k(x_{k-1}))). Each metric is averaged over shots, then over seeds.
- **Output:** analysis/phase3_results/p4_trained_smoothing.json. The ledger keys `smooth.*` in `gml_final/numbers.md` are computed from it.

**Layer 4, mean of 3 seeds**

| Metric | C0, untrained | C0, trained 35k | C4, untrained | C4, trained 35k | C4 + fresh (final) |
|---|---|---|---|---|---|
| Cosine, all node pairs | 0.994 | 0.517 | 0.981 | 0.493 | 0.246 |
| Cosine, fired vs unfired detectors | 0.994 | 0.491 | 0.974 | -0.033 | -0.104 |
| Pairs with cosine > 0.99 | 0.735 | 0.153 | 0.281 | 0.000 | 0.000 |
| Dirichlet energy (scale-free) | 3.14 | 10.9 | 0.026 | 2.02 | 6.53 |
| Effective rank (of 96) | 1.04 | 1.42 | 5.11 | 4.48 | 3.75 |
| Mean norm: fired / unfired / boundary | 58.3 / 55.0 / 372.7 | 0.82 / 0.67 / 16.8 | 19.8 / 19.8 / 19.8 | 6.26 / 2.29 / 7.92 | 5.43 / 1.34 / 12.7 |
| Linear probe for the fired bit (balanced accuracy) | 0.758 | 0.665 | 1.000 | 1.000 | 1.000 |

**Linear probe for the fired bit, layers 0 to 4**
- C0 untrained: 1.000, 0.999, 0.918, 0.831, 0.758
- C0 trained 35k: 1.000, 0.997, 0.913, 0.801, 0.665 (seeds at layer 4: 0.676, 0.661, 0.658)
- C4 untrained, C4 trained 35k, final: 1.000 at every layer for every seed

**Cosine fired vs unfired, layers 1 to 4:** C0 trained 0.106, 0.297, 0.596, 0.491. C4 trained 0.147, 0.012, -0.023, -0.033.

## Interpretation (smoothing inside the trained models)
- **At initialisation, C0 oversmooths at d=5 as at d=3:** layer-4 cosine 0.994, 74% of pairs above 0.99, rank 1.04. Norms grow about 14x (fired) to 16x (unfired) for detectors and about 107x at the boundary node from layer 0 to layer 4.
- **After training, C0 is no longer oversmoothed in the classic sense:** cosine 0.52, and the Dirichlet energy is high (10.9), not near 0. "The trained default oversmooths" is not supported.
- **What persists in trained C0:**
  - Its states stay near rank 1 (1.42).
  - The boundary node's norm is about 25x an unfired detector's.
  - The fired bit becomes hard to read linearly with depth (probe 1.00 → 0.66), and fired and unfired detectors become more alike (cosine 0.11 at layer 1, 0.49 at layer 4).
- **C4 keeps the local signal:** fired and unfired states are about orthogonal from layer 2 on, the probe stays at 1.00, the rank stays about 4.5, and the boundary norm is about 3.5x an unfired detector's.
- **C4 at initialisation** also has high cosine (0.98) but no norm blow-up and rank 5. What it prevents at the start is the blow-up and the rank collapse, not high cosine.
- **Caveats:**
  - The probe at 1.00 for C4 is partly by construction: the skip path carries the encoder output, which contains the fired bit, to every layer.
  - A linear probe only measures linear readability; C0 may still carry the fired bit non-linearly.
  - The link to errors is correlational: a large representational difference comes with a 10.6% error difference at 35k (S5).
- **Wording for C2:** "Residual + LayerNorm keeps each detector's own signal through depth, which the default backbone loses; the classic oversmoothing seen at initialisation is mostly undone by training in both."
