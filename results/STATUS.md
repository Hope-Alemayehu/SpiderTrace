# SpiderTrace: status, open work and ideas (as of 2026-11-23)

A one-page overview. The detailed record, with every number and its source JSON, is `results/LOG.md`. The Phase 1 diagnosis is `analysis/REPORT_phase1.md`. Numbers here are copied from those two files or from `analysis/phase3_results/s3_details.json`.

## 1. The task in one paragraph

A graph neural network decodes a rotated surface code (memory-Z, rounds = d) under uniform circuit-level depolarizing noise. Each shot is a set of fired detectors on the detector error model (DEM) graph, plus one shared boundary node. The model predicts whether a logical flip happened. MWPM (PyMatching) is the baseline. Three arms share one backbone:
- **GNN-A** predicts the flip only.
- **GNN-Raw** also predicts the raw Pauli errors per data qubit (auxiliary head, weight λ).
- **GNN-ZX** also predicts the ZX-propagated Pauli frame per data qubit (the SpiderTrace idea).

Main setting: d=5, p=0.003. Headline metric: errors on the **big test** (200,000 fresh shots, seed 12347), where MWPM makes **664** errors. The 7,500-shot primary test set (MWPM 36 errors) is too small to separate decoders and is kept only for continuity.

## 2. What has been done

### Phase 0: pipeline and first runs (Runs 1 to 5, Sept to Oct 2026)
- Built `qec_run.py`: one command trains the arms, sweeps λ, selects by validation, scores on a fixed test split, runs McNemar against MWPM and between arms, and writes one JSON with full provenance (commit, flags, library versions).
- Runs 1 to 4 (d=3 and d=5) compared A, Raw and ZX. Run 4 fixed the aux head collapsing to "identity everywhere": `--aux-data-qubits-only` and `--aux-class-weight inv-freq`.
- **ZX-target bug found and fixed (2026-10-06, commit d92b05a).** Faults were injected at the start of their tick layer instead of after their own noise instruction, so 34% of d=5 shots had a wrong ZX target. After the fix every target is consistent. **All GNN-ZX results from Runs 1 to 4 are invalid.**
- Run 5 (d=3, 1 seed) is the only post-fix ZX result: ZX 0.0589, Raw 0.0597, A 0.0592, MWPM 0.0545. ZX vs Raw p = 0.63. Within-arm noise was larger than the differences between arms. **No post-fix ZX run exists at d=5 yet.**

### Data scaling (Runs 6 to 11, GNN-A only)

| Run | Train shots | Primary test errors (MWPM 36) | Big test errors (MWPM 664) |
|---|---|---|---|
| Run 6 | 35,000 | 93 | n/a |
| Run 8 | 300,000 | 65 | n/a |
| Run 10 | 600,000 | 53 | n/a |
| Run 11 | 1,000,000 | 37 | 1,021 (1.54x) |

- More data helps clearly. More epochs on the same 35k did not (Run 7).
- Run 9 (LayerNorm + mean-max readout, 35k, constant lr) learned more slowly than the default backbone, so normalisation was shelved at the time.
- Run 11's apparent parity with MWPM on the primary set (37 vs 36) comes from MWPM doing unusually badly on that set. The big test shows 1.54x.

### Phase 1: diagnosis of the trained model (`analysis/REPORT_phase1.md`)
- The DEM graph has two detector sublattices (X-type and Z-type) linked only through the boundary node. All observable-flipping edges sit in one of them.
- The boundary node is a hub. About 60% of a d=5 detector's 4-hop signal passes through it. In the trained d=3 model, removing the boundary edges drops flip predictions from 5.9% of shots to 0.1%: the decision is read through the hub.
- At initialisation the 4-layer network oversmooths and blows up: cosine 0.997 between all node pairs at layer 4, effective rank 1.03, norms grow 5x per detector and 16x at the hub. This matches the +147 initial logit at d=5.
- After training the embeddings stay very low-rank (1.3 to 1.6 of 96 channels).
- This produced the Phase 3 candidate list: C1 mean readout, C2 fired-only readout, C3 per-round boundary nodes, C4 residual + normalisation, C5 MLP head. The flags for C2 to C6 are in commit 5110582.

### Phase 3: architecture and recipe ablations (Nov 2026)

| Session | Config | Recipe | Big-test errors (per run) | Ratio to MWPM | Outcome |
|---|---|---|---|---|---|
| S1 | C0 (default) | constant lr, early stopping | 1,348 / 1,429 | about 2.1 | baseline |
| S1 | C2 (fired-only readout) | same | 1,477 / 1,379 | about 2.2 | no benefit; dropped |
| S2 | C0 rerun (identical command) | same | 1,647 / 1,658 | about 2.5 | revealed GPU nondeterminism |
| S2 | C0_recipe | cosine + clip 1.0, early stopping on | 1,479 / 1,206 | about 2.0 | spikes gone, spread not |
| S3 | C0 | recipe v2: cosine over 60 epochs, clip 1.0, no early stopping | 1,272 / 1,249 / 1,184 | 1.86 | spread shrank |
| S3 | **C4 (residual + LayerNorm)** | recipe v2 | **968 / 972 / 986** | **1.47** | **beats C0 under the decision rule** |

- **Decision rule (from S3 on):** config B beats config A only if every B run beats every A run on the big test (3 runs each), and the mean gap is larger than the run-to-run range within either config. Per-seed McNemar is reported but is not decisive.
- **C4 vs C0:** worst C4 run 986 vs best C0 run 1,184. Mean gap 259.7 errors, against ranges of 88 (C0) and 18 (C4).
- **C4 at 300k matches Run 11 at 1M** (968 to 986 vs 1,021; paired p = 0.055 to 0.22). Run 11 is a single run under the old recipe.

### Infrastructure
- Analysis scripts that rebuild every LOG number from the JSONs: `analysis/phase3_s1_details.py`, `phase3_s2_details.py`, `phase3_s3_details.py` (the last one also applies the decision rule).
- `tests/conftest.py` forces the matplotlib Agg backend, so the suite no longer hangs on a plot window. `tests/test_custom.py` (an interactive script) is skipped under pytest. Suite: 101 passed, 1 skipped.

## 3. What we know, and how sure we are

**Established (data):**
- More training data helps GNN-A at d=5.
- Residual + LayerNorm (C4) beats the default backbone under recipe v2, by about 260 big-test errors.
- Fired-only readout (C2) does not help.
- Gradient clipping removes the huge first-epoch gradients (norms up to 9,355) and gives a much better first epoch. After epoch 1 it clips only 4 to 7% of steps.
- GPU training on Kaggle is not reproducible. Same commit, command and seed changed big-test errors by 229 to 299 under the old recipe. Runs diverge within the first epoch.
- Under recipe v2 the spread between runs is much smaller (range 88 for C0, 18 for C4, against 273 for C0_recipe).

**Not yet known:**
- Which half of C4 matters (residual, LayerNorm, or both together).
- Whether the full cosine schedule or the removal of early stopping shrank the spread (recipe v2 changed both).
- How large same-seed nondeterminism is under recipe v2 (never measured).
- Whether ZX supervision helps at d=5 with correct targets (never run).
- Whether C4 helps without recipe v2 (never run).

**Hypotheses (consistent with the data, untested):**
- Residual connections stop the 4 layers from washing out each node's local information (Phase 1 found near-total oversmoothing at initialisation).
- LayerNorm keeps feature scales stable under sum aggregation and the add pool. Note that C4's gradients are larger than C0's, so "smoother optimisation" is not the explanation.
- Early stopping at a high lr was the main source of the run-to-run spread.

## 4. In progress

**aux_C4_v2 (Kaggle, tonight):** Raw and ZX arms, seeds 0 1 2, λ = 0.1, `--aux-data-qubits-only --aux-class-weight inv-freq`, C4 backbone, recipe v2, 300k shots, big test. Estimated 8.3 h, from C4's 4,885 to 4,896 s per seed and arm. The aux-head overhead is not measured.
- **This is the first post-fix ZX result at d=5** and the first test of the core SpiderTrace idea on the current best backbone.
- **Compare against:** the C4 A-arm runs (`results/run_20261123T003558Z_d5_p0.003.json`), offline, with the decision rule. The JSON itself only contains ZX vs Raw and ZX vs MWPM.

## 5. What is left (planned)

In rough priority order:
1. **Analyse aux_C4_v2.** Decision-rule comparisons: ZX vs A, Raw vs A, ZX vs Raw. Also check the aux heads did not collapse (non-identity recall).
2. **Split C4:** residual-only and LayerNorm-only, recipe v2, 3 runs each.
3. **Measure recipe-v2 noise:** one same-seed rerun of C4 (or C0) under recipe v2.
4. **C4 at 1M shots** under recipe v2: does the data gain stack on the architecture gain?
5. **Remaining Phase 3 candidates on the C4 backbone:**
   - C5, `--head mlp`. Phase 1 found the pooled embedding is very low-rank and the decision is not linear in pooled sums.
   - C3, `--boundary per-round`. Cuts the hub's degree from 72 to about 12. Mainly matters for larger graphs and long memory experiments.
   - Depth 8 with residual + norm (`--depth 8 --residual --norm layer`). Phase 1 suggested the earlier 8-layer failure came from the init pathology that C4 addresses.
   - C1 `--readout mean` and C6 `--sublattice-types`: low priority at fixed d.

## 6. Worth exploring

**Make results trustworthy**
- **Deterministic GPU training:** `torch.use_deterministic_algorithms(True)` plus deterministic scatter ops in PyG. If it works, one run per seed becomes meaningful again.
- **More runs per config** (5 instead of 3) for small effects. With 3 vs 3 the rank test cannot go below one-sided p = 0.05.
- **Save trained weights.** No trained checkpoint exists anywhere: `qec_run.py` deletes its per-epoch checkpoints when a run finishes. Keeping the best model of each run would allow re-scoring, mechanism probes and latency tests without retraining.
- **Always keep a log.** The C2 rerun JSON was lost on Kaggle. `tee` the log and `cat` the JSON at the end of every cell.

**Understand why C4 works**
- Run `analysis/p3_oversmoothing.py` on a trained C0 and a trained C4 (once weights are saved): cosine similarity, effective rank and hub dominance per layer.
- Check C4's initial logit scale against C0's +147.

**Tune the recipe for C4**
- lr was never retuned for the normalised backbone (Run 9 noted the same).
- Try more epochs (for example 100 with full cosine), since best epochs sit late in the schedule (42 to 56 of 60).
- Wider hidden size, now that the residual path makes a deeper or wider network trainable.

**Extend the scope**
- d=7, and transfer of a d=5 model to longer memory experiments (where the hub degree explodes, so C3 becomes relevant).
- A sweep over p, to see where the GNN gap to MWPM grows or shrinks.
- Decoding latency against MWPM, for a practical comparison.

## 7. Things to keep in mind
- **Headline metric:** the big test (200,000 shots, MWPM 664 errors). Ignore apparent parity on the 7,500-shot set.
- **Verdicts:** use the decision rule (3 runs, every run better, gap larger than the within-config range). One pair of runs proves nothing on this hardware.
- **ZX results:** only from commit d92b05a on. Anything earlier used broken targets.
- **Two changes at once** (Run 9, C4, recipe v2) give a result but not a cause. Plan the split run straight after.
- **Gap to MWPM:** the best decoder is still 1.47x worse than MWPM on the big test.
