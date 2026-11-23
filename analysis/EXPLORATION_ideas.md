# Exploration: directions to extend the current path (2026-11-23)

Read-only exploration. No code was changed and no training was run. Two goals:
1. close the gap to MWPM at d=5, p=0.003;
2. a strong, honest arXiv preprint.

**Starting point:** C4 (residual + LayerNorm) with recipe v2 makes 968 / 972 / 986 big-test errors over 3 runs. The mean of 975 is 1.47x MWPM's 664 (`results/LOG.md`, Phase 3 S3).

**Sources:** `results/STATUS.md`, `results/LOG.md`, `analysis/REPORT_phase1.md`, `qec_run.py`, `gnn_models.py`, `qec_zx_dataset.py`, `graph_variants.py`, and the papers linked in section 4. Section 1 lists the new measurements made for this note. Each has the command or file it came from.

Labels used below:
- **Evidence:** from the repo's JSONs, the code, or a measurement in section 1.
- **Literature:** from a cited paper.
- **Speculation:** a guess, not yet checked.

## 1. New measurements made for this note (no training)

**A. Voting and overlap of the 3 C4 runs on the big test.**
- **Source:** the `test_wrong_idx` lists in `results/run_20261123T003558Z_d5_p0.003.json` and `results/run_20261122T202714Z_d5_p0.003.json`.
- **Majority vote of 3 runs:**
  - C4: 921 errors, against a mean of 975 (5.6% fewer).
  - C0 (recipe v2): 1,132, against a mean of 1,235 (8.3% fewer).
- **Shots wrong in all 3 C4 runs:** 666. MWPM is also wrong on 407 of them and right on 259.
- **Shots MWPM gets wrong:** on 106 of them, no C4 run is wrong.
- **"Oracle" (right if the C4 vote or MWPM is right):** 483 errors, 0.73x MWPM. This is an upper bound on what choosing between the two can give. It is not a decoder.
- **Simple rule (use the C4 vote when all 3 runs agree, otherwise MWPM's answer):** 817 errors, 1.23x MWPM. It needs no training, but it needs MWPM at inference.

**B. Correlated matching vs plain MWPM** (PyMatching 2.4.0, local CPU).
- **Setup:**
  - 200,000 fresh shots from stim 1.16.0 with sampler seed 12347, the same code path as the big test.
  - Plain matching: `Matching.from_detector_error_model(dem)`.
  - Correlated matching: the same call with `enable_correlations=True`, and `decode_batch(..., enable_correlations=True)`.
- **Result on these shots:**

| Decoder | Errors | Decode time (200k shots) |
|---|---|---|
| Plain MWPM | 690 | 0.2 s |
| Correlated matching | 515 | 0.6 s |

- **Paired comparison:** correlated matching fixes 338 of MWPM's errors and adds 163 new ones (p = 4e-15). That is **25% fewer errors than MWPM**.
- **Caveat:** these are not the Kaggle big-test shots. Stim is version 1.16.0 on both machines, but Kaggle runs Linux x86_64 and this machine runs Windows. The same seed gave different shots (MWPM 690 here vs 664 on Kaggle). **The big test is not reproducible across machines.** See direction 2.
- **Estimate (speculation, but simple arithmetic):** scaled to the Kaggle big test, correlated matching would make about 664 x 0.75 = about 495 errors. On that estimate C4 is about 2.0x correlated matching.

**C. How much hyperedge structure the graph discards.**
- **Source:** the non-decomposed DEM from `build_circuit(5, 0.003)`.
- **Error mechanisms by number of detectors (1,677 mechanisms in total):**

| Detectors per mechanism | Mechanisms | Total probability | Observable-flipping probability |
|---|---|---|---|
| 1 | 72 | 0.580 | 0.099 |
| 2 | 504 | 1.450 | 0.035 |
| 3 | 480 | 0.285 | 0.050 |
| 4 | 621 | 0.256 | 0 |

- **Hyperedges are common:** 66% of the mechanisms touch 3 or 4 detectors. 3-detector mechanisms carry 27% of the observable-flipping probability.
- **What the graph keeps:** `build_dem_graph` uses the *decomposed* DEM and merges components by endpoint set. The co-occurrence inside each hyperedge is lost.
- **Phase 1 finding:** the X-type and Z-type detector sublattices are joined only through the boundary node.

**D. Throughput.**
- **Sampling is cheap:**
  - Stim samples 1,000,000 shots in 0.68 s.
  - Building PyG `Data` objects runs at about 6,900 shots/s (local CPU).
  - Collating 256 graphs takes 5.2 ms.
- **A Kaggle T4 step is about 70 ms:** C4 takes 4,890 s for 60 epochs of 1,172 steps.
- **Fired detectors (5,000 local shots):** mean 5.13 per shot out of 121 nodes, 7.0% of shots have none, maximum 22.

**E. The C4 runs now fit training data better than validation data.**
- **Best val loss vs train loss at that epoch:**

| Run | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| C4 | 0.0156 vs 0.0130 | 0.0143 vs 0.0126 | 0.0151 vs 0.0110 |
| C0, recipe v2 | 0.0187 vs 0.0179 | 0.0172 vs 0.0179 | 0.0182 vs 0.0162 |

- **The gap is probably understated:** train loss is measured with dropout on, which biases it upward. The real train/val gap is likely larger than shown.
- **Change since Runs 6 and 7:** there train and val loss were nearly equal at 35k shots.

## 2. Ranked directions

GPU-hour figures are for a Kaggle T4 and one comparison of 3 runs. The reference is a C4 3-run config: 3 x about 4,890 s, so about 4.1 h of training plus about 0.1 h of setup. Kaggle's 30 h per week allows about 6 such configs.

### 1. Fresh training data every epoch (streamed from stim)
- **What:** draw new training shots every epoch instead of reusing the same 300k for 60 epochs. Same GPU time, but 60 x 300k = 18M distinct shots instead of 300k.
- **Why it might help here:**
  - **Evidence, data:** data was the strongest lever so far. Primary-set errors went 93, 65, 53, 37 from 35k to 1M shots (Runs 6, 8, 10, 11). C4 at 300k now shows a train/val gap (1.E), so it is starting to overfit its fixed set.
  - **Evidence, cost:** sampling costs almost nothing next to the GPU step (1.D).
  - **Literature:** Lange et al. use a fresh dataset of 10^7 samples per epoch and about 10^9 samples in total, and state "No data is reused, which implies that the network cannot overfit" ([arXiv:2307.01241](https://arxiv.org/abs/2307.01241)). A 2026 preprint reports convergence at about 3x10^8 on-the-fly examples ([arXiv:2604.08358](https://arxiv.org/pdf/2604.08358); read from search excerpts only).
- **Code changes:**
  - A streaming train dataset with a fixed seed schedule (for example extra_train_seed + epoch, so it never touches val, test or big-test seeds). Val and test stay fixed.
  - **Option, faster:** the graph topology is fixed, so a batch can be built straight from the stim detector matrix (B x 120 bits) with tensor ops. This skips per-shot `Data` objects.
  - **Aux targets:** the ZX and raw targets are XORs of the fired mechanisms' Pauli strings, so they can be vectorised as a GF(2) matrix product: errors (B x 1,677) times a 1,677 x 2N symplectic matrix.
  - **Test:** the new batches must equal `Batch.from_data_list` on the same shots.
- **GPU hours:** about 4.5 h for 3 runs at the current 60 epochs. That is the same as C4, if generation overlaps with training in a loader worker. An optional 120-epoch version is about 9 h.
- **Success:** C4 + fresh data beats C4 (mean 975) under the decision rule, and the train/val gap closes.
- **Failure:** no gain under the rule. That would point to capacity (direction 7) rather than data as the limit.

### 2. Honest baselines and a reproducible evaluation set (no GPU)
- **What:**
  - Add correlated matching to every run, and ideally belief-matching too.
  - Store the big-test syndromes and labels on disk, so every decoder is scored on the same shots on any machine.
  - Save each run's per-shot logits on the big test and its best weights.
- **Why it might help here:**
  - **Evidence, the bar moves:** correlated matching makes 25% fewer errors than MWPM on 200k shots (1.B). A preprint that reports only "x MWPM" would compare against the weaker baseline.
  - **Evidence, reproducibility:** the big test cannot be reproduced on a different machine even with the same stim version (1.B).
  - **Evidence, no saved models:** no trained weights exist anywhere. `qec_run.py` deletes its checkpoints when a run finishes (`results/STATUS.md`). Every analysis below that needs logits needs this.
  - **Literature, belief-matching:** it raises the circuit-noise threshold from about 0.82% to about 0.94% ([Higgott et al., PRX 13, 031007; arXiv:2203.04948](https://arxiv.org/abs/2203.04948)).
  - **Literature, both baselines:** Lange et al. compare against both MWPM and belief-matching ([arXiv:2307.01241](https://arxiv.org/abs/2307.01241)).
  - **Literature, AlphaQubit:** it is reported against correlated matching and tensor-network decoding ([Bausch et al., Nature 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11602728/); [blog](https://blog.google/technology/google-deepmind/alphaqubit-quantum-error-correction/)).
- **Code changes:**
  - Write the big test once to an `.npz` file: 200k x 120 bits packed, about 3 MB.
  - An `--eval-set path` flag.
  - A `correlated_matching` entry next to `mwpm` in the JSON.
  - Save big-test logits (200k floats per run) and `best_state` next to the JSON.
  - Belief-matching needs an external package (for example [BeliefMatching](https://github.com/KuantumBS/BeliefMatching)) and a CPU run on BlueQubit.
- **GPU hours:** 0. CPU: seconds for correlated matching. Belief-matching is slower; I have not measured it.
- **Success:** every result table has MWPM, correlated matching (and belief-matching), with paired tests on identical stored shots, and logits on disk.
- **Failure:** not applicable; this is infrastructure. Its risk is the headline itself: C4's ratio to correlated matching (about 2.0x, estimated) is less flattering than 1.47x. That is the honest number.

### 3. Hyperedge (X-Z correlation) edges in the input graph
- **What:** give the GNN the co-occurrence information that decomposition throws away. Two options:
  - **(a)** add "correlation edges" between the detectors of each 3- or 4-detector mechanism, with features: mechanism log-odds, a hyperedge flag, and size;
  - **(b)** a factor-graph input: one node per error mechanism (1,677) connected to its detectors. A GNN on this graph is a learned form of belief propagation.
- **Why it might help here:**
  - **Evidence, what is missing:** hyperedges make up 66% of mechanisms and 27% of the observable-flipping probability (1.C). In the current graph, the X and Z sublattices meet only at the boundary hub (Phase 1). Every X-Z correlation must squeeze through one node that already dominates the max pool.
  - **Evidence, the information is worth a lot:** correlated matching, which uses exactly this information, makes 25% fewer errors than MWPM (1.B).
  - **Literature, the source of neural advantage:** Varbanov et al. find their network beats matching "due to better handling errors leading to multiple correlated syndrome defects, such as Y errors" ([PRR 7, 013029; arXiv:2307.03280](https://arxiv.org/abs/2307.03280)).
  - **Literature, hypergraph decoders:** hypergraph neural decoders are motivated by the same limit of pairwise GNNs ([HyperNQ, arXiv:2511.01741](https://arxiv.org/abs/2511.01741); search excerpts only).
- **Code changes:**
  - **(a)** a new `graph_variants` option (for example `--hyperedges pairs`), built from `circuit.detector_error_model(decompose_errors=False)`, one extra edge feature, and a test that the default graph is unchanged. About a day.
  - **(b)** a heterogeneous graph and a new backbone. Several days.
- **GPU hours:**
  - **(a)** about 5 to 7 h for 3 runs (speculation: edge count roughly doubles).
  - **(b)** possibly 5 to 15x more per epoch (speculation). It would need the RTX 5090 or a smaller schedule.
- **Success:** C4 + hyperedges beats C4 under the rule.
- **Mechanism check:** do the gains fall on shots where X-type and Z-type detectors fire next to each other?
- **Failure:** no gain under the rule, while correlated matching still gains 25%. That would mean the GNN cannot use the information in this form.

### 4. Matching-informed hybrid: give the GNN the matcher's answer
- **What:** add the MWPM (or correlated-matching) prediction as an input, and train the GNN to output the final label. Two forms:
  - a global bit concatenated to the pooled vector;
  - a per-edge "in the matching" feature, via PyMatching's edge output.
- **Why it might help here:**
  - **Evidence, complementary errors:** the GNN and MWPM fail on partly different shots. 106 of MWPM's 664 errors are right in every C4 run, and the oracle bound is 483 (1.A).
  - **Evidence, even a crude rule helps:** "use MWPM when the 3 runs disagree" already gets 817 with no training (1.A).
  - **Literature:** the opposite design, a neural pre-decoder followed by matching, is active work. One 2026 preprint reports that pre-decoders can make results *worse* than correlated matching alone, because leftover chains run parallel to the logical ([arXiv:2604.12841](https://arxiv.org/pdf/2604.12841); search excerpts only). See also [QAdapt, arXiv:2607.28422](https://arxiv.org/pdf/2607.28422) and [Gicev et al., arXiv:2110.05854](https://arxiv.org/abs/2110.05854). Feeding the matcher's output *into* the GNN avoids that failure mode (speculation).
- **Code changes:**
  - Compute the matching prediction per shot when building the data (200k shots take 0.2 to 0.6 s).
  - Carry it as a graph attribute.
  - A `--matching-input {none,mwpm,correlated}` flag in the model.
  - The per-edge version needs `decode_to_edges_array` and a mapping onto `edge_index`.
- **GPU hours:** about 4.5 h for 3 runs.
- **Success:** all 3 runs below MWPM's 664, with the gap larger than the range. For the preprint, below correlated matching (estimated about 495).
- **Failure:** the network copies the input bit and lands at MWPM's level.
- **Caveat for framing:** this is a hybrid decoder with matching in the inference path. It is a different claim from "a GNN decodes" and must be reported that way.

### 5. Fixed-topology batched backbone (faster and deterministic)
- **What:** every shot uses the same 121-node graph. Message passing can therefore be written as fixed matrix products: gather with a fixed incidence matrix, then aggregate with its transpose. This replaces PyG's per-batch `scatter_add`.
- **Why it might help here:**
  - **Evidence, the noise problem:** GPU nondeterminism moved same-seed results by 229 to 299 errors (S2). That forced the 3-run rule and a range-based decision.
  - **Speculation, determinism:** scatter-add with float atomics is the usual source of nondeterminism. Dense matmuls are typically deterministic for a fixed configuration.
  - **Speculation, speed:** the 70 ms step is large for 31k nodes (256 x 121) on a T4. Some of it is probably the per-graph `Data` handling, which goes away.
- **Code changes:**
  - A second backbone class implementing GINEConv with fixed index tensors.
  - An equivalence test against `GNNBackbone`: same weights, same outputs within float tolerance.
  - A run with `torch.use_deterministic_algorithms(True)` to check that two same-seed GPU runs match bit for bit.
  - This combines naturally with the streaming batches in direction 1.
- **GPU hours:** about 1 h to verify determinism (2 short same-seed runs). After that it saves time.
- **Success:** two same-seed runs on Kaggle give identical big-test errors, and steps are faster.
- **Payoff:** a single run per seed becomes a valid comparison again, and seed variance can be separated from hardware noise. That is a methods result worth a section in the preprint.
- **Failure:** still nondeterministic, or slower.

### 6. Ensembles (and save logits so they can be measured properly)
- **What:** average the 3 runs' probabilities, instead of reporting only per-run results.
- **Why it might help here:**
  - **Evidence:** a plain majority vote of the existing runs already gives 921 vs 975 for C4, and 1,132 vs 1,235 for C0 (1.A). Averaging probabilities usually beats voting (speculation; it needs saved logits, direction 2).
  - **Literature:** the AlphaQubit paper is widely reported to use model ensembles. I did not verify this in the paper text for this note.
- **Code changes:** none beyond saving logits (direction 2), plus an offline script.
- **GPU hours:** 0. The 3 runs already exist for every config.
- **Success:** the averaged ensemble beats its best member under paired tests. Report it in its own row, as "3-model ensemble, 3x inference cost".
- **Failure:** no gain over voting. It is cheap either way.

### 7. Capacity: wider and deeper C4
- **What:**
  - hidden size 96 to 192 or 256 (needs a `--hidden` flag; `HIDDEN` is a constant in `gnn_models.py`);
  - depth 4 to 6 or 8 with residual + LayerNorm (`--depth` exists).
- **Why it might help here:**
  - **Literature:** Lange et al.'s circuit-level network has 7 graph layers up to width 512 and 1.36M parameters, against 78,821 here ([arXiv:2307.01241](https://arxiv.org/abs/2307.01241)).
  - **Evidence:** Phase 1 suggested that the earlier 8-layer failure came from the init pathology that C4 fixes.
  - **Evidence against doing it yet:** with a fixed 300k set, C4 already overfits (1.E), so more capacity alone would probably widen the gap. Do this after direction 1.
- **Code changes:** a `--hidden` flag (small).
- **GPU hours:** 8 to 12 h for 3 runs at about 2x step time (`analysis/phase2_results/step_time.json`: depth 8 measured 2.0x C0 on CPU). A good use of the RTX 5090.
- **Success:** a larger C4 with fresh data beats C4 with fresh data under the rule.
- **Failure:** no gain, meaning capacity is not the limit at this data budget.

### 8. Failure analysis of the shared hard shots (no GPU)
- **What:** describe the 666 shots all 3 C4 runs get wrong, and in particular the 259 where MWPM is right (1.A).
- **What to measure:** number of fired detectors, boundary involvement, time extent, whether X-type and Z-type detectors co-fire (Y-like patterns), and the chain length MWPM chose.
- **Why it might help here:** it would show which of directions 3, 4, 7 and 9 to try first, instead of guessing.
  - **Evidence:** unanimous errors are systematic, not noise.
  - **Evidence:** 407 of them are also wrong for MWPM, which suggests a hard core that may be near-irreducible.
- **Code changes:** an analysis script. It needs the big-test syndromes: regenerate on Kaggle's Linux CPU, or wait for the stored set (direction 2).
- **GPU hours:** 0.
- **Success:** a clear pattern, for example "80% of the GNN-only failures involve both sublattices". That points straight to direction 3.
- **Failure:** no pattern. That still argues for data or capacity rather than structure.

### 9. Training noise distribution: higher or mixed p
- **What:** train on a mixture of p (for example 0.002 to 0.005), or at a higher p, and evaluate at 0.003.
- **Why it might help here:**
  - **Speculation:** at p = 0.003 most shots are easy (7% have no fired detector; the mean is 5.1). Higher p gives more decision-relevant examples per step.
  - **Literature:** Lange et al. train on an even mix of p in {1, ..., 5} x 10^-3 ([arXiv:2307.01241](https://arxiv.org/abs/2307.01241)). A 2026 preprint trains at p = 0.7% and evaluates at lower rates ([arXiv:2604.08358](https://arxiv.org/pdf/2604.08358); search excerpts only).
- **Code changes:**
  - Edge weights currently come from the p = 0.003 DEM.
  - Either keep the eval-p weights for all training data, or add the training p as an input.
  - Streaming generation from several circuits (combine with direction 1).
- **GPU hours:** about 4.5 h for 3 runs.
- **Success:** beats the single-p model at p = 0.003 under the rule. A bonus is one model for a p sweep in the preprint.
- **Failure:** worse at p = 0.003, which would mean the distribution mismatch costs more than the extra signal gains.

### 10. Syndrome-only graph with matching-distance edges (Lange-style)
- **What:** build each shot's graph from the fired detectors (plus the boundary) only.
  - Edges between fired detectors.
  - Edge features: the DEM shortest-path weight and the path's observable parity. This is the graph MWPM actually solves. All-pairs shortest paths on the 121-node DEM graph are computed once.
- **Why it might help here:**
  - **Evidence, throughput:** the mean is 5.1 fired detectors out of 121 nodes (1.D), so about 20x fewer nodes. That would make 10^8-sample training plausible on a T4 (speculation).
  - **Speculation, depth:** distances between fired detectors are given directly, so no 4-hop message passing has to discover them.
  - **Literature:** Lange et al. use fired-detector nodes with inverse-distance edges and beat MWPM and belief-matching ([arXiv:2307.01241](https://arxiv.org/abs/2307.01241)).
- **Code changes:** large. A new dataset builder, variable graph sizes, a different readout, and new tests. This is a new decoder family rather than an ablation.
- **GPU hours:** unknown until built. Probably less per sample than now (speculation).
- **Success:** at matched GPU hours, beats C4 under the rule.
- **Failure:** worse, or no faster.

## 3. Framing for the preprint (no GPU)

Possible contributions, in order of strength today:
1. **ZX-propagated auxiliary targets.** This is the project's original idea. Tonight's aux_C4_v2 run is the first test at d=5 with corrected targets (Runs 1 to 4 used broken targets). A clean negative result, "ZX supervision does not help a well-trained GNN at d=5", is still publishable if the protocol is solid.
2. **Benchmarking methodology for neural decoders.**
   - **Measured:** same-seed GPU nondeterminism (229 to 299 errors) was larger than architecture effects (S2). A 3-run cross-run rule with a range test fixes the comparison.
   - **If direction 5 works:** a deterministic fixed-topology implementation.
   - Few neural-decoder papers report run-to-run variance. This is useful and honest.
3. **Diagnosis-driven architecture.** Phase 1 found a hub-dominated, oversmoothing, low-rank representation. C4 (residual + LayerNorm) fixes the training-side problem and gives 1.47x MWPM against 1.86x for the default backbone.

**Positioning:**
- **Don't claim state of the art:** Lange et al. beat MWPM and belief-matching with about 10^9 samples and 1.36M parameters ([arXiv:2307.01241](https://arxiv.org/abs/2307.01241)). This repo is at 300k samples and 79k parameters, so a sample-efficiency and controlled-ablation story fits the evidence better.
- **Report against all three baselines:** MWPM, correlated matching and belief-matching, on stored shots (direction 2).

## 4. Top 3, and why

1. **Fresh data every epoch (direction 1).**
   - **Why:** best-supported lever (Runs 6 to 11; the C4 train/val gap; the Lange and on-the-fly literature).
   - **Cost:** no extra GPU time.
   - **Also enables:** directions 7 and 9, and makes tonight's ZX targets cheap to compute at scale.
   - **If it fails:** that is informative too: it points to capacity.
2. **Honest baselines and a reproducible evaluation set (direction 2).**
   - **Why:** zero GPU, and required for an honest preprint. Correlated matching moves the bar by 25% (1.B). The big test is not reproducible across machines (1.B). No trained weights are kept.
   - **Also enables:** directions 4, 6 and 8 depend on it.
3. **Hyperedge correlation edges (direction 3).**
   - **Why:** the most specific structural gap in this graph. 27% of observable-flipping probability lives in hyperedges the input never shows (1.C). X-Z information can only pass through the hub (Phase 1). This exact information gives correlated matching its 25% (1.B) and gives neural decoders their edge in the literature (Varbanov et al.).
   - **Cost:** variant (a) is about a day of code and one 3-run session.

**Runner-up: the matching-informed hybrid (direction 4).** If the goal is narrowly "beat MWPM at d=5 soon", it is the most direct route. The crude no-training rule already reaches 1.23x, and the oracle bound is 0.73x (1.A). It is ranked below the top 3 because it changes the claim from a GNN decoder to a hybrid, and its honest bar is correlated matching (about 495, estimated), not MWPM.

**Suggested order for the next two weeks (about 30 GPU h each):**
- **Week 1:**
  - CPU and code only: directions 2 and 5.
  - Analyse aux_C4_v2.
  - GPU: C4 + fresh data, 3 runs (about 4.5 h). C4 + hyperedge pairs, 3 runs (about 6 h). The residual-only and LayerNorm-only splits already planned (about 8 h).
- **Week 2:** combine whatever passed, then capacity (direction 7) on the RTX 5090 if it arrives.

## Sources
- Lange et al., "Data-driven decoding of quantum error correcting codes using graph neural networks", PRR 2025: [arXiv:2307.01241](https://arxiv.org/abs/2307.01241) ([v2 HTML](https://arxiv.org/html/2307.01241v2), read for this note)
- Higgott, Bohdanowicz, Kubica, Flammia, Campbell, "Improved decoding of circuit noise and fragile boundaries of tailored surface codes", PRX 13, 031007: [arXiv:2203.04948](https://arxiv.org/abs/2203.04948)
- Varbanov, Serra-Peralta, Byfield, Terhal, "Neural network decoder for near-term surface-code experiments", PRR 7, 013029: [arXiv:2307.03280](https://arxiv.org/abs/2307.03280)
- Bausch et al., "Learning high-accuracy error decoding for quantum processors", Nature 2024: [PMC11602728](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11602728/), [Google blog](https://blog.google/technology/google-deepmind/alphaqubit-quantum-error-correction/), [arXiv:2310.05900](https://arxiv.org/abs/2310.05900)
- Gicev, Hollenberg, Usman, "A scalable and fast artificial neural network syndrome decoder for surface codes": [arXiv:2110.05854](https://arxiv.org/abs/2110.05854)
- PyMatching on PyPI: [pypi.org/project/PyMatching](https://pypi.org/project/PyMatching/). Correlated matching was confirmed in the installed 2.4.0 (`enable_correlations`).
- BeliefMatching implementation: [github.com/KuantumBS/BeliefMatching](https://github.com/KuantumBS/BeliefMatching)
- Read from search excerpts only (not verified in full text): [HyperNQ, arXiv:2511.01741](https://arxiv.org/abs/2511.01741); [Scalable Neural Decoders, arXiv:2604.08358](https://arxiv.org/pdf/2604.08358); [AI-based pre-decoders, arXiv:2604.12841](https://arxiv.org/pdf/2604.12841); [QAdapt, arXiv:2607.28422](https://arxiv.org/pdf/2607.28422)
