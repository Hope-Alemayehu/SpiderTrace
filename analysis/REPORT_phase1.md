# Phase 1: diagnosis of GNN-A on the decoding graph (d=3 and d=5, p=0.003, rounds=d)

Branch `arch-exploration` at f7ac456. Nothing committed; `qec_run.py` and `gnn_models.py` untouched. All scripts and outputs are in `analysis/` (`phase1_results/*.json`, training logs, trained d=3 weights). Every number below is measured, except where marked (theory) or (formula).

**Environment:** `docs/flywheel_report/.venv` (Python 3.12.10, torch 2.14.1+cpu, torch_geometric 2.7.0, stim 1.16.0, pymatching 2.4.0), CPU only.

**Graph:** the GNN's own input graph, `qec_zx_dataset.build_dem_graph(build_circuit(d, 0.003))`. `analysis/common.py` re-derives the raw edge probabilities with a verbatim copy of the merge loop and asserts it reproduces `build_dem_graph`'s edges and `w_norm` exactly.

## 1. Graph structure (`p1_graph.py`, `p1b_sublattices.py`)

| | d=3 | d=5 |
|---|---|---|
| Nodes (detectors + 1 boundary) | 25 (24 + 1) | 121 (120 + 1) |
| Undirected edges | 78 | 502 |
| Boundary edges = boundary-node degree | 24 | 72 |
| Detectors adjacent to the boundary node | 24 of 24 (all) | 72 of 120 |
| Detector degree: histogram {degree: count} | {3: 4, 5: 8, 6: 8, 8: 4} | {3: 6, 4: 2, 5: 8, 6: 26, 7: 16, 8: 20, 9: 18, 11: 12, 12: 12} |
| Detector degree: mean, max | 5.5, 8 | 7.77, 12 |
| Diameter with the boundary node (hops) | 2 | 4 |
| Max hops from any detector to the boundary node | 1 | 2 |
| Mean hops between detector pairs, with boundary | 1.80 | 2.54 |
| Without the boundary node | disconnected: components of 16 and 8 | disconnected: components of 72 and 48 |
| Diameter within a component, without boundary | 6 | 10 |

**The two components are the Z-type and X-type detector sublattices.** Component A holds the detectors of rounds 0 to d (16 at d=3, 72 at d=5) and **all** observable-flipping edges (8 and 18). Component B holds the rounds 1 to d-1 detectors (8 and 48) and **no** observable-flipping edge. The decomposed DEM never puts an X-type and a Z-type detector on the same edge, so the boundary node is the only path between them. Yet component B carries 46% of the fired detectors at d=5 (2.37 of 5.13 per shot), and 67% of d=5 shots fire in both components.

Consequences:
- With the hub, everything is within 4 hops at d=5 and 2 hops at d=3. A 4-layer GNN covers the whole graph. Depth is not a coverage limit at these sizes.
- Without the hub, the label-carrying sublattice has diameter 10 at d=5.
- The boundary node has degree 72 at d=5 (16T + 32 in general, 33,376 at the T=2084 graph from the rounds study) and degree 24 at d=3, where it touches every detector.

## 2. Hop span of logical-error chains (`p1_graph.py` for theory, `p2_chains.py` for 200,000 sampled shots per d)

**Minimal logical operator** (shortest odd-parity closed walk through the boundary, on the parity-doubled graph; the minimum-weight one under the raw edge weights coincides with it):

| | d=3 | d=5 |
|---|---|---|
| Edges | 3 (2 boundary edges + 1) | 5 (2 boundary edges + 3) |
| Detectors | 2 | 4 |
| Detector span without boundary (hops) | 1 | 3 |
| Minimum weight (sum of -log p/(1-p)) | 13.36 | 22.87 |

**Sampled error chains** (fired decomposed-DEM edges mod 2, grouped into connected components over detectors; boundary edges do not join chains through the hub):

| d=5 | All chains | Odd parity (would flip the observable if uncorrected) | MWPM failure loops |
|---|---|---|---|
| Count | 636,265 | 47,614 | 650 (MWPM LER 0.00325) |
| Edges: histogram | 1: 97.0%, 2: 2.8%, 3: 0.15%, 4+: 0.007% | 1: 97.4%, 2: 2.5%, 3: 0.11% | 5: 208, 6: 308, 7: 67, 8: 45, 9: 14, 10: 7, 11: 1 |
| Detectors: mean | 1.64 | 1.03 | 5.04 (4 to 10) |
| Detector span without boundary: histogram (hops) | 0: 37.8%, 1: 61.0%, 2: 1.2%, 3: 0.02% | 0: 97.4%, 1: 2.5%, 2: 0.07%, 3: 1 chain | **3: 557 (85.7%), 4: 72 (11.1%), 5: 17, 6: 3, 7: 1** |
| Detector span with boundary | same, max 3 | same | 2: 438, 3: 196, 4: 16 |
| Boundary edges | 0: 61%, 1: 39%, 2: 0.03% | 1: 100% | 2: 100% |

| d=3 | All chains | Odd parity | MWPM failure loops |
|---|---|---|---|
| Count | 124,841 | 17,342 | 1,335 (MWPM LER 0.00668) |
| Edges | 1: 98.1%, 2: 1.8% | 1: 97.8%, 2: 2.1% | 3: 752, 4: 531, 5: 46, 6: 5, 7: 1 |
| Detector span without boundary | 0: 60%, 1: 39.5%, 2: 0.3% | 0: 97.8%, 1: 2.2%, 2: 8 chains | 1: 954, 2: 368, 3: 12, 4: 1 |

Checks: every MWPM failure contained exactly one odd loop (0 with zero, 0 with several), and the parity read off the matched edges equals `Matching.decode` on all 400,000 shots. The decomposed DEMs have no undetectable components.

**Reading.** The chains that actually flip the observable are almost always a single boundary edge (one detector, span 0). What a decoder has to resolve is the *failure loop*: the true chain XOR the correction, which is a minimal logical operator. At d=5 these span 3 to 4 hops between detectors (97% within 4) and 2 to 4 hops through the hub. So:
- From a loop's central detector, ceil(span/2) ≤ 4 layers sees all of it in every case; from an end detector, 97% of loops at d=5 are within 4 hops.
- 4 layers is enough receptive field at d=5. At d=7 the minimal loop has 6 detectors spanning 5 hops, so an end detector would need 5 layers; the centre still needs 3.

The receptive field is therefore not the first-order limit. What the hub does is let every node reach every other node in 4 hops through one shared vector.

## 3. Oversmoothing on a trained d=3 GNN-A (`p3_oversmoothing.py`)

**Training** (repo defaults: default GNN-A, 78,053 parameters, Adam lr 1e-3 constant, batch 256, early stopping on val loss with patience 20, model seed 0, split seed 12345, CPU, 4 threads):

| Pool shots (train / val / test) | Epochs run (best) | Wall | Split-test LER | Fresh 20k-shot LER (95% CI) | MWPM on the same 20k | Never-flip on the same 20k |
|---|---|---|---|---|---|---|
| 10,000 (7,000 / 1,500 / 1,500) | 75 (55) | 250 s | 21/1500 = 0.0140 | 276/20000 = **0.0138** [0.0122, 0.0155] | 138 = 0.0069 | 1,269 = 0.0635 |
| 30,000 (21,000 / 4,500 / 4,500) | 97 (77) | 917 s | 40/4500 = 0.0089 | 193/20000 = **0.0097** [0.0083, 0.0111] | 138 = 0.0069 | 1,269 = 0.0635 |

Both learned (2.0x and 1.4x MWPM's error count). The 30k model is used as "the trained model" below; the 10k model gives the same picture (its JSON is saved too). Metrics are over the first 4,000 fresh shots (0.85 fired detectors per shot on average; 1,067 shots have at least 2 fired detectors), in eval mode, so no dropout.

**Per layer, trained 30k model.** Layer 0 is the node encoder output; layer k is after conv k and ReLU.

| Layer | Cosine, all node pairs | Cosine, fired vs fired | Cosine, unfired vs unfired | Cosine, fired vs unfired | Cosine, boundary vs unfired | Pairs with cosine > 0.99 | Dirichlet energy: all / fired / unfired / boundary | Mean norm: fired / unfired / boundary | Effective rank | Linear probe for the fired bit, balanced accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.790 | 0.973 | 0.846 | 0.394 | 0.492 | 0.000 | 0.51 / 1.92 / 0.36 / 1.06 | 5.29 / 3.22 / 3.30 | 3.31 | 1.000 |
| 1 | 0.757 | 0.944 | 0.796 | 0.194 | 0.592 | 0.045 | 2.24 / 5.22 / 1.34 / 6.40 | 8.33 / 1.73 / 6.66 | 1.46 | 0.998 |
| 2 | 0.770 | 0.794 | 0.806 | 0.295 | 0.462 | 0.352 | 0.44 / 0.58 / 0.32 / 1.11 | 4.69 / 4.00 / 4.20 | 1.50 | 0.950 |
| 3 | 0.851 | 0.802 | 0.858 | 0.608 | 0.813 | 0.331 | 1.66 / 1.36 / 1.02 / 5.20 | 4.16 / 4.49 / 16.9 | 1.34 | 0.961 |
| 4 | 0.907 | 0.838 | 0.921 | 0.811 | 0.760 | 0.369 | 1.24 / 0.88 / 0.77 / 3.87 | 2.56 / 4.06 / 12.1 | 1.64 | **0.895** |

Definitions: Dirichlet energy = mean over directed edges of ||x_u − x_v||² divided by the layer's mean ||x||² (scale-free; per group = edges whose source is in the group). Effective rank = participation ratio of the singular values of the centred node-embedding matrix (96 channels). Probe = class-balanced ridge classifier fit on half the shots, balanced accuracy on the other half.

**Same model at initialisation** (seed 0, before training):

| Layer | Cosine, all | Fired vs unfired | Pairs > 0.99 | Dirichlet all / boundary | Norm: detectors / boundary | Effective rank | Probe |
|---|---|---|---|---|---|---|---|
| 0 | 0.836 | 0.701 | 0.000 | 0.39 / 0.89 | 3.5 / 3.4 | 3.58 | 1.000 |
| 1 | 0.947 | 0.949 | 0.036 | 2.56 / 8.19 | 3.1 / 14.9 | 1.19 | 1.000 |
| 2 | 0.975 | 0.986 | 0.380 | 1.67 / 5.34 | 6.2 / 22.9 | 1.17 | 0.958 |
| 3 | 0.988 | 0.993 | 0.673 | 0.95 / 2.98 | 9.9 / 28.6 | 1.18 | 0.927 |
| 4 | **0.997** | 0.998 | **0.920** | 1.10 / 3.47 | 17.3 / 55.3 | **1.03** | 0.893 |

**Does "4 layers makes nodes indistinguishable at d=3" hold?**
- **At initialisation, nearly:** cosine 0.997 across all node pairs, 92% of pairs above 0.99, effective rank 1.03, and norms growing 5x per detector and 16x at the boundary across the 4 layers. The untrained network smooths and blows up at once. This matches the d=5 initial logit of +147 found earlier.
- **After training, no, but the representation is partly smoothed and very low-rank.** Fired and unfired detectors stay linearly separable at layer 4 (balanced accuracy 0.895, from 1.000 at the input), Dirichlet energy stays around 1 instead of collapsing to 0, and only 37% of pairs exceed 0.99 cosine. But the fired-vs-unfired cosine rises from 0.19 (layer 1) to 0.81 (layer 4), the fired nodes' norm drops below the unfired nodes' by layer 4 (2.6 vs 4.1), and the effective rank is 1.3 to 1.6 out of 96 channels at every layer after the first. The model has learned to keep the fired bit decodable while using a 1- to 2-dimensional subspace.
- **The boundary node is different from everything.** Its norm is 3x the detectors' at layer 4 (12.1 vs 4.1), it has the highest Dirichlet energy at every layer, and it dominates the max pool. The Dirichlet energy is concentrated on the hub's edges, not spread across the lattice.

## 4. How much of each node's signal passes through the boundary node

**Structural (formula, `p1_graph.py`).** GINEConv aggregates (1+ε)x_v + Σ messages with ε = 0 at init, so walk counts of A + I bound how inputs mix. Fraction of length-k walks ending at a detector that visit the boundary node at least once:

| k | d=3, all detectors (all are boundary-adjacent) | d=5, all detectors | d=5, boundary-adjacent (72) | d=5, not adjacent (48) |
|---|---|---|---|---|
| 1 | 0.163 | 0.081 | 0.136 | 0.000 |
| 2 | 0.504 | 0.365 | 0.583 | 0.038 |
| 3 | 0.661 | 0.502 | 0.664 | 0.260 |
| 4 | **0.771** | **0.625** | 0.771 | 0.406 |
| 8 | 0.947 | 0.875 | 0.928 | 0.796 |

**Empirical, on the trained d=3 model (30k):** recomputing the forward pass with every boundary edge removed changes detector embeddings by 29% (layer 1), 75% (2), 75% (3) and 79% (layer 4) in relative norm; at initialisation 20%, 51%, 69%, 74%. Without the boundary pathway the trained model predicts a flip on 0.1% of shots instead of 5.9%: **its flip predictions go entirely through the hub.** The 10k model behaves the same (90% change at layer 4; flip predictions 5.4% to 3.1%).

So at d=5 a typical detector's layer-4 representation is about 60% hub-derived (77% if it touches the boundary), and in the trained model the decision itself is read through the hub.

## 5. Shots with zero fired detectors (`p1_graph.py`, 1,000,000 seeded shots)

| | d=3 | d=5 |
|---|---|---|
| Zero-fired fraction (95% CI) | 0.5988 [0.5978, 0.5997] | **0.0761 [0.0756, 0.0767]** |
| Probability that no DEM error fires at all | 0.5986 | 0.0758 |
| Observable flips among zero-fired shots | 0 of 598,786 | 0 of 76,137 |
| Mean fired detectors per shot | 0.86 | 5.13 |

Zero-fired shots are, to within sampling, exactly the shots with no error at all, and the correct answer for them is always "no flip". For C2 the zero-fired case needs a defined output, and that output can be a constant.

## Which Phase 2 configs are worth running

Ranked by what the numbers say.

1. **C2, `--readout fired`: yes, first priority.** Mean fired detectors per shot at d=5 is 5.1 of 121 nodes (section 5), the hub out-norms the detectors 3x and dominates the max pool (section 3), and the fired/unfired signal survives to layer 4 but is diluted at the readout. Zero-fired shots (7.6%) are always "no flip", so the explicit handling is a fixed vector plus, if wanted, a hard rule; both are testable. Risk: the current model routes its decision through the hub (section 4); fired-only pooling forces it onto the fired nodes, which have the information but have not been the readout path.
2. **C5, `--head mlp`: yes.** The pooled vector is read by one linear layer, while the learned node embeddings have effective rank 1.3 to 1.6 and the required decision (parity of boundary-to-boundary crossings) is not linear in pooled sums. Cheap: about 45k extra parameters, same per-epoch cost.
3. **C3, `--boundary per-round`: yes, but it is the invasive one.** The hub carries 60 to 77% of 4-hop walk mass at d=5 and is the only bridge between the label-carrying sublattice and the other one; failure loops only need 2 to 4 hops through it. Per-round boundary nodes (6 at d=5) cut the hub degree from 72 to about 12 and keep the hub mixing local in time. The expected gain at fixed d=5 is uncertain, since the hub is not a coverage problem here; its value is for long-T transfer, where the hub degree reaches 33,376. Test: every merged DEM edge maps one-to-one to an edge of the new graph, with the same raw probability and observable flag.
4. **C4, `--depth N --residual` + normalization: run it as an optimisation fix at depth 4, and as the 8-layer retest.** Depth is not the bottleneck at d=5 (section 2: all failure loops within 4 hops with the hub, and from a loop's centre without it). But the untrained net oversmooths and blows up in 4 layers (section 3), and the 8-layer failure is consistent with that, so residual + norm at depth 4 tests whether the init pathology costs accuracy once Run 9's readout change is removed. Keep the two factors separate: depth 4 with residual + norm, then depth 8 with residual + norm. Run 9's negative result means do not expect a win from normalisation alone.
5. **C1, `--readout mean`: low priority at fixed d.** At d=5 the add pool is exactly 121x the mean pool, so dropping it is a scale change the linear head can absorb; the real change is losing the max pool, which section 3 shows is mostly the hub's vector. It matters for transfer to larger graphs, not for this ablation. Run it only if a Kaggle session has room, or fold it into C2 (fired-only pooling has no add/max question).

Not in the list but visible in the data: the hub also has the largest activations at every layer, so any of C2 to C4 that reduces its influence should be checked for the initial-logit scale as well (the earlier +147 at d=5).

## Files

| File | Purpose |
|---|---|
| `analysis/common.py` | graph loader, verified raw-probability replica |
| `analysis/p1_graph.py` | sections 1, 2 (theory), 4 (formula), 5 |
| `analysis/p1b_sublattices.py` | the two no-boundary components |
| `analysis/p2_chains.py` | section 2 (sampled chains, MWPM failure loops) |
| `analysis/p3_oversmoothing.py` | sections 3 and 4 (empirical) |
| `analysis/phase1_results/` | all JSON outputs, `train_log_{10k,30k}.txt`, `gnn_a_d3_{10k,30k}.pt` |

Commands: `python analysis/p1_graph.py`, `python analysis/p1b_sublattices.py`, `python analysis/p2_chains.py --shots 200000`, `python analysis/p3_oversmoothing.py --shots 30000 --epochs 150 --patience 20 --tag 30k` (and `--shots 10000 --tag 10k`), all from the repo root with the venv above.
