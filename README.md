# SpiderTrace

SpiderTrace traces how Pauli X, Y and Z errors propagate through Clifford circuits, and renders each propagation step as a ZX diagram (via PyZX).

The repository has two parts:

1. **The `spidertrace` package**: a small Pauli-propagation engine plus ZX-diagram visualisation.
2. **A research extension**: a neural QEC decoder study that uses Pauli propagation to build auxiliary training targets for a GNN decoder. See [Research Extension](#research-extension-neural-qec-decoder-with-zx-supervision).

---

## The `spidertrace` package

### Scope

Supported:

- Gates: `H`, `CNOT`, `CZ`
- Errors: single-qubit `X`, `Y`, `Z` on any number of qubits (Y also arises from propagation)
- Output: a per-gate trace of the Pauli frame, plus ZX diagrams of each step

Not supported:

- State simulation
- Non-Clifford gates (T, arbitrary rotations), `S`, `SWAP`, measurement and reset

Propagation is conjugation, `P → U P U†`, with global phases ignored. The engine does not decode or correct errors.

### Installation

```bash
git clone https://github.com/Hope-Alemayehu/SpiderTrace.git
cd SpiderTrace
pip install -e .          # installs spidertrace + pyzx
```

### Python API

```python
from spidertrace import Gate, PauliError, propagate_errors, save_complete_visualization

circuit = [Gate("H", (0,)), Gate("CNOT", (0, 1)), Gate("CZ", (0, 1))]
errors  = [PauliError(0, "X")]

trace = propagate_errors(circuit, errors)     # list[TraceStep], one per gate
for step in trace:
    print(step.gate.name, step.errors_after)  # errors_after: {qubit: "X" | "Y" | "Z"}

save_complete_visualization(circuit, errors, trace, "my_circuit")  # PNGs
```

`propagate_errors` keeps at most one Pauli per qubit. When a gate cancels a Pauli, that qubit is dropped from the dict.

### Propagation rules

**H**: `X ↔ Z`, `Y → Y`

**CNOT** (control `c`, target `t`). The rule is implemented in the symplectic (x, z) form: `x_t ^= x_c`, `z_c ^= z_t`.

| Input | Output |
|---|---|
| X_c | X_c X_t |
| Z_t | Z_c Z_t |
| X_t | X_t |
| Z_c | Z_c |
| X_c Z_t | Y_c Y_t |

Y inputs are handled by the same rule, e.g. `Y_c → Y_c X_t`.

**CZ** (symmetric). The full 16-entry conjugation table is in [engine.py](spidertrace/engine.py). The main cases:

| Input | Output |
|---|---|
| X_c | X_c Z_t |
| X_t | Z_c X_t |
| Z_c / Z_t | unchanged |
| Y_c | Y_c Z_t |
| Y_t | Z_c Y_t |
| X_c X_t | Y_c Y_t |

### ZX diagram output

[zx_visual.py](spidertrace/zx_visual.py) produces:

1. the clean circuit
2. the circuit with the initial errors
3. one diagram after each gate

```bash
python -m spidertrace.display_all_zx   # built-in example: H(0), CNOT(0,1), X on q0
python -m tests.test_custom            # interactive circuit builder (stdin prompts)
```

### Tests

```bash
python -m pytest tests --ignore=tests/test_custom.py
```

The tests cover H, CNOT and CZ propagation (including Y inputs and the CZ symmetry check), multi-gate circuits and ZX visual generation. `tests/test_custom.py` is interactive, so it is excluded above.

---

## Research Extension: Neural QEC Decoder with ZX Supervision

This study asks whether Pauli propagation traces are a useful auxiliary training signal for a GNN decoder. All three arms share one GNN backbone and differ only in their auxiliary head:

| Arm | Trained on | Aux target |
|---|---|---|
| **GNN-A** | syndrome → logical flip | none |
| **GNN-Raw** | syndrome → logical flip + aux | `raw_target`: fired faults' Paulis at their **original** location |
| **GNN-ZX** | syndrome → logical flip + aux | `zx_target`: the same Paulis **propagated** to the final frame |

The aux head is used only in training. At inference, all three arms predict the logical flip from the syndrome graph alone.

### Pipeline ([qec_zx_dataset.py](qec_zx_dataset.py))

- **Circuit**: Stim rotated surface-code memory-Z circuit, `rounds = d`, with uniform circuit-level depolarizing noise `p`. Noise is applied after Clifford gates, before each round on data qubits, before measurement and after reset.
- **Single source of truth**: each shot comes from one call to the (non-decomposed) DEM sampler with `return_errors=True`. The detector pattern, logical label and both aux targets therefore come from the same set of fired DEM errors. `raw_target` and `zx_target` have identical shape `(num_qubits, 4)` (one-hot over I/X/Y/Z) and differ only in whether propagation is applied.
- **Propagation seam**: `ZXPropagator` has two implementations:
  - `ReferenceZXPropagator`: Stim `FlipSimulator`. **This is the default in `build_fault_tables`, and it is what all current training runs use.**
  - `SpiderTraceAdapter`: drives `spidertrace.engine.propagate_errors` over the circuit's gate layers. It handles `R`/`MR` resets as `X→I, Y→Z, Z→Z`. `python qec_zx_dataset.py` validates it against the reference on single-qubit data faults at layer boundaries. Writing it exposed a CNOT Y-propagation bug in the engine, which has since been fixed. To use it, pass `propagator=SpiderTraceAdapter(circ)` to `build_fault_tables` or `make_dataloader`.
- **Decoding graph** (`build_dem_graph`): a fixed graph derived from the decomposed DEM, i.e. the graph MWPM decodes on. It has one node per detector plus a single boundary node. Parallel mechanisms are XOR-merged. Edge weight is `-log(p/(1-p))`.
  - Node features: `[fired, x, y, t, is_boundary]`
  - Edge features: `[w_norm, dx, dy, dt, is_boundary_edge, flips_obs]`

### Model and training ([gnn_models.py](gnn_models.py))

- **Backbone**: a linear node encoder, then 4 × `GINEConv` layers (hidden size 96, 2-layer MLP, dropout 0.1). The pooled readout concatenates add, mean and max pooling (288 dims). `--mp-layers` overrides the layer count.
- **Heads**:
  - Flip head: `Linear(288, 1)`
  - Aux head (Raw/ZX only): `Linear(288, num_qubits × 4)`
- **Loss**: `BCE(flip) + λ · mean per-qubit CE(aux)`, with `pos_weight = 1.0`.
- **Optimisation**: Adam (lr 1e-3), early stopping on validation loss (patience 10).
- **Sampling**: `build_loaders` uses a class-balanced `WeightedRandomSampler` by default. Pass `balanced_sampler=False` for the natural class prior.

### Running

```bash
pip install stim pymatching torch torch_geometric scipy matplotlib   # tested: stim 1.16.0, pymatching 2.4.0, torch 2.12, PyG 2.7

python qec_zx_dataset.py                         # pipeline smoke test + SpiderTrace adapter validation
python gnn_models.py --dry-run                   # 20-epoch d=3 sanity check of all three arms
python gnn_models.py --d 5 --p 0.003 --shots 50000 --seeds 0 1 2 --lambdas 0.01 0.1 0.5
python gnn_models.py --d 5 --p 0.003 --mcnemar   # paired McNemar, GNN-ZX vs GNN-Raw
python qec_run.py --d 5 --p 0.003                # fair runner (see below)
python evaluate_mwpm_circuit.py                  # MWPM baseline, same circuit-level noise
python validate_dem_graph.py                     # GNN-A on DEM graph, 4 vs 8 layers vs MWPM
```

`gnn_models.py` and `qec_run.py` handle the aux arms differently:

- **`gnn_models.py` (`run_experiment`, `--mcnemar`)**:
  - Only GNN-ZX sweeps λ. GNN-Raw uses the first λ in the list.
  - λ is selected by validation loss.
  - Metrics are reported on the validation split.
- **`qec_run.py`**:
  - Raw and ZX both sweep the same λ grid, and λ is selected by **validation LER**.
  - LER is reported on a held-out test split. That split is fixed by `--split-seed`, so it is identical across arms and seeds.
  - It runs a paired McNemar test (ZX vs Raw) on that test split.
  - It writes a timestamped JSON with provenance: git commit, dirty flag, config and library versions.

### Current results

For reference, MWPM under the same circuit-level noise at d=5, p=0.003 reaches **LER 0.0036**, from 50k shots ([mwpm_circuit_level.json](results/mwpm_circuit_level.json)). The observable flip rate is 0.159, so the trivial "never flip" decoder scores LER ≈ 0.159.

| File | Setup | GNN-A | GNN-Raw | GNN-ZX |
|---|---|---|---|---|
| [gnn_d5_p0.003.json](results/gnn_d5_p0.003.json) | d=5, p=0.003, 50k shots, 1 seed, 100 epochs, balanced sampler, **pre-DEM-graph (kNN) graph** | 0.1813 | 0.1428 (λ=0.01) | 0.1406 (λ=0.1) |
| [gnn_d3_p0.01.json](results/gnn_d3_p0.01.json) | d=3, p=0.01, 10k shots, 1 seed, 30 epochs, pre-DEM-graph | 0.2400 | 0.2235 | 0.2430 |
| [dem_graph_gnnA_d5_p0.003.json](results/dem_graph_gnnA_d5_p0.003.json) | d=5, p=0.003, 50k shots, DEM graph, natural sampling | 0.0157 (4 layers); 0.1505 (8 layers, collapsed to all-zero) | — | — |

How to read these numbers:

- **The pre-DEM-graph results are not competitive.** At d=5, the ordering is GNN-ZX < GNN-Raw < GNN-A. That is one seed, and the ZX–Raw gap is 0.002. All three are near or above the trivial baseline and about 40× worse than MWPM.
- **The DEM graph improved GNN-A by an order of magnitude.** It reaches 0.0157, but that is still about 4× MWPM and misses the 0.012 target set in `validate_dem_graph.py`.
- **Not yet run on the DEM graph**: GNN-Raw, GNN-ZX, a McNemar result and `qec_run.py`. No committed result supports a claim that ZX supervision helps.
- [gnn_d5_p0.003_bugged_posweight.json](results/gnn_d5_p0.003_bugged_posweight.json) is kept as a record of the class-weighted `pos_weight` bug, which made the models collapse.

### Earlier MLP baseline (legacy)

The first iteration used an MLP instead of a GNN and a simpler noise model. It is kept for reference.

- **Data** ([generate_dataset.py](generate_dataset.py), [data/](data/)):
  - Data-qubit depolarizing noise only (`before_round_data_depolarization`), `rounds = 2`.
  - ZX features: data-qubit faults propagated by SpiderTrace through the noiseless `rounds=1` circuit.
  - 50k shots per file, for d ∈ {3, 5, 7} × p ∈ {0.001, 0.005, 0.01, 0.03, 0.05, 0.1}.
- **Models** ([train.py](train.py)):
  - `BaselineA`: syndrome → flip
  - `BaselineB`: ZX features → flip. The ZX features are given at inference, so this is an oracle.
  - `ProposedModel`: syndrome → flip + ZX-reconstruction aux head
- **Results**:
  - `results/d{3,5,7}_results.json` and `results/d7_results_20k.json`
  - MWPM baseline: [evaluate_mwpm.py](evaluate_mwpm.py) → `results/d{3,5,7}_mwpm.json`
  - Figures: [plot_result.py](plot_result.py) → `figures/*.pdf`
- **Outcome**: `ProposedModel` did not meaningfully improve on `BaselineA`. For example, at d=3, p=0.1 it scored 0.2126 vs 0.2078.

---

## Project structure

```
SpiderTrace/
├── spidertrace/              # the package
│   ├── circuit.py            #   Gate
│   ├── error.py              #   PauliError
│   ├── engine.py             #   propagate_errors, TraceStep, gate rules
│   ├── zx_visual.py          #   ZX diagram generation (PyZX)
│   └── display_all_zx.py     #   example: all diagrams for one circuit
├── tests/                    # engine / CZ / ZX-visual tests; test_custom.py is interactive
│
├── qec_zx_dataset.py         # circuit-level pipeline, DEM graph, SpiderTraceAdapter + validation
├── gnn_models.py             # GNN-A / GNN-Raw / GNN-ZX, training, McNemar
├── qec_run.py                # fair, provenance-recording experiment runner
├── validate_dem_graph.py     # GNN-A on DEM graph vs MWPM (4 vs 8 layers)
├── evaluate_mwpm_circuit.py  # MWPM baseline, circuit-level noise
│
├── generate_dataset.py       # legacy: .npz dataset (data-noise only)
├── train.py                  # legacy: MLP baselines
├── evaluate_mwpm.py          # legacy: MWPM on .npz data
├── plot_result.py            # legacy: figures from MLP results
├── verify_dataset.py, data_npz.py   # legacy: .npz sanity checks
├── data/  results/  figures/
└── pyproject.toml
```

## Known issues

- `pyproject.toml` declares the console scripts `spidertrace-test` and `spidertrace-custom` pointing at `spidertrace.test_simple` / `spidertrace.test_custom`. Those modules live in `tests/`, so both scripts are broken.
- `qec_zx_dataset.build_fault_tables` compiles its DEM sampler without a seed, and the `seed` argument of `sample_tuples` is unused. As a result, `gnn_models.build_loaders` does not produce reproducible datasets. `qec_run.py` works around this by compiling a seeded sampler itself.
- Early stopping is on validation loss, which for Raw/ZX includes the `λ·aux` term. The stopping criterion therefore varies with λ.

## Acknowledgments

Built on [PyZX](https://github.com/Quantomatic/pyzx), [Stim](https://github.com/quantumlib/Stim) and [PyMatching](https://github.com/oscarhiggott/PyMatching).
