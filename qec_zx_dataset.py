"""
qec_zx_dataset.py
=================
Single-source-of-truth data pipeline for the GNN-A / GNN-Raw / GNN-ZX study.

For a rotated surface code at distance ``d`` with uniform circuit-level
depolarizing noise at rate ``p`` (``rounds = d``), this emits, per shot,
a matched tuple:

    (graph, raw_target, zx_target, logical_label)

derived from ONE call to the DEM error sampler, so the detector graph, the
logical label, and BOTH auxiliary targets are guaranteed mutually consistent.

Fairness guarantee
------------------
Both ``raw_target`` and ``zx_target`` are per-qubit one-hot Pauli tensors of
IDENTICAL shape ``(num_qubits, 4)`` over {I, X, Y, Z}, built from the SAME set
of fired DEM-error representatives. They differ ONLY by the propagation
transform:

    raw_target  =  product of representative Paulis at their ORIGINAL location
    zx_target   =  product of the same Paulis PROPAGATED to the final frame

There is no separate "measurement-error vector": Stim's error explanation
assigns every DEM error a single Pauli representative (a bulk measurement error
is rendered as a Pauli on its ancilla/measure qubit). Both targets consume that
identical representative set, so measurement and reset errors are handled
*structurally* consistently. Under propagation, an ancilla-qubit Pauli is
correctly "explained away" (it dies at the next reset and leaves ~no data-qubit
residual), which is the physically right treatment of a measurement error --
and it happens identically in the construction of both targets.

Swap in SpiderTrace
-------------------
``ZXPropagator`` defines the seam. The reference implementation uses Stim's
``FlipSimulator`` (a stabilizer Pauli-frame tracker) and doubles as a
validation oracle for SpiderTrace. Replace it with a thin adapter around your
engine; everything else is unchanged.

Caveat on sub-tick timing: the reference injects a fault at its ``tick_offset``
(layer boundary). For 2-qubit *gate* faults that occur mid-layer, exact
propagation depends on within-layer position; SpiderTrace, which knows the exact
gate, is authoritative there. Use the reference to unit-test SpiderTrace on
before-round data faults (clean layer boundaries) where the two MUST agree.

Verified against stim 1.16.0.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional
import numpy as np
import stim

PAULI_CHAR = "_XYZ"  # index 0=I,1=X,2=Y,3=Z  (matches stim.PauliString indexing)


# --------------------------------------------------------------------------- #
# 1. Circuit
# --------------------------------------------------------------------------- #
def build_circuit(d: int, p: float, rounds: Optional[int] = None,
                  basis: str = "z") -> stim.Circuit:
    """Rotated surface code, uniform 4-parameter circuit-level depolarizing noise."""
    if rounds is None:
        rounds = d
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=d,
        rounds=rounds,
        after_clifford_depolarization=p,       # 2-qubit gate noise
        before_round_data_depolarization=p,    # data idling per round
        before_measure_flip_probability=p,     # measurement errors
        after_reset_flip_probability=p,        # reset errors
    )


# --------------------------------------------------------------------------- #
# 2. ZX propagation seam (replace ReferenceZXPropagator with SpiderTrace)
# --------------------------------------------------------------------------- #
class ZXPropagator:
    """Interface: given a representative Pauli fault, return the final-frame
    Pauli string on all qubits."""
    def propagate(self, qubits: Sequence[int], paulis: Sequence[int],
                  tick_offset: int) -> stim.PauliString:
        raise NotImplementedError


class ReferenceZXPropagator(ZXPropagator):
    """Reference propagator via stim.FlipSimulator. Use to validate SpiderTrace.

    NOTE: the circuit's leading RESET clears any frame injected before tick 0,
    so faults MUST be injected at their tick_offset (which is always >= the
    resets). This is handled below.
    """
    def __init__(self, circuit: stim.Circuit):
        self.N = circuit.num_qubits
        # Deterministic propagation requires the noiseless circuit.
        self._instructions = list(circuit.without_noise().flattened())

    def propagate(self, qubits, paulis, tick_offset) -> stim.PauliString:
        sim = stim.FlipSimulator(
            batch_size=1, disable_stabilizer_randomization=True, num_qubits=self.N
        )
        ticks = 0
        injected = (tick_offset == 0)
        if injected:
            for q, pl in zip(qubits, paulis):
                sim.set_pauli_flip(PAULI_CHAR[pl], qubit_index=q, instance_index=0)
        for inst in self._instructions:
            sim.do(inst)
            if inst.name == "TICK":
                ticks += 1
                if ticks == tick_offset and not injected:
                    for q, pl in zip(qubits, paulis):
                        sim.set_pauli_flip(PAULI_CHAR[pl], qubit_index=q, instance_index=0)
                    injected = True
        return sim.peek_pauli_flips()[0]


class SpiderTraceAdapter(ZXPropagator):
    """ZXPropagator backed by SpiderTrace's Pauli propagation engine.

    Reuses ``_stim_to_spider_gates`` (from generate_dataset.py) for the
    H / CNOT / CZ conversion, and drives ``spidertrace.engine.propagate_errors``
    over the gate layers that follow the injection point.

    Resets are modelled directly with stim's Z-basis reset rule: ``R`` / ``MR``
    discard the X-like component of the frame and keep the Z-like component
    (X -> I, Y -> Z, Z -> Z). This is how an X-type ancilla fault gets
    "explained away" at the next reset -- matching ``stim.FlipSimulator``. The
    trailing ``M`` (no reset) is ignored so data-qubit frames persist to the
    final frame. These are exactly the conditions under which the reference and
    SpiderTrace MUST agree, per the module docstring.

    The injection seam matches ReferenceZXPropagator: a fault at ``tick_offset``
    T is acted on by precisely the operations whose "tick bucket" (number of
    TICKs already seen) is >= T -- i.e. everything after the T-th TICK.
    """

    def __init__(self, circuit: stim.Circuit):
        # Lazy imports so users of the reference path don't need spidertrace.
        from generate_dataset import _stim_to_spider_gates

        self.N = circuit.num_qubits
        # Ordered event stream over the NOISELESS circuit. Each event is
        # (tick_bucket, "gates", [Gate, ...]) or (tick_bucket, "reset", [q, ...]).
        self._events: list = []
        instrs = list(circuit.without_noise().flattened())
        self.num_ticks = sum(1 for i in instrs if i.name == "TICK")

        tick = 0
        pending = stim.Circuit()        # run of consecutive H/CNOT/CZ instrs
        pending_tick = 0

        def flush():
            if len(pending) > 0:
                gates = _stim_to_spider_gates(pending)
                if gates:
                    self._events.append((pending_tick, "gates", gates))

        for inst in instrs:
            name = inst.name
            if name == "TICK":
                flush()
                pending = stim.Circuit()
                tick += 1
                pending_tick = tick
            elif name in ("H", "CX", "CNOT", "CZ"):
                if len(pending) == 0:
                    pending_tick = tick
                pending.append(inst)
            elif name in ("R", "MR"):
                # Reset clears the frame on its targets. Flush gates first so
                # ordering (gates-then-reset within a tick) is preserved.
                flush()
                pending = stim.Circuit()
                qs = [t.qubit_value for t in inst.targets_copy() if t.is_qubit_target]
                self._events.append((tick, "reset", qs))
            # M (no reset), DETECTOR, OBSERVABLE_INCLUDE, QUBIT_COORDS: no frame effect.
        flush()

    def propagate(self, qubits, paulis, tick_offset) -> stim.PauliString:
        from spidertrace.engine import propagate_errors
        from spidertrace.error import PauliError

        # current Pauli frame: qubit -> "X"/"Y"/"Z"
        errors = {q: PAULI_CHAR[pl] for q, pl in zip(qubits, paulis)}

        for ev_tick, kind, payload in self._events:
            if ev_tick < tick_offset:
                continue
            if kind == "reset":
                # Z-basis reset (R / MR -> |0>): the X-like component
                # anticommutes with the prepared state and is discarded; the
                # Z-like component commutes and survives in the frame. So
                # X -> I, Y -> Z, Z -> Z (matches stim.FlipSimulator exactly).
                for q in payload:
                    t = errors.get(q)
                    if t == "X":
                        errors.pop(q)
                    elif t == "Y":
                        errors[q] = "Z"
                    # t == "Z" or None: unchanged
            else:  # "gates"
                if not errors:
                    continue
                err_list = [PauliError(q, t) for q, t in errors.items()]
                trace = propagate_errors(payload, err_list)
                if trace:
                    errors = dict(trace[-1].errors_after)

        ps = stim.PauliString(self.N)
        for q, t in errors.items():
            ps[q] = PAULI_CHAR.index(t)   # "X"->1, "Y"->2, "Z"->3
        return ps


# --------------------------------------------------------------------------- #
# 3. Per-DEM-error fault tables (precomputed once per circuit)
# --------------------------------------------------------------------------- #
@dataclass
class FaultTables:
    num_qubits: int
    num_errors: int
    raw_pauli: List[stim.PauliString]     # length num_errors, over num_qubits
    zx_pauli: List[stim.PauliString]      # length num_errors, over num_qubits
    detector_coords: np.ndarray           # (num_detectors, 3) -> [x, y, t]
    num_detectors: int


def build_fault_tables(circuit: stim.Circuit,
                       propagator: Optional[ZXPropagator] = None) -> Tuple[FaultTables, stim.CompiledDemSampler]:
    """Precompute, for each DEM error, its raw and ZX-propagated Pauli string.

    Returns the tables and a DEM sampler bound to the SAME dem (so sampler
    error-column i corresponds to error i in these tables -- verified).
    """
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    sampler = dem.compile_sampler()
    ne = dem.num_errors
    N = circuit.num_qubits

    if propagator is None:
        propagator = ReferenceZXPropagator(circuit)

    expl = circuit.explain_detector_error_model_errors(
        reduce_to_one_representative_error=True)
    assert len(expl) == ne, (
        f"explanation/dem error count mismatch ({len(expl)} vs {ne}); "
        "do not rely on column alignment."
    )

    raw_pauli: List[stim.PauliString] = []
    zx_pauli: List[stim.PauliString] = []

    for e in expl:
        loc = e.circuit_error_locations[0]            # representative location
        qubits, paulis = [], []
        for gtc in loc.flipped_pauli_product:
            gt = gtc.gate_target
            q = gt.qubit_value
            pl = 1 if gt.is_x_target else (2 if gt.is_y_target else 3)
            qubits.append(q)
            paulis.append(pl)

        # raw: Pauli(s) at original qubit location, NOT propagated
        raw = stim.PauliString(N)
        for q, pl in zip(qubits, paulis):
            raw[q] = pl
        raw_pauli.append(raw)

        # zx: same fault propagated to final frame (via SpiderTrace / reference)
        if qubits:
            zx_pauli.append(propagator.propagate(qubits, paulis, loc.tick_offset))
        else:
            zx_pauli.append(stim.PauliString(N))

    # detector coords -> (num_detectors, 3)
    coord_map = circuit.get_detector_coordinates()
    nd = circuit.num_detectors
    coords = np.zeros((nd, 3), dtype=np.float32)
    for di, c in coord_map.items():
        coords[di, :len(c)] = c[:3]

    return FaultTables(N, ne, raw_pauli, zx_pauli, coords, nd), sampler


# --------------------------------------------------------------------------- #
# 4. PauliString -> one-hot (num_qubits, 4)
# --------------------------------------------------------------------------- #
def _accumulate_onehot(pauli_indices: np.ndarray, fired_cols: np.ndarray,
                       table: List[stim.PauliString], N: int) -> np.ndarray:
    """Product over fired errors of their Pauli strings -> (N, 4) one-hot."""
    acc = stim.PauliString(N)
    for i in fired_cols:
        acc = acc * table[i]           # Pauli-group product (phase ignored)
    out = np.zeros((N, 4), dtype=np.float32)
    idx = np.array([acc[q] for q in range(N)], dtype=np.int64)  # 0..3
    out[np.arange(N), idx] = 1.0
    return out


# --------------------------------------------------------------------------- #
# 5. Graph construction from a single syndrome
# --------------------------------------------------------------------------- #
def build_graph(fired_detectors: np.ndarray, coords: np.ndarray,
                k: int = 6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Defect-graph: nodes = fired detectors, space-time kNN edges.

    Returns (node_feat (n,4), edge_index (2,E), edge_attr (E,4)).
    Node feature = [x, y, t, is_virtual]. A single virtual node is added so
    empty syndromes still yield a valid 1-node graph (anchor for readout).
    """
    n = len(fired_detectors)
    if n == 0:
        # virtual-only graph
        x = np.array([[0, 0, 0, 1]], dtype=np.float32)
        ei = np.zeros((2, 0), dtype=np.int64)
        ea = np.zeros((0, 4), dtype=np.float32)
        return x, ei, ea

    pos = coords[fired_detectors]                       # (n,3)
    x = np.concatenate([pos, np.zeros((n, 1), np.float32)], axis=1)  # is_virtual=0

    # pairwise space-time distances
    diff = pos[:, None, :] - pos[None, :, :]            # (n,n,3)
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)

    kk = min(k, n - 1) if n > 1 else 0
    src, dst, attr = [], [], []
    for i in range(n):
        if kk > 0:
            nbrs = np.argpartition(dist[i], kk)[:kk]
            for j in nbrs:
                src.append(i); dst.append(int(j))
                attr.append(diff[i, j])                 # (dx,dy,dt)
    if src:
        ei = np.array([src, dst], dtype=np.int64)
        ea = np.array(attr, dtype=np.float32)
        ea = np.concatenate([ea, np.linalg.norm(ea, axis=1, keepdims=True)], axis=1)  # +|d|
    else:
        ei = np.zeros((2, 0), dtype=np.int64)
        ea = np.zeros((0, 4), dtype=np.float32)
    return x, ei, ea


# --------------------------------------------------------------------------- #
# 6. Dataset assembly (one sample call = one source of truth)
# --------------------------------------------------------------------------- #
def sample_tuples(circuit: stim.Circuit, tables: FaultTables,
                  sampler: stim.CompiledDemSampler, num_shots: int,
                  k_edges: int = 6, seed: Optional[int] = None):
    """Yields dicts of numpy arrays. Convert to torch_geometric.data.Data downstream."""
    # ---- THE single source of truth ----
    dets, obs, errs = sampler.sample(
        shots=num_shots, return_errors=True,
        recorded_errors_to_replay=None,
    )
    N = tables.num_qubits
    for s in range(num_shots):
        fired_cols = np.where(errs[s])[0]
        fired_dets = np.where(dets[s])[0]

        raw_t = _accumulate_onehot(None, fired_cols, tables.raw_pauli, N)
        zx_t = _accumulate_onehot(None, fired_cols, tables.zx_pauli, N)
        x, ei, ea = build_graph(fired_dets, tables.detector_coords, k=k_edges)
        y = int(obs[s, 0])

        yield {
            "x": x, "edge_index": ei, "edge_attr": ea,
            "y": y, "raw_target": raw_t, "zx_target": zx_t,
        }


# --------------------------------------------------------------------------- #
# 7. PyG wrapper
# --------------------------------------------------------------------------- #
def to_pyg_list(tuples_iter):
    """Convert sample dicts to a list[torch_geometric.data.Data]."""
    import torch
    from torch_geometric.data import Data
    data_list = []
    for t in tuples_iter:
        data_list.append(Data(
            x=torch.from_numpy(t["x"]),
            edge_index=torch.from_numpy(t["edge_index"]),
            edge_attr=torch.from_numpy(t["edge_attr"]),
            y=torch.tensor([t["y"]], dtype=torch.long),
            # graph-level fixed-size targets -> shape (1, N, 4); batches to (B,N,4)
            raw_target=torch.from_numpy(t["raw_target"]).unsqueeze(0),
            zx_target=torch.from_numpy(t["zx_target"]).unsqueeze(0),
        ))
    return data_list


def make_dataloader(d: int, p: float, num_shots: int, batch_size: int = 256,
                    rounds: Optional[int] = None, k_edges: int = 6,
                    propagator: Optional[ZXPropagator] = None, shuffle: bool = True):
    """End-to-end: build circuit -> tables -> sample -> PyG DataLoader.

    Pass propagator=SpiderTraceAdapter(circuit) to use your engine instead of
    the reference. The same loader serves all three models:
        GNN-A   ignores raw_target and zx_target (trains on y only)
        GNN-Raw uses raw_target as the auxiliary head's target
        GNN-ZX  uses zx_target  as the auxiliary head's target
    """
    from torch_geometric.loader import DataLoader
    circ = build_circuit(d, p, rounds=rounds)
    tables, sampler = build_fault_tables(circ, propagator=propagator)
    tuples = sample_tuples(circ, tables, sampler, num_shots, k_edges=k_edges)
    data_list = to_pyg_list(tuples)
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle), tables


# --------------------------------------------------------------------------- #
# 7b. SpiderTrace validation against the reference oracle
# --------------------------------------------------------------------------- #
def _pauli_indices(ps: stim.PauliString, N: int) -> List[int]:
    """Per-qubit Pauli indices (0..3), ignoring the overall sign."""
    return [ps[q] for q in range(N)]


def _data_qubits(circuit: stim.Circuit) -> List[int]:
    """Data qubit indices = targets of the final (destructive) M instruction."""
    for inst in reversed(list(circuit.without_noise().flattened())):
        if inst.name == "M":
            return [t.qubit_value for t in inst.targets_copy() if t.is_qubit_target]
    return []


def validate_adapter(d: int = 3, p: float = 0.02, n_trials: int = 50,
                     seed: int = 0) -> Tuple[int, int]:
    """Compare SpiderTraceAdapter to ReferenceZXPropagator on random single-qubit
    data faults across random tick offsets (clean layer boundaries, where the
    two MUST agree). Prints a match count and any mismatch details."""
    circ = build_circuit(d, p)
    ref = ReferenceZXPropagator(circ)
    adapter = SpiderTraceAdapter(circ)
    N = circ.num_qubits
    data = _data_qubits(circ)
    rng = np.random.default_rng(seed)

    matches = 0
    mismatches = []
    for _ in range(n_trials):
        q = int(rng.choice(data))
        pl = int(rng.integers(1, 4))                 # 1=X, 2=Y, 3=Z
        tick = int(rng.integers(0, adapter.num_ticks + 1))
        r = ref.propagate([q], [pl], tick)
        a = adapter.propagate([q], [pl], tick)
        if _pauli_indices(r, N) == _pauli_indices(a, N):
            matches += 1
        else:
            mismatches.append((q, PAULI_CHAR[pl], tick, r, a))

    print(f"SpiderTrace adapter validation: {matches}/{n_trials} match")
    for q, plc, tick, r, a in mismatches:
        diff = {i: (PAULI_CHAR[r[i]], PAULI_CHAR[a[i]])
                for i in range(N) if r[i] != a[i]}
        print(f"  MISMATCH fault={plc} on q={q} @tick={tick}: "
              f"(ref, adapter) differ at {diff}")
    return matches, n_trials


def validate_divergence_rate(d: int = 3, p: float = 0.02, shots: int = 2000,
                             target: float = 0.856, tol: float = 0.05,
                             seed: int = 0) -> float:
    """Re-run the smoke-test divergence metric with the SpiderTrace adapter and
    confirm it stays within ``tol`` of the reference baseline ``target``."""
    circ = build_circuit(d, p)
    tables, sampler = build_fault_tables(circ, propagator=SpiderTraceAdapter(circ))
    tuples = list(sample_tuples(circ, tables, sampler, shots, seed=seed))
    differ = float(np.mean(
        [not np.array_equal(t["raw_target"], t["zx_target"]) for t in tuples]))
    ok = abs(differ - target) <= tol
    print(f"SpiderTrace divergence rate (raw != zx): {differ:.3f}  "
          f"(reference baseline {target:.3f}, tol {tol:.2f}) -> "
          f"{'OK' if ok else 'OUT OF RANGE'}")
    return differ


# --------------------------------------------------------------------------- #
# 8. Smoke test (core pipeline, no torch needed)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    d, p, shots = 3, 0.02, 2000
    circ = build_circuit(d, p)
    tables, sampler = build_fault_tables(circ)
    print(f"d={d} p={p}: qubits={tables.num_qubits} detectors={tables.num_detectors} "
          f"dem_errors={tables.num_errors}")

    tuples = list(sample_tuples(circ, tables, sampler, shots))
    ys = np.array([t["y"] for t in tuples])
    raw_nz = np.mean([t["raw_target"][:, 0].sum() < tables.num_qubits for t in tuples])
    zx_nz = np.mean([t["zx_target"][:, 0].sum() < tables.num_qubits for t in tuples])
    avg_nodes = np.mean([len(t["x"]) for t in tuples])
    # how often the two targets actually differ (sanity: propagation does something)
    differ = np.mean([not np.array_equal(t["raw_target"], t["zx_target"]) for t in tuples])
    print(f"logical-flip rate: {ys.mean():.4f}")
    print(f"shots with non-I raw target: {raw_nz:.3f}, zx target: {zx_nz:.3f}")
    print(f"avg graph nodes: {avg_nodes:.2f}")
    print(f"shots where raw_target != zx_target: {differ:.3f}  (propagation is active)")
    print("OK: matched (graph, raw_target, zx_target, logical_label) tuples produced.")

    # ---- SpiderTrace adapter validations ----
    print("\n--- SpiderTrace adapter validation ---")
    validate_adapter(d=d, p=p, n_trials=50, seed=0)
    validate_divergence_rate(d=d, p=p, shots=shots, target=0.856, tol=0.05)