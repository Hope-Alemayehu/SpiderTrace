"""Consistency checks for the auxiliary targets built by qec_zx_dataset.py.

A correctly propagated end-of-circuit Pauli frame must reproduce the logical
label: in the memory-Z circuit the observable is the parity of the final Z-basis
measurements of a set of data qubits, so the label equals the parity of X or Y
components of the final frame on those qubits. raw_target is NOT a propagated
frame, so its match rate is reported for comparison only and never asserted.

Known failure (2026-10-06, see results/LOG.md): ReferenceZXPropagator injects
each fault at the START of its tick layer (qec_zx_dataset.py:106-117), but the
noise instruction comes AFTER the gate, reset or measurement it models in that
layer. The tests that assert ZX consistency are therefore marked xfail(strict),
so they start failing (XPASS) once the propagator is fixed and the markers must
be removed.

Run as a script for the full report:
    python tests/test_zx_target_consistency.py
"""
import os
import sys
from functools import lru_cache

import numpy as np
import pytest
import stim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qec_zx_dataset import build_circuit, build_fault_tables, sample_tuples  # noqa: E402

SAMPLED_CONFIGS = [(3, 0.003), (5, 0.003)]
N_SHOTS = 5000
SAMPLER_SEED = 12345
KNOWN_BUG = ("ReferenceZXPropagator injects faults at the start of their tick layer, "
             "before the gate/reset/measurement they follow (results/LOG.md, 2026-10-06)")


# ---- helpers ------------------------------------------------------------------------
@lru_cache(maxsize=None)
def setup(d, p):
    """Circuit, fault tables (built exactly as the training pipeline builds them),
    observable qubits, and the non-decomposed DEM whose error order the tables use."""
    circuit = build_circuit(d, p)
    tables, _unseeded = build_fault_tables(circuit)
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    return circuit, tables, observable_qubits(circuit), dem


def observable_qubits(circuit):
    """Qubits whose final measurement records enter OBSERVABLE_INCLUDE(0)."""
    measured, obs = [], set()
    for inst in circuit.flattened():
        if stim.gate_data(inst.name).produces_measurements:
            measured += [t.value for t in inst.targets_copy()]
        if inst.name == "OBSERVABLE_INCLUDE":
            for t in inst.targets_copy():
                obs.add(measured[len(measured) + t.value])
    return tuple(sorted(obs))


def x_parity_onehot(onehot, obs):
    """Parity of X or Y on the observable qubits of an (N, 4) one-hot I/X/Y/Z target."""
    idx = onehot.argmax(axis=1)
    return int(np.isin(idx[list(obs)], (1, 2)).sum() % 2)


def x_parity_pauli(ps, obs):
    return sum(1 for q in obs if ps[q] in (1, 2)) % 2


def sampled_match_rates(d, p, shots=N_SHOTS, seed=SAMPLER_SEED):
    """Sample shots with the repo's pipeline and compare each target's X-parity on
    the observable qubits with the logical label.

    build_fault_tables() returns an unseeded sampler (README, Known issues), so this
    compiles a seeded sampler from the identical DEM, as qec_run.make_datasets does.
    """
    circuit, tables, obs, dem = setup(d, p)
    assert dem.num_errors == tables.num_errors
    sampler = dem.compile_sampler(seed=seed)
    n = zx_ok = raw_ok = flips = 0
    for t in sample_tuples(circuit, tables, sampler, shots):
        zx_ok += x_parity_onehot(t["zx_target"], obs) == t["y"]
        raw_ok += x_parity_onehot(t["raw_target"], obs) == t["y"]
        flips += t["y"]
        n += 1
    return {"d": d, "p": p, "shots": n, "zx": zx_ok / n, "raw": raw_ok / n,
            "flip_rate": flips / n}


def dem_errors(dem):
    """(probability, detector set, observable flip) for every DEM error, in order."""
    out = []
    for inst in dem.flattened():
        if inst.type == "error":
            dets = frozenset(t.val for t in inst.targets_copy() if t.is_relative_detector_id())
            obs = sum(1 for t in inst.targets_copy() if t.is_logical_observable_id()) % 2
            out.append((inst.args_copy()[0], dets, obs))
    return out


def flat_instructions(circuit):
    """Flatten REPEAT blocks, keyed by the (instruction_offset, iteration_index)
    stack-frame path that stim reports in CircuitErrorLocation.stack_frames. Each
    frame's iteration_index is the iteration of the loop that CONTAINS it (0 at top
    level), so a REPEAT frame carries its parent's iteration, not its own."""
    flat, index = [], {}

    def walk(block, prefix, enclosing_it):
        for off, item in enumerate(block):
            if isinstance(item, stim.CircuitRepeatBlock):
                body = item.body_copy()
                for it in range(item.repeat_count):
                    walk(body, prefix + ((off, enclosing_it),), it)
            else:
                index[prefix + ((off, enclosing_it),)] = len(flat)
                flat.append(item)

    walk(circuit, (), 0)
    return flat, index


def _noiseless(inst):
    """The instruction without its noise: None for pure noise channels, and noisy
    measurements (e.g. M(p)) lose their flip probability."""
    gd = stim.gate_data(inst.name)
    if gd.is_noisy_gate and not gd.produces_measurements:
        return None
    if gd.produces_measurements and inst.gate_args_copy():
        return stim.CircuitInstruction(inst.name, inst.targets_copy())
    return inst


def _inject(sim, loc):
    for gt in loc.flipped_pauli_product:
        t = gt.gate_target
        sim.set_pauli_flip("XYZ"[(0 if t.is_x_target else 1 if t.is_y_target else 2)],
                           qubit_index=t.qubit_value, instance_index=0)


def _signature(sim):
    dets = frozenset(np.flatnonzero(sim.get_detector_flips()[:, 0]).tolist())
    return dets, int(sim.get_observable_flips()[0, 0])


def propagate_layer_start(circuit, loc):
    """Replicates ReferenceZXPropagator.propagate (qec_zx_dataset.py:101-118): inject
    before the first instruction if tick_offset == 0, else right after the
    tick_offset-th TICK. Also returns the detector/observable signature."""
    sim = stim.FlipSimulator(batch_size=1, disable_stabilizer_randomization=True,
                             num_qubits=circuit.num_qubits)
    ticks, injected = 0, loc.tick_offset == 0
    if injected:
        _inject(sim, loc)
    for inst in circuit.without_noise().flattened():
        sim.do(inst)
        if inst.name == "TICK":
            ticks += 1
            if ticks == loc.tick_offset and not injected:
                _inject(sim, loc)
                injected = True
    return sim.peek_pauli_flips()[0], _signature(sim)


def propagate_at_instruction(circuit, loc, flat=None, index=None):
    """Reference: inject the fault exactly where its noise instruction sits, i.e.
    after every earlier instruction in the same tick layer."""
    if flat is None:
        flat, index = flat_instructions(circuit)
    at = index[tuple((sf.instruction_offset, sf.iteration_index) for sf in loc.stack_frames)]
    assert flat[at].name == loc.instruction_targets.gate
    sim = stim.FlipSimulator(batch_size=1, disable_stabilizer_randomization=True,
                             num_qubits=circuit.num_qubits)
    for i, inst in enumerate(flat):
        if i == at:
            _inject(sim, loc)
            continue
        clean = _noiseless(inst)
        if clean is not None:
            sim.do(clean)
    return sim.peek_pauli_flips()[0], _signature(sim)


def preceding_ops_on_fault_qubits(circuit, loc, flat, index):
    """Non-noise operations in the fault's tick layer that come before its noise
    instruction and act on one of its qubits. Injecting at the start of the layer
    wrongly pushes the fault through exactly these operations."""
    at = index[tuple((sf.instruction_offset, sf.iteration_index) for sf in loc.stack_frames)]
    qubits = {gt.gate_target.qubit_value for gt in loc.flipped_pauli_product}
    start = at
    while start > 0 and flat[start - 1].name != "TICK":
        start -= 1
    ops = []
    for inst in flat[start:at]:
        gd = stim.gate_data(inst.name)
        if gd.is_noisy_gate and not gd.produces_measurements:
            continue
        if gd.is_unitary or gd.is_reset or gd.produces_measurements:
            if qubits & {t.value for t in inst.targets_copy() if t.is_qubit_target}:
                ops.append(inst.name)
    return ops


@lru_cache(maxsize=None)
def dem_walk(d, p):
    """Every DEM error with its label check and both injection signatures."""
    circuit, tables, obs, dem = setup(d, p)
    expl = circuit.explain_detector_error_model_errors(reduce_to_one_representative_error=True)
    errs = dem_errors(dem)
    assert len(expl) == len(errs) == tables.num_errors
    flat, index = flat_instructions(circuit)
    rows = []
    for i, (e, (prob, dets, L)) in enumerate(zip(expl, errs)):
        loc = e.circuit_error_locations[0]
        zx = tables.zx_pauli[i]
        if loc.flipped_pauli_product:
            frame_ls, sig_ls = propagate_layer_start(circuit, loc)
            frame_ex, sig_ex = propagate_at_instruction(circuit, loc, flat, index)
            pre = preceding_ops_on_fault_qubits(circuit, loc, flat, index)
        else:  # no Pauli representative (pure measurement flip): repo stores identity
            frame_ls = frame_ex = stim.PauliString(circuit.num_qubits)
            sig_ls = sig_ex = None
            pre = []
        rows.append({
            "index": i, "prob": prob, "tick_offset": loc.tick_offset,
            "gate": loc.instruction_targets.gate,
            "fault": " ".join(f"{'XYZ'[0 if g.gate_target.is_x_target else 1 if g.gate_target.is_y_target else 2]}"
                              f"{g.gate_target.qubit_value}" for g in loc.flipped_pauli_product),
            "flips_obs": L,
            "zx_label_ok": x_parity_pauli(zx, obs) == L,
            "raw_label_ok": x_parity_pauli(tables.raw_pauli[i], obs) == L,
            "repo_frame_replicated": frame_ls == zx,
            "layer_start_signature_ok": sig_ls is None or sig_ls == (dets, L),
            "exact_signature_ok": sig_ex is None or sig_ex == (dets, L),
            "frame_differs_from_exact": frame_ls != frame_ex,
            "frame_exact": frame_ex,
            "preceding_ops": pre,
        })
    return rows


def shots_with_wrong_zx_target(d, p, shots=N_SHOTS, seed=SAMPLER_SEED):
    """Fraction of sampled shots whose stored zx_target (product of repo frames over
    the fired DEM errors) differs from the product of exact-position frames."""
    circuit, tables, obs, dem = setup(d, p)
    exact = [r["frame_exact"] for r in dem_walk(d, p)]
    _, _, errs = dem.compile_sampler(seed=seed).sample(shots=shots, return_errors=True)
    wrong = 0
    for row in errs:
        repo = stim.PauliString(circuit.num_qubits)
        ref = stim.PauliString(circuit.num_qubits)
        for i in np.flatnonzero(row):
            repo *= tables.zx_pauli[i]
            ref *= exact[i]
        wrong += repo != ref
    return wrong / shots


# ---- tests ----------------------------------------------------------------------------
@pytest.mark.parametrize("d,p", SAMPLED_CONFIGS)
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=KNOWN_BUG)
def test_sampled_zx_targets_match_label(d, p):
    r = sampled_match_rates(d, p)
    print(f"d={d} p={p}: ZX match {r['zx']:.4f}, Raw match {r['raw']:.4f} over {r['shots']} shots")
    assert r["zx"] == 1.0


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=KNOWN_BUG)
def test_dem_errors_zx_frames_match_label_d3():
    bad = [r for r in dem_walk(3, 0.003) if not r["zx_label_ok"]]
    assert not bad, f"{len(bad)} DEM errors have label-inconsistent ZX frames"


def test_layer_start_replica_matches_repo_frames():
    """The diagnosis is about the repo's own code: the replica reproduces every
    stored zx_target frame."""
    rows = dem_walk(3, 0.003)
    assert all(r["repo_frame_replicated"] for r in rows)


def test_exact_position_injection_reproduces_every_dem_signature():
    """The reference used for the diagnosis is right: injecting at the noise
    instruction reproduces each DEM error's detectors and observable flip."""
    rows = dem_walk(3, 0.003)
    assert all(r["exact_signature_ok"] for r in rows)


def test_every_wrong_frame_follows_a_same_layer_operation():
    """Every frame that differs from the exact reference belongs to a fault whose
    tick layer has a gate, reset or measurement on its qubits before the noise.
    (The converse does not hold: in this circuit every noise instruction follows
    such an operation, and the frame survives whenever the operation commutes
    with the fault.)"""
    rows = dem_walk(3, 0.003)
    wrong = [r for r in rows if r["frame_differs_from_exact"]]
    assert wrong and all(r["preceding_ops"] for r in wrong)


# ---- report -----------------------------------------------------------------------------
def report():
    print("Sampled shots (seeded sampler, seed 12345): X-parity of target on observable "
          "qubits vs logical label")
    for d, p in SAMPLED_CONFIGS + [(3, 0.01)]:
        r = sampled_match_rates(d, p)
        print(f"  d={d} p={p}: shots={r['shots']}  ZX match={r['zx']:.4f}  "
              f"Raw match={r['raw']:.4f}  (label=1 rate {r['flip_rate']:.4f})")

    print("\nSampled shots whose stored zx_target differs from the exact-position target")
    for d, p in SAMPLED_CONFIGS + [(3, 0.01)]:
        print(f"  d={d} p={p}: {shots_with_wrong_zx_target(d, p):.4f} of {N_SHOTS} shots")

    for d, p in ((3, 0.003), (5, 0.003)):
        rows = dem_walk(d, p)
        bad = [r for r in rows if not r["zx_label_ok"]]
        wrong = [r for r in rows if r["frame_differs_from_exact"]]
        sig_bad = [r for r in rows if not r["layer_start_signature_ok"]]
        print(f"\nDEM walk d={d} p={p}: {len(rows)} errors")
        print(f"  label-inconsistent ZX frames: {len(bad)} (total prob {sum(r['prob'] for r in bad):.5f})")
        print(f"  ZX frame differs from exact-position injection: {len(wrong)} "
              f"(total prob {sum(r['prob'] for r in wrong):.5f})")
        print(f"  layer-start injection gives wrong detector/observable signature: {len(sig_bad)}")
        print(f"  exact-position injection reproduces every signature: "
              f"{all(r['exact_signature_ok'] for r in rows)}")
        causes = {}
        for r in wrong:
            key = f"tick {'0' if r['tick_offset'] == 0 else '>0'}, after {'+'.join(sorted(set(r['preceding_ops'])))}"
            causes[key] = causes.get(key, 0) + 1
        print(f"  wrong frames by cause: {causes}")
        if d == 3:
            print("  label-inconsistent ZX frames (d=3):")
            print(f"    {'idx':>4} {'prob':>9} {'tick':>4} {'gate':<11} {'fault':<10} {'L':>1}  preceded by")
            for r in sorted(bad, key=lambda r: -r["prob"]):
                print(f"    {r['index']:>4} {r['prob']:>9.6f} {r['tick_offset']:>4} {r['gate']:<11} "
                      f"{r['fault']:<10} {r['flips_obs']:>1}  {', '.join(r['preceding_ops'])}")


if __name__ == "__main__":
    report()
