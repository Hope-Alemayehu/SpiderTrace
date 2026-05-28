#!/usr/bin/env python3
"""
Knowledge distillation dataset generator for QEC decoder training.

Generates (syndrome_bits, zx_features, logical_flip) triplets by:
  1. Sampling syndromes from stim's DEM sampler (rounds=2 noisy circuit)
  2. Mapping fired DEM mechanisms to data-qubit fault locations
  3. Propagating fault locations through SpiderTrace for ZX features
  4. Saving .npz files per (distance, error_rate) combination

Usage:
  python generate_dataset.py --dry-run        # 50 shots, d=3, p=0.1
  python generate_dataset.py                  # full generation -> data/
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import stim

from spidertrace.circuit import Gate
from spidertrace.engine import propagate_errors
from spidertrace.error import PauliError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

PAULI_TO_INT = {"I": 0, "X": 1, "Z": 2, "Y": 3}


# ─── Step 1: Surface code circuit helpers ────────────────────────────────────

def get_data_qubits(circuit):
    """Extract data qubit indices from the final M instruction."""
    for instr in reversed(list(circuit)):
        if hasattr(instr, 'name') and instr.name == 'M':
            return [t.qubit_value for t in instr.targets_copy() if t.is_qubit_target]
    return []


def _noiseless_circuit(d: int) -> stim.Circuit:
    return stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=1)


# ─── Step 2: Per-shot sampling via DEM ───────────────────────────────────────

def sample_shots(d, p, n_shots):
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=2,
        before_round_data_depolarization=p,
    )
    data_qubits = get_data_qubits(circuit)
    qubit_to_idx = {q: i for i, q in enumerate(data_qubits)}

    dem = circuit.detector_error_model(decompose_errors=True)

    # Build fault map: dem mechanism index -> list of (qubit, pauli) for data qubits only
    fault_map = []
    try:
        explanations = circuit.explain_detector_error_model_errors(
            dem_filter=dem,
            reduce_to_one_representative_error=True,
        )
        for expl in explanations:
            faults = []
            for loc in expl.circuit_error_locations:
                for pt in loc.flipped_pauli_product:
                    q = pt.gate_target.qubit_value
                    if q in qubit_to_idx:
                        if pt.gate_target.is_x_target:
                            faults.append((q, 'X'))
                        elif pt.gate_target.is_y_target:
                            faults.append((q, 'Y'))
                        elif pt.gate_target.is_z_target:
                            faults.append((q, 'Z'))
            fault_map.append(faults)
    except Exception as e:
        print(f"explain_errors failed: {e}")
        fault_map = [[] for _ in range(len(list(dem.flattened())))]

    sampler = dem.compile_sampler()
    det_data, obs_data, err_data = sampler.sample(
        shots=n_shots, bit_packed=False, return_errors=True
    )

    results = []
    for i in range(n_shots):
        fired = np.flatnonzero(err_data[i])
        seen = {}
        for idx in fired:
            if idx < len(fault_map):
                for q, pauli in fault_map[idx]:
                    seen[q] = pauli  # last write wins per qubit
        fault_locations = list(seen.items())
        results.append((
            det_data[i].astype(np.uint8),
            fault_locations,
            int(obs_data[i, 0]),
        ))
    return results, data_qubits


# ─── Step 3: ZX feature vectors ──────────────────────────────────────────────

def _stim_to_spider_gates(stim_circuit: stim.Circuit) -> list:
    """Extract H / CNOT / CZ instructions from a Stim circuit as SpiderTrace Gates."""
    gates = []
    for instr in stim_circuit:
        if not isinstance(instr, stim.CircuitInstruction):
            continue
        name = instr.name
        qubits = [t.qubit_value for t in instr.targets_copy() if t.is_qubit_target]
        if name == "H":
            gates.extend(Gate("H", (q,)) for q in qubits)
        elif name in ("CNOT", "CX"):
            for i in range(0, len(qubits) - 1, 2):
                gates.append(Gate("CNOT", (qubits[i], qubits[i + 1])))
        elif name == "CZ":
            for i in range(0, len(qubits) - 1, 2):
                gates.append(Gate("CZ", (qubits[i], qubits[i + 1])))
    return gates


def get_zx_features(
    fault_locations: list,
    stim_circuit: stim.Circuit,
    data_qubits: list,
) -> np.ndarray:
    """
    Run SpiderTrace on the given fault locations and return a fixed-length ZX
    feature vector of shape (len(data_qubits),) encoding the final error state.

    Encoding: 0=I  1=X  2=Z  3=Y
    Returns a zero vector if fault_locations is empty or propagation yields nothing.
    """
    qubit_to_idx = {q: i for i, q in enumerate(data_qubits)}
    vec = np.zeros(len(data_qubits), dtype=np.uint8)

    if not fault_locations:
        return vec

    errors = [PauliError(q, pl) for q, pl in fault_locations]
    gates = _stim_to_spider_gates(stim_circuit)
    trace = propagate_errors(gates, errors)

    if trace:
        for qubit, pauli in trace[-1].errors_after.items():
            if qubit in qubit_to_idx:
                vec[qubit_to_idx[qubit]] = PAULI_TO_INT.get(pauli, 0)

    return vec


# ─── Step 4: Dataset generation ──────────────────────────────────────────────

def generate_dataset(d: int, p: float, n_shots: int, output_dir: str):
    """
    Generate training triplets for a distance-d code at error rate p and save
    to output_dir/d{d}_p{p:.4f}.npz with arrays:
      syndrome_bits : uint8 (n_shots, n_detectors)
      zx_features   : uint8 (n_shots, d*d)
      logical_flip  : uint8 (n_shots,)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    noiseless = _noiseless_circuit(d)
    spider_gates = _stim_to_spider_gates(noiseless)

    log.info("d=%d p=%.4f  sampling %d shots ...", d, p, n_shots)
    shots, data_qubits = sample_shots(d, p, n_shots)
    qubit_to_idx = {q: i for i, q in enumerate(data_qubits)}
    n_data = len(data_qubits)
    n_detectors = shots[0][0].shape[0]

    syndrome_arr = np.zeros((n_shots, n_detectors), dtype=np.uint8)
    zx_arr = np.zeros((n_shots, n_data), dtype=np.uint8)
    logical_arr = np.zeros(n_shots, dtype=np.uint8)

    non_trivial = skipped = 0

    for i, (syndrome_bits, fault_locations, logical_flip) in enumerate(shots):
        syndrome_arr[i] = syndrome_bits
        logical_arr[i] = logical_flip

        if fault_locations:
            non_trivial += 1
            try:
                errors = [PauliError(q, pl) for q, pl in fault_locations]
                trace = propagate_errors(spider_gates, errors)
                if trace:
                    for qubit, pauli in trace[-1].errors_after.items():
                        if qubit in qubit_to_idx:
                            zx_arr[i, qubit_to_idx[qubit]] = PAULI_TO_INT.get(pauli, 0)
            except Exception as exc:
                log.warning("Shot %d SpiderTrace error: %s", i, exc)
                skipped += 1

        if (i + 1) % 5000 == 0:
            log.info(
                "  d=%d p=%.4f  %d/%d shots | non-trivial=%d skipped=%d",
                d, p, i + 1, n_shots, non_trivial, skipped,
            )

    log.info(
        "d=%d p=%.4f  done — shots=%d  non-trivial=%d  skipped=%d",
        d, p, n_shots, non_trivial, skipped,
    )

    out_path = out_dir / f"d{d}_p{p:.4f}.npz"
    np.savez(out_path, syndrome_bits=syndrome_arr, zx_features=zx_arr, logical_flip=logical_arr)
    log.info("Saved %s", out_path)


# ─── Dry run ─────────────────────────────────────────────────────────────────

def dry_run():
    """50 shots at d=3, p=0.1 — confirms syndrome_bits and zx_features are both non-zero in the same shot."""
    d, p, n = 3, 0.1, 50
    log.info("Dry run: d=%d p=%.2f n=%d", d, p, n)

    noiseless = _noiseless_circuit(d)
    shots, data_qubits = sample_shots(d, p, n)

    print(f"\nCircuit stats (rounds=2 noisy):")
    print(f"  detectors  : {shots[0][0].shape[0]}")
    print(f"  data qubits: {data_qubits}")

    non_trivial = sum(1 for _, fl, _ in shots if fl)
    print(f"\nShots with fault_locations: {non_trivial}/{n}")

    # Print first 3 shots where BOTH syndrome and ZX are non-zero
    printed = 0
    for i, (syndrome_bits, fault_locations, logical_flip) in enumerate(shots):
        if not fault_locations:
            continue
        zx_features = get_zx_features(fault_locations, noiseless, data_qubits)
        both = bool(syndrome_bits.any()) and bool(zx_features.any())
        print(f"\nShot {i}:")
        print(f"  fault_locations : {fault_locations}")
        print(f"  syndrome_bits   : {syndrome_bits.tolist()}")
        print(f"  zx_features     : {zx_features.tolist()}")
        print(f"  logical_flip    : {logical_flip}")
        print(f"  syndrome non-zero: {bool(syndrome_bits.any())}  |  zx non-zero: {bool(zx_features.any())}  |  BOTH: {both}")
        printed += 1
        if printed >= 3:
            break

    if printed == 0:
        print("\n(no shots with fault_locations — try a higher p or more shots)")


# ─── Step 5: Entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpiderTrace QEC dataset generator")
    parser.add_argument("--dry-run", action="store_true",
                        help="50 shots at d=3 p=0.1, prints first 3 shots with both non-zero")
    parser.add_argument("--d", type=int, default=None, help="Code distance (single run)")
    parser.add_argument("--p", type=float, default=None, help="Error rate (single run)")
    parser.add_argument("--shots", type=int, default=None, help="Number of shots (single run)")
    parser.add_argument("--only-d", type=int, default=None,
                        help="Full sweep for one distance only (all 6 error rates)")
    parser.add_argument("--shots-override", type=int, default=None,
                        help="Override default shot count when used with --only-d")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        raise SystemExit(0)

    # Single (d, p, shots) run
    if args.d is not None or args.p is not None or args.shots is not None:
        if None in (args.d, args.p, args.shots):
            parser.error("--d, --p, and --shots must all be provided together")
        generate_dataset(args.d, args.p, args.shots, args.output_dir)
        raise SystemExit(0)

    ERROR_RATES = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]
    SHOTS_FOR_D = {3: 50000, 5: 50000, 7: 20000}

    # Single-distance sweep (--only-d)
    if args.only_d is not None:
        if args.only_d not in SHOTS_FOR_D:
            parser.error(f"--only-d must be one of {list(SHOTS_FOR_D)}")
        n = args.shots_override if args.shots_override is not None else SHOTS_FOR_D[args.only_d]
        for err_rate in ERROR_RATES:
            generate_dataset(args.only_d, err_rate, n, args.output_dir)
        raise SystemExit(0)

    # Full sweep
    for d in (3, 5, 7):
        for err_rate in ERROR_RATES:
            generate_dataset(d, err_rate, SHOTS_FOR_D[d], args.output_dir)
