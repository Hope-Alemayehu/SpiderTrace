"""
evaluate_mwpm_circuit.py
========================
MWPM baseline under CIRCUIT-LEVEL depolarizing noise, sampled directly from
Stim -- no .npz files. Uses the SAME circuit/noise model as the GNN training
pipeline (qec_zx_dataset.build_circuit), so the LER here is the apples-to-apples
classical-decoder baseline the GNNs are measured against.

For each (d, p): build the noisy circuit, sample detectors + observables, decode
every syndrome with PyMatching (from the matched detector error model), and
report logical error rate, recall, precision.

CLI:
    python evaluate_mwpm_circuit.py
    python evaluate_mwpm_circuit.py --d 3 5 --p 0.003 0.005 0.007 0.01 --shots 50000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pymatching
import stim

from qec_zx_dataset import build_circuit


def evaluate(d: int, p: float, num_shots: int) -> dict:
    """Sample `num_shots` shots from the circuit-level-noise circuit at (d, p)
    and decode with MWPM. Returns LER / recall / precision for the logical flip."""
    circuit = build_circuit(d, p)

    # Match decoder to the exact noise model (decompose_errors -> graphlike DEM).
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)

    sampler = circuit.compile_detector_sampler()
    detectors, observables = sampler.sample(
        shots=num_shots, separate_observables=True)

    # Batched MWPM decode -> predicted observable flips, shape (shots, num_obs).
    preds = matching.decode_batch(detectors)

    pred = preds[:, 0].astype(int)
    true = observables[:, 0].astype(int)

    n = len(true)
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())

    return {
        "logical_error_rate": float((pred != true).mean()),
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "num_shots": int(n),
        "observable_flip_rate": float(true.mean()),
    }


def main():
    ap = argparse.ArgumentParser(
        description="MWPM baseline under circuit-level noise (direct from Stim)")
    ap.add_argument("--d", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--p", type=float, nargs="+",
                    default=[0.003, 0.005, 0.007, 0.01])
    ap.add_argument("--shots", type=int, default=50000)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    results = {}
    for d in args.d:
        results[str(d)] = {}
        for p in args.p:
            m = evaluate(d, p, args.shots)
            results[str(d)][str(p)] = m
            print(f"d={d} p={p:<6} LER={m['logical_error_rate']:.4f}  "
                  f"recall={m['recall']:.4f}  precision={m['precision']:.4f}  "
                  f"(flip_rate={m['observable_flip_rate']:.4f}, "
                  f"shots={m['num_shots']})")

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.results_dir) / "mwpm_circuit_level.json"
    with open(out, "w") as f:
        json.dump({"config": {"shots": args.shots}, "results": results},
                  f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
