"""
validate_dem_graph.py
=====================
MWPM-equivalent sanity check for the DEM-derived decoding graph.

Trains GNN-A on the new graph at d=5, p=0.003, 50k shots, 100 epochs, at BOTH 4
and 8 message-passing layers (same dataset for a fair comparison), and reports
LER vs the MWPM baseline (0.0036). Target: LER < 0.012 (within ~2-3x of MWPM).

Uses natural-distribution sampling (balanced_sampler=False) + pos_weight=1.0,
which is LER-optimal at this data scale (50k shots does not collapse, and
balanced sampling would shift the prior and over-predict, inflating LER).
"""

import json
from pathlib import Path

import numpy as np
import torch

import gnn_models
from gnn_models import build_loaders, _train_one

MWPM_LER = 0.0036
TARGET_LER = 0.012


def main():
    d, p, shots, seed, epochs = 5, 0.003, 50000, 1, 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Building dataset: d={d} p={p} shots={shots} seed={seed} (natural sampling)")
    train_loader, val_loader, num_qubits = build_loaders(
        d, p, shots, seed, batch_size=256, balanced_sampler=False)

    results = {}
    for layers in (4, 8):
        gnn_models.NUM_MP_LAYERS = layers
        print(f"\n############ GNN-A | {layers} MP layers ############")
        res = _train_one("A", num_qubits, train_loader, val_loader,
                         epochs, 0.0, 1.0, device)   # lambda_aux=0, pos_weight=1.0
        m = res["metrics"]
        results[str(layers)] = m
        print(f">>> layers={layers}: LER={m['logical_error_rate']:.4f}  "
              f"recall={m['recall']:.3f}  precision={m['precision']:.3f}")

    print("\n================ SUMMARY ================")
    print(f"MWPM baseline LER = {MWPM_LER}   target LER < {TARGET_LER}")
    for layers, m in results.items():
        ler = m["logical_error_rate"]
        verdict = "PASS (<target)" if ler < TARGET_LER else \
                  f"{ler / MWPM_LER:.1f}x MWPM"
        print(f"  GNN-A {layers} layers: LER={ler:.4f}  -> {verdict}")

    Path("results").mkdir(exist_ok=True)
    out = Path("results") / "dem_graph_gnnA_d5_p0.003.json"
    with open(out, "w") as f:
        json.dump({"config": {"d": d, "p": p, "shots": shots, "seed": seed,
                              "epochs": epochs, "sampler": "natural",
                              "pos_weight": 1.0},
                   "mwpm_ler": MWPM_LER, "target_ler": TARGET_LER,
                   "by_layers": results}, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
