import stim
import pymatching
import numpy as np
import json
from pathlib import Path


def evaluate_mwpm(d, data_dir="data", results_dir="results"):
    files = sorted(Path(data_dir).glob(f"d{d}_p*.npz"))
    results = {}

    for f in files:
        p = float(f.stem.split("_p")[1])
        # Build matching from the noisy circuit at the same p used to generate syndromes
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=d,
            rounds=2,
            before_round_data_depolarization=p,
        )
        matching = pymatching.Matching.from_stim_circuit(circuit)
        data = np.load(f)
        n = len(data["syndrome_bits"])
        n_test = n - int(0.8 * n) - int(0.1 * n)
        syndromes = data["syndrome_bits"][-n_test:].astype(np.uint8)
        flips_true = data["logical_flip"][-n_test:].astype(int)

        preds = []
        for i in range(n_test):
            try:
                pred = int(matching.decode(syndromes[i])[0])
            except Exception:
                pred = 0
            preds.append(pred)

        preds = np.array(preds)
        n_total = len(flips_true)
        tp = int(((preds == 1) & (flips_true == 1)).sum())
        fp = int(((preds == 1) & (flips_true == 0)).sum())
        fn = int(((preds == 0) & (flips_true == 1)).sum())

        results[str(p)] = {
            "logical_error_rate": float((fp + fn) / n_total),
            "recall":    float(tp / (tp + fn + 1e-8)),
            "precision": float(tp / (tp + fp + 1e-8)),
        }
        print(f"d={d} p={p}: LER={results[str(p)]['logical_error_rate']:.4f}")

    Path(results_dir).mkdir(exist_ok=True)
    out = Path(results_dir) / f"d{d}_mwpm.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    distances = [args.d] if args.d else [3, 5, 7]
    for d in distances:
        evaluate_mwpm(d, args.data_dir, args.results_dir)
