#!/usr/bin/env python3
"""
qec_run.py -- fair, provenance-recording experiment runner for the ZX-supervised
GNN decoder comparison (GNN-A vs GNN-Raw vs GNN-ZX).

WHY THIS FILE EXISTS (the contract it enforces):
  1. No number is ever produced without its provenance written next to it
     (git commit + dirty flag, full config, seeds, library versions, timestamp).
  2. All three arms are treated IDENTICALLY. Option A: Raw and ZX sweep the same
     lambda grid and each picks best-by-validation; GNN-A collapses to a single
     point because it has no aux head. No arm gets a search advantage.
  3. lambda is selected on the VALIDATION split. The reported LER is on a held-out
     TEST split that is NEVER used for selection or early stopping.
  4. The test split is fixed by --split-seed, INDEPENDENT of the model seed, so
     every arm and every seed sees the identical test set. This is what makes the
     comparison fair and makes McNemar a valid paired test.
  5. McNemar (ZX vs Raw) runs on that shared test set.

  >>> The ONLY repo-dependent part is the ADAPTER SECTION below (4 functions).
      Everything else is complete and codebase-independent. I wrote the adapters
      from the audit description, NOT your real source -- treat them as guesses and
      confirm each call against your actual gnn_models.py / qec_zx_dataset.py.
      A Claude Code prompt to do exactly that is provided separately.
"""

import argparse
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# scipy is optional; used only for the exact McNemar binomial test.
try:
    from scipy.stats import binomtest  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ======================================================================================
# ADAPTER SECTION  --  the only repo-dependent code. Confirm every line against your source.
# ======================================================================================
#
# The harness needs exactly four things from your repo. Contracts are specified precisely;
# the bodies below have been verified line-by-line against gnn_models.py and
# qec_zx_dataset.py at commit 4758764. See ADAPTER NOTES at the bottom of this section
# for the two places where the repo does not natively satisfy the contract.

# qec_zx_dataset.build_fault_tables() is expensive: it calls
# explain_detector_error_model_errors() and propagates EVERY DEM error through a
# FlipSimulator. run_arm() calls make_datasets() once per (arm, seed) with the same
# split_seed, so without caching that cost is paid 3 x len(seeds) times. Caching also
# guarantees the test split is object-identical across arms and seeds, which is what
# makes McNemar's pairing valid.
_DATASET_CACHE = {}
_NUM_QUBITS_CACHE = {}


def _num_qubits_for(d):
    """Qubit count of the rotated surface-code circuit at distance d.

    The models are constructed with `num_qubits` (the aux head is
    Linear(READOUT_DIM, num_qubits * 4)), but the harness only passes `d`. The
    count is independent of p -- noise instructions add no qubits -- so any p
    gives the right answer.
    """
    if d not in _NUM_QUBITS_CACHE:
        from qec_zx_dataset import build_circuit
        _NUM_QUBITS_CACHE[d] = build_circuit(d, 0.001).num_qubits
    return _NUM_QUBITS_CACHE[d]


def _model_type(model):
    """Map a model instance back to its arm key, for train_model's config dict."""
    if not getattr(model, "has_aux", False):
        return "A"
    return "Raw" if model.aux_key == "raw_target" else "ZX"


def make_datasets(d, p, shots, split_seed, val_frac=0.15, test_frac=0.15):
    """Build train / val / test datasets for one (d, p).

    CONTRACT:
      * Returns (train_ds, val_ds, test_ds) as objects your DataLoader can consume
        (from the audit these are PyG graphs built by qec_zx_dataset.py).
      * The split MUST be deterministic in `split_seed` and INDEPENDENT of the model
        seed, so the test set is identical across arms/seeds. Do the split here, not
        inside training.
      * Every graph must carry the fields the models read: syndrome node features,
        the flip label, and BOTH aux targets ("raw_target" and "zx_target") so the
        same dataset feeds all three arms.

    There is no `build_dataset` in qec_zx_dataset.py. The real pipeline is the
    four-step sequence below. `to_pyg_list` emits Data objects carrying x /
    edge_index / edge_attr / y / raw_target / zx_target, so one dataset feeds all
    three arms as the contract requires.
    """
    from qec_zx_dataset import (
        build_circuit, build_fault_tables, sample_tuples, to_pyg_list,
    )

    key = (d, p, shots, split_seed, val_frac, test_frac)
    if key in _DATASET_CACHE:
        return _DATASET_CACHE[key]

    circ = build_circuit(d, p)                  # rotated surface code, rounds=d
    tables, _unseeded = build_fault_tables(circ)
    _NUM_QUBITS_CACHE[d] = tables.num_qubits

    # DETERMINISM FIX (see ADAPTER NOTE 1): build_fault_tables() returns a sampler
    # from `dem.compile_sampler()` with NO seed (qec_zx_dataset.py:240), so its
    # shots come from OS entropy and are not reproducible -- and would differ on
    # every call, breaking the "identical test set across arms/seeds" contract.
    # Rebuild the DEM with the IDENTICAL arguments used there and compile a seeded
    # sampler. Same construction => same error ordering, so sampler error-column i
    # still corresponds to tables.raw_pauli[i] / tables.zx_pauli[i].
    dem = circ.detector_error_model(decompose_errors=False, flatten_loops=True)
    assert dem.num_errors == tables.num_errors, (
        "DEM error-column alignment broken: reseeded sampler does not match "
        f"fault tables ({dem.num_errors} vs {tables.num_errors})."
    )
    sampler = dem.compile_sampler(seed=split_seed)

    full = to_pyg_list(sample_tuples(circ, tables, sampler, shots))

    rng = np.random.default_rng(split_seed)
    idx = rng.permutation(len(full))
    n_test = int(round(test_frac * len(full)))
    n_val = int(round(val_frac * len(full)))
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    take = lambda sel: [full[i] for i in sel]
    out = (take(train_idx), take(val_idx), take(test_idx))
    _DATASET_CACHE[key] = out
    return out


def build_model(kind, d, **hparams):
    """Instantiate one arm. kind in {"A", "Raw", "ZX"}.

    CONTRACT: a fresh, untrained model. GNN-A has no aux head; Raw/ZX share a class
    and differ only by which aux target they regress.

    Class names were correct. Signature is (num_qubits, dropout=0.1) -- NOT (d) --
    so `d` is translated here. gnn_models.MODEL_REGISTRY already maps the arm keys.
    """
    from gnn_models import MODEL_REGISTRY
    if kind not in MODEL_REGISTRY:
        raise ValueError(f"unknown arm {kind!r}")
    return MODEL_REGISTRY[kind](_num_qubits_for(d), **hparams)


def train(model, train_ds, val_ds, lambda_aux, epochs, seed, device):
    """Train one model with early stopping on the VALIDATION split.

    CONTRACT:
      * Seed all RNGs from `seed` at the START (torch, numpy, cuda) so a run is
        reproducible from (seed, config) alone.
      * Early-stop / checkpoint-select on VAL only. Never touch test here.
      * For kind "A", lambda_aux is ignored (no aux head).
      * Returns the trained model (best-val checkpoint restored).

    Real signature is train_model(model, train_loader, val_loader, config: Dict)
    returning (history, best_state) -- a config dict, not kwargs, and no `device`
    parameter (it reads the device off the model's parameters, so the model must be
    moved first). The best-val state_dict is returned, not loaded, so we restore it
    here to satisfy the contract. lambda_aux is correctly ignored for GNN-A:
    compute_loss() gates the aux term on `model.has_aux`.
    """
    import torch
    from torch_geometric.loader import DataLoader   # confirmed: gnn_models.py:44
    from gnn_models import train_model

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # train_model does `device = next(model.parameters()).device` -- move first.
    model = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=hparam_batch_size(), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=hparam_batch_size(), shuffle=False)

    config = {
        "epochs": epochs,
        "lambda_aux": lambda_aux,
        "patience": 10,             # train_model's own default
        "pos_weight": None,         # -> 1.0, the plain-BCE / LER-optimal choice
        "model_type": _model_type(model),   # required: used in train_model's logging
        "balanced_sampler": False,  # logging only; we shuffle at the natural prior
    }
    history, best_state = train_model(model, train_loader, val_loader, config)
    model.load_state_dict(best_state)
    return model


def predict_flips(model, dataset, device):
    """Run the trained model over a dataset and return per-example predictions.

    CONTRACT:
      * Returns (y_pred, y_true) as int numpy arrays of shape (N,), aligned to
        `dataset` order. y_pred is the thresholded logical-flip prediction.
      * Computing predictions HERE (rather than trusting a repo evaluate()) keeps
        the harness in control of the metric and gives McNemar its per-example mask.

    Confirmed: forward(data) -> (flip_logit, aux_logits), aux None for GNN-A, so
    out[0] is right. Label field is `data.y` (long, shape (1,) per graph -> (B,)).
    `logit > 0` is exactly equivalent to the repo's `sigmoid(logit) > 0.5`.
    """
    import torch
    from torch_geometric.loader import DataLoader

    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=hparam_batch_size(), shuffle=False)
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            logit = out[0] if isinstance(out, (tuple, list)) else out  # flip head first
            logit = logit.reshape(-1)
            preds.append((logit > 0).long().cpu().numpy())
            trues.append(batch.y.reshape(-1).long().cpu().numpy())
    return np.concatenate(preds).astype(int), np.concatenate(trues).astype(int)


def hparam_batch_size():
    return 256  # matches build_loaders()' default in gnn_models.py

# --------------------------------------------------------------------------------------
# ADAPTER NOTES -- where the repo does not natively satisfy the contract above.
#
# 1. Dataset sampling was NOT reproducible. qec_zx_dataset.sample_tuples() accepts a
#    `seed` argument and never uses it; the shots come from the sampler built at
#    qec_zx_dataset.py:240 by `dem.compile_sampler()` with no seed. gnn_models'
#    build_loaders() sets torch/numpy seeds before calling it, but stim's sampler has
#    its own RNG and ignores both -- so every call drew different shots and the
#    per-seed "reproducible" runs in gnn_models were not reproducible. make_datasets()
#    above works around this by compiling a seeded sampler itself. The underlying bug
#    is still in qec_zx_dataset.py and affects anything else calling that pipeline.
#
# 2. Early stopping is on val loss, which for Raw/ZX INCLUDES the lambda*aux term
#    (gnn_models.compute_loss). The stopping objective therefore changes with lambda,
#    so runs at different lambda are not stopped on a common scale. Lambda selection
#    in this harness uses val LER, which is unaffected, but the per-run checkpoint
#    choice is not lambda-neutral. Changing it would require editing train_model.
# --------------------------------------------------------------------------------------

# ======================================================================================
# END ADAPTER SECTION.  Nothing below here is repo-dependent.
# ======================================================================================


def logical_error_rate(y_pred, y_true):
    """LER = fraction of shots where the predicted flip disagrees with truth."""
    y_pred = np.asarray(y_pred).astype(int)
    y_true = np.asarray(y_true).astype(int)
    return float(np.mean(y_pred != y_true))


def mcnemar(correct_a, correct_b):
    """Paired McNemar test. correct_a/correct_b are boolean arrays over the SAME
    test examples. Returns the 2x2 discordant counts and a two-sided p-value
    (exact binomial when scipy is present, normal approx otherwise)."""
    a = np.asarray(correct_a).astype(bool)
    b = np.asarray(correct_b).astype(bool)
    assert a.shape == b.shape, "McNemar requires aligned per-example results"
    b01 = int(np.sum(a & ~b))   # A right, B wrong
    b10 = int(np.sum(~a & b))   # A wrong, B right
    n = b01 + b10
    if n == 0:
        p = 1.0
    elif _HAVE_SCIPY:
        p = float(binomtest(min(b01, b10), n, 0.5, alternative="two-sided").pvalue)
    else:
        # continuity-corrected normal approximation
        stat = (abs(b01 - b10) - 1) ** 2 / n if n > 0 else 0.0
        # survival of chi-square with 1 dof = erfc(sqrt(stat/2))
        from math import erfc, sqrt
        p = float(erfc(sqrt(stat / 2.0)))
    return {"discordant_a_right_b_wrong": b01,
            "discordant_a_wrong_b_right": b10,
            "n_discordant": n,
            "p_value": p,
            "method": "exact_binomial" if _HAVE_SCIPY else "normal_approx"}


def provenance(config):
    """Everything needed to reproduce this run, written next to the results."""
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    dirty = git("status", "--porcelain")
    versions = {"python": sys.version.split()[0], "numpy": np.__version__}
    for mod in ("torch", "torch_geometric", "stim", "pymatching"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(dirty),
        "git_dirty_files": dirty.splitlines() if dirty else [],
        "argv": sys.argv,
        "config": config,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "library_versions": versions,
        "scipy_available_for_mcnemar": _HAVE_SCIPY,
    }


def run_arm(kind, d, p, shots, split_seed, model_seeds, lambdas, epochs, device):
    """Run one arm across all seeds, sweeping lambda per seed (Option A) and
    selecting best-by-validation. Returns the full grid plus the selected-lambda
    test results, and keeps the per-example correctness mask on the shared test set
    for McNemar (keyed by model_seed)."""
    # GNN-A has no aux head: its lambda grid is a single no-op point.
    arm_lambdas = [0.0] if kind == "A" else list(lambdas)

    grid = []            # every (seed, lambda) -> val/test metrics
    selected = []        # per seed: the best-by-val lambda and its TEST ler
    test_correct = {}    # seed -> boolean per-example correctness on shared test set
    test_truth = {}      # seed -> y_true on shared test set (sanity: identical across arms)

    for seed in model_seeds:
        # Split is fixed by split_seed, so the test set is identical across arms & seeds.
        train_ds, val_ds, test_ds = make_datasets(d, p, shots, split_seed)

        best = None  # (val_ler, lambda, test_ler, correct_mask, y_true)
        for lam in arm_lambdas:
            model = build_model(kind, d=d)
            model = train(model, train_ds, val_ds, lambda_aux=lam,
                          epochs=epochs, seed=seed, device=device)

            # Selection metric: LER on VAL. Reported metric: LER on TEST.
            vp, vt = predict_flips(model, val_ds, device)
            val_ler = logical_error_rate(vp, vt)
            tp, tt = predict_flips(model, test_ds, device)
            test_ler = logical_error_rate(tp, tt)

            grid.append({"seed": seed, "lambda": lam,
                         "val_ler": val_ler, "test_ler": test_ler})

            if best is None or val_ler < best[0]:
                best = (val_ler, lam, test_ler, (tp == tt), tt)

        val_ler, lam, test_ler, correct_mask, y_true = best
        selected.append({"seed": seed, "selected_lambda": lam,
                         "val_ler": val_ler, "test_ler": test_ler})
        test_correct[seed] = correct_mask
        test_truth[seed] = y_true

    test_lers = np.array([s["test_ler"] for s in selected], dtype=float)
    summary = {
        "arm": kind,
        "test_ler_mean": float(np.mean(test_lers)),
        "test_ler_std": float(np.std(test_lers, ddof=1)) if len(test_lers) > 1 else 0.0,
        "n_seeds": len(test_lers),
        "selected_per_seed": selected,
        "lambda_grid": grid,
    }
    return summary, test_correct, test_truth


def main():
    ap = argparse.ArgumentParser(description="Fair, provenance-recording GNN decoder runner (Option A).")
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--shots", type=int, default=50000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--split-seed", type=int, default=12345,
                    help="Fixes the train/val/test split; independent of model seeds.")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.5, 1.0])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    config = vars(args).copy()
    prov = provenance(config)

    results = {"provenance": prov, "arms": {}, "mcnemar_zx_vs_raw": {}}

    correctness = {}  # arm -> {seed -> mask}
    truths = {}       # arm -> {seed -> y_true}
    for kind in ("A", "Raw", "ZX"):
        summary, test_correct, test_truth = run_arm(
            kind, args.d, args.p, args.shots, args.split_seed,
            args.seeds, args.lambdas, args.epochs, args.device)
        results["arms"][kind] = summary
        correctness[kind] = test_correct
        truths[kind] = test_truth
        print(f"[{kind}] test LER = {summary['test_ler_mean']:.4f} "
              f"+/- {summary['test_ler_std']:.4f} over {summary['n_seeds']} seeds")

    # Sanity: Raw and ZX must have seen the identical test set for McNemar to be valid.
    for seed in args.seeds:
        if not np.array_equal(truths["ZX"][seed], truths["Raw"][seed]):
            raise RuntimeError(
                f"Test sets differ between ZX and Raw at seed {seed}. "
                "The split is not independent of model seed -- McNemar would be invalid.")

    # McNemar per seed on the shared test set (primary = first seed).
    for seed in args.seeds:
        results["mcnemar_zx_vs_raw"][str(seed)] = mcnemar(
            correctness["ZX"][seed], correctness["Raw"][seed])
    primary = str(args.seeds[0])
    results["mcnemar_zx_vs_raw"]["_primary_seed"] = primary
    print(f"[McNemar ZX vs Raw, seed {primary}] "
          f"p = {results['mcnemar_zx_vs_raw'][primary]['p_value']:.4g}")

    # Write ONE timestamped, provenance-carrying artifact. Never overwrites.
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outpath = outdir / f"run_{stamp}_d{args.d}_p{args.p}.json"
    outpath.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {outpath}")
    print("This file is now the source of truth: it carries its own git commit, "
          "config, seeds, and library versions. The number cannot exist without them.")


if __name__ == "__main__":
    main()