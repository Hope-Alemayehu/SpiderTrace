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
import time
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
    Linear(backbone.out_dim, num_qubits * 4)), but the harness only passes `d`. The
    count is independent of p -- noise instructions add no qubits -- so any p
    gives the right answer.
    """
    if d not in _NUM_QUBITS_CACHE:
        from qec_zx_dataset import build_circuit
        _NUM_QUBITS_CACHE[d] = build_circuit(d, 0.001).num_qubits
    return _NUM_QUBITS_CACHE[d]


_DATA_QUBITS_CACHE = {}

# Which dataset field each aux arm regresses. Mirrors gnn_models' aux_key.
AUX_KEY = {"Raw": "raw_target", "ZX": "zx_target"}


def _data_qubits_for(d):
    """Indices of the DATA qubits (the ones the final destructive M reads out).

    The aux target carries one slot per qubit index in the circuit, but most of
    those slots are dead: ancilla slots are ~99% identity (resets wipe them) and
    some indices are never used at all (26 slots at d=3 for 9 data + 8 real
    ancilla qubits). Restricting the aux head to data qubits removes the dead
    slots. Checked against the code distance: a rotated surface code of distance
    d has exactly d*d data qubits.
    """
    if d not in _DATA_QUBITS_CACHE:
        from qec_zx_dataset import build_circuit, _data_qubits
        circ = build_circuit(d, 0.001)
        dq = sorted(_data_qubits(circ))
        assert len(dq) == d * d, (
            f"expected {d*d} data qubits at d={d}, found {len(dq)}: {dq}")
        assert dq and dq[-1] < circ.num_qubits, "data qubit index out of range"
        _DATA_QUBITS_CACHE[d] = dq
    return _DATA_QUBITS_CACHE[d]


def compute_aux_class_weights(train_ds, aux_key, num_qubits, qubit_idx=None,
                              scheme="inv-freq"):
    """Per-Pauli (I,X,Y,Z) weights for the aux cross-entropy.

    Computed from the TRAINING split ONLY -- never val or test -- so no
    information leaks into selection or reporting.

    scheme "inv-freq": weight_c proportional to 1 / frequency_c, normalised so
    that sum_c freq_c * weight_c == 1. That normalisation keeps the EXPECTED
    aux loss at roughly the same scale as the unweighted version, so the same
    lambda grid keeps meaning roughly the same thing.

    Returns (weights, frequencies) as plain lists of 4 floats.
    """
    if scheme != "inv-freq":
        raise ValueError(f"unknown aux class-weight scheme {scheme!r}")
    counts = np.zeros(4, dtype=np.int64)
    for g in train_ds:
        idx = getattr(g, aux_key).view(-1, 4).argmax(dim=-1).numpy()
        if qubit_idx is not None:
            idx = idx[qubit_idx]
        counts += np.bincount(idx, minlength=4)
    total = int(counts.sum())
    freq = counts / max(total, 1)
    # A class absent from the training split is treated as if it occurred once,
    # so its weight stays finite instead of blowing up to infinity.
    freq_eff = np.maximum(freq, 1.0 / max(total, 1))
    w = 1.0 / freq_eff
    norm = float((freq * w).sum())
    if norm > 0:
        w = w / norm
    return [float(x) for x in w], [float(x) for x in freq]


def aux_diagnostics(model, dataset, device, num_qubits):
    """Is the aux head alive, or has it collapsed to predicting identity?

    Reported on the TEST split for the selected lambda. Reporting only -- never
    used for lambda selection or early stopping. Returns None for GNN-A, which
    has no aux head (as opposed to a collapsed one).
    """
    import torch
    from torch_geometric.loader import DataLoader

    if not getattr(model, "has_aux", False):
        return None

    model = model.to(device)
    model.eval()
    qubit_idx = getattr(model, "aux_index", None)
    conf = np.zeros((4, 4), dtype=np.int64)     # rows = true, cols = predicted
    loader = DataLoader(dataset, batch_size=hparam_batch_size(), shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, aux_logits = model(batch)
            B = aux_logits.size(0)
            true = getattr(batch, model.aux_key).view(B, num_qubits, 4).argmax(-1)
            if qubit_idx is not None:
                true = true.index_select(1, qubit_idx)
            pred = aux_logits.argmax(-1)
            t = true.reshape(-1).cpu().numpy()
            p = pred.reshape(-1).cpu().numpy()
            np.add.at(conf, (t, p), 1)

    total = int(conf.sum())
    correct = int(np.trace(conf))
    ident_total = int(conf[0, :].sum())
    nonid_total = total - ident_total
    nonid_correct = int(np.trace(conf)) - int(conf[0, 0])
    pred_ident = int(conf[:, 0].sum())
    recall = [float(conf[c, c] / conf[c, :].sum()) if conf[c, :].sum() else None
              for c in range(4)]
    frac_pred_identity = pred_ident / total if total else 0.0
    return {
        "aux_slots": int(getattr(model, "aux_slots", num_qubits)),
        "n_entries": total,
        "target_nonidentity_density": (nonid_total / total) if total else 0.0,
        "aux_acc_overall": (correct / total) if total else 0.0,
        "aux_acc_identity": (int(conf[0, 0]) / ident_total) if ident_total else 0.0,
        "aux_acc_nonidentity": (nonid_correct / nonid_total) if nonid_total else 0.0,
        "frac_pred_identity": frac_pred_identity,
        "per_class_recall": {"I": recall[0], "X": recall[1],
                             "Y": recall[2], "Z": recall[3]},
        "confusion_true_by_pred": conf.tolist(),
        "collapsed_to_identity": bool(frac_pred_identity > 0.999),
    }


def _model_type(model):
    """Map a model instance back to its arm key, for train_model's config dict."""
    if not getattr(model, "has_aux", False):
        return "A"
    return "Raw" if model.aux_key == "raw_target" else "ZX"


def make_datasets(d, p, shots, split_seed, val_frac=0.15, test_frac=0.15,
                  train_shots=None, extra_train_seed=None):
    """Train / val / test datasets for one (d, p), optionally with a resized train split.

    With train_shots=None this is exactly the base split below: `shots` draws from
    the sampler seeded with split_seed, permuted by split_seed.

    With train_shots=N, val and test are the base val and test objects themselves,
    so they are identical to the default run's. Only train changes, and the train
    sets are nested across N:
      * N <= base train size: the first N base train shots.
      * N >  base train size: all base train shots, then N - base extra shots from
        a second sampler seeded with extra_train_seed (default split_seed + 1).
        The extra shots come in fixed-size chunks (_fresh_shots), so a smaller
        N's extras are an exact prefix of a larger N's.
    extra_train_seed must differ from split_seed: the same seed would redraw the
    base pool, test shots included, into train.
    """
    base = _make_base_datasets(d, p, shots, split_seed, val_frac, test_frac)
    if train_shots is None:
        return base
    if train_shots < 1:
        raise ValueError(f"train_shots must be >= 1, got {train_shots}")
    extra_seed = resolve_extra_train_seed(split_seed, extra_train_seed)
    if extra_seed == split_seed:
        raise ValueError(
            f"extra_train_seed ({extra_seed}) equals split_seed: the extra train shots "
            "would repeat the base pool, test shots included.")

    key = (d, p, shots, split_seed, val_frac, test_frac, train_shots, extra_seed)
    if key in _DATASET_CACHE:
        return _DATASET_CACHE[key]
    train_base, val_ds, test_ds = base
    if train_shots <= len(train_base):
        train_ds = train_base[:train_shots]
    else:
        train_ds = train_base + _fresh_shots(
            d, p, train_shots - len(train_base), extra_seed)
    out = (train_ds, val_ds, test_ds)
    _DATASET_CACHE[key] = out
    return out


def resolve_extra_train_seed(split_seed, extra_train_seed):
    """The sampler seed for extra train shots: split_seed + 1 unless given."""
    return split_seed + 1 if extra_train_seed is None else extra_train_seed


def resolve_big_test_seed(split_seed, big_test_seed):
    """The sampler seed for the big test set: split_seed + 2 unless given."""
    return split_seed + 2 if big_test_seed is None else big_test_seed


# Fresh shots (extra train shots, the big test set) are drawn from ONE seeded
# sampler in sample() calls of this fixed size. The sampler's output depends only
# on its seed and the sequence of call sizes, so drawing n shots is an exact prefix
# of drawing more than n.
_FRESH_SHOT_CHUNK = 10000


def _fresh_shots(d, p, n, seed):
    """n shots from a sampler seeded with `seed`, as PyG Data objects.

    Same circuit, fault tables and DEM construction as _make_base_datasets, so
    the shots come from the same distribution with the same graph and targets.
    """
    from qec_zx_dataset import (
        build_circuit, build_fault_tables, sample_tuples, to_pyg_list,
    )
    circ = build_circuit(d, p)
    tables, _unseeded = build_fault_tables(circ)
    dem = circ.detector_error_model(decompose_errors=False, flatten_loops=True)
    assert dem.num_errors == tables.num_errors, (
        "DEM error-column alignment broken: reseeded sampler does not match "
        f"fault tables ({dem.num_errors} vs {tables.num_errors})."
    )
    sampler = dem.compile_sampler(seed=seed)
    out = []
    while len(out) < n:
        out.extend(to_pyg_list(sample_tuples(circ, tables, sampler, _FRESH_SHOT_CHUNK)))
    return out[:n]


def _make_base_datasets(d, p, shots, split_seed, val_frac=0.15, test_frac=0.15):
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


def build_model(kind, d, aux_qubits=None, **hparams):
    """Instantiate one arm. kind in {"A", "Raw", "ZX"}.

    CONTRACT: a fresh, untrained model. GNN-A has no aux head; Raw/ZX share a class
    and differ only by which aux target they regress.

    Class names were correct. Signature is (num_qubits, dropout=0.1) -- NOT (d) --
    so `d` is translated here. gnn_models.MODEL_REGISTRY already maps the arm keys.
    `hparams` go to the model unchanged; norm= and readout= configure the shared
    GNNBackbone identically for every arm.
    """
    from gnn_models import MODEL_REGISTRY
    if kind not in MODEL_REGISTRY:
        raise ValueError(f"unknown arm {kind!r}")
    # GNN-A has no aux head, so it never takes aux_qubits: it is bit-identical
    # whether or not the aux flags are set.
    if aux_qubits is not None and kind != "A":
        hparams["aux_qubits"] = aux_qubits
    return MODEL_REGISTRY[kind](_num_qubits_for(d), **hparams)


def _seed_all(seed):
    """Seed every RNG that affects a run: torch (weight init, loader shuffling),
    numpy, and CUDA. Called by run_arm BEFORE the model is built and again at the
    start of train()."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(model, train_ds, val_ds, lambda_aux, epochs, seed, device,
          aux_class_weight=None, checkpoint=None, ckpt_key=None,
          lr=1e-3, lr_schedule="constant", lr_min=1e-5, grad_clip=None, patience=10):
    """Train one model with early stopping on the VALIDATION split.

    CONTRACT:
      * Seed all RNGs from `seed` at the START (torch, numpy, cuda) so a run is
        reproducible from (seed, config) alone.
      * Early-stop / checkpoint-select on VAL only. Never touch test here.
      * For kind "A", lambda_aux is ignored (no aux head).
      * Returns (model, train_info): the trained model (best-val checkpoint
        restored) and a record of the run for the JSON: epochs run, the epoch
        whose weights were restored, wall time, and the per-epoch loss curves.
      * With a RunCheckpoint, the training state is saved after every epoch
        under `ckpt_key`. A model that already finished is loaded instead of
        retrained, and an interrupted one continues after its last saved epoch.
        wall_time_s then sums the time spent in every process.
      * lr / lr_schedule / lr_min / grad_clip / patience go to train_model (see
        its docstring). Their defaults are the original training.

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

    _seed_all(seed)

    # train_model does `device = next(model.parameters()).device` -- move first.
    model = model.to(device)

    if checkpoint is not None:
        done = checkpoint.load_done(ckpt_key)
        if done is not None:
            model.load_state_dict(done["best_state"])
            print(f"  [{_model_type(model):>3}] {ckpt_key} finished before the resume: "
                  "loaded its weights from the checkpoint")
            return model, done["train_info"]

    train_loader = DataLoader(train_ds, batch_size=hparam_batch_size(), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=hparam_batch_size(), shuffle=False)

    config = {
        "epochs": epochs,
        "lambda_aux": lambda_aux,
        "patience": patience,       # default 10, train_model's own default
        "lr": lr,
        "lr_schedule": lr_schedule,
        "lr_min": lr_min,
        "grad_clip": grad_clip,
        "pos_weight": None,         # -> 1.0, the plain-BCE / LER-optimal choice
        "model_type": _model_type(model),   # required: used in train_model's logging
        "balanced_sampler": False,  # logging only; we shuffle at the natural prior
        # None -> unweighted aux CE, i.e. the original behaviour.
        "aux_class_weight": aux_class_weight,
    }
    t0 = time.perf_counter()
    prior_s = 0.0          # training time spent in earlier processes (resume)
    resumed_after = None
    if checkpoint is not None:
        state = checkpoint.load_progress(ckpt_key)
        if state is not None:
            config["resume_state"] = state
            prior_s = state["elapsed_s"]
            resumed_after = state["epoch"]
        config["on_epoch_end"] = lambda st: checkpoint.save_progress(
            ckpt_key, st, prior_s + time.perf_counter() - t0)
    history, best_state = train_model(model, train_loader, val_loader, config)
    wall_time_s = prior_s + time.perf_counter() - t0
    model.load_state_dict(best_state)
    train_info = {
        "epochs_run": len(history["val_loss"]),
        "best_epoch": history["best_epoch"],   # 1-based; 0 = initial weights
        "wall_time_s": wall_time_s,            # train_model only, not eval
        "n_train": len(train_ds),
        "val_loss": history["val_loss"],
        "train_loss": history["train_loss"],
        "optim": {"lr": lr, "lr_schedule": lr_schedule, "lr_min": lr_min,
                  "grad_clip": grad_clip, "patience": patience},
        "lr": history["lr"],                   # lr used in each epoch
    }
    for k in ("grad_norm_mean", "grad_norm_max", "clipped_frac"):
        if k in history:                       # only with grad_clip
            train_info[k] = history[k]
    if resumed_after is not None:
        train_info["resumed_after_epoch"] = resumed_after
    if checkpoint is not None:
        checkpoint.save_done(ckpt_key, best_state, train_info)
    return model, train_info


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
    commit = git("rev-parse", "HEAD")
    # git("...") returns None when git could not be run at all, and "" when the
    # tree is genuinely clean. Distinguish them: an unreachable git must NOT be
    # recorded as a clean tree.
    git_available = commit is not None and dirty is not None
    if not git_available:
        print("WARNING: git could not be reached from this process. This run's "
              "provenance CANNOT record the commit or whether the tree was dirty "
              "(both recorded as null). Do not trust this file for reproducibility.")
    versions = {"python": sys.version.split()[0], "numpy": np.__version__}
    for mod in ("torch", "torch_geometric", "stim", "pymatching"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_available": git_available,
        "git_commit": commit,
        "git_dirty": bool(dirty) if git_available else None,
        "git_dirty_files": (dirty.splitlines() if dirty else []) if git_available else None,
        "argv": sys.argv,
        "config": config,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "library_versions": versions,
        "scipy_available_for_mcnemar": _HAVE_SCIPY,
    }


# --------------------------------------------------------------------------------------
# Per-epoch checkpointing. Every run saves its training state to
# <outdir>/checkpoint_<tag>/ after each epoch, so a crash loses at most the epoch in
# progress, and `--resume` continues from there. The directory is deleted once the
# final JSON is written, so a run that finishes leaves the same files as before.
#
# Contents of the directory:
#   partial.json      provenance at start, config, resumes, finished models (training
#                     record and val/test LER) and the model in progress (its curves)
#   progress_<key>.pt full training state of the model in progress (see
#                     gnn_models._training_state), rewritten after every epoch
#   done_<key>.pt     best weights and training record of each finished model
# <key> names one model: arm, model seed and lambda.
# --------------------------------------------------------------------------------------
# Config keys allowed to differ between a run and its resume.
_RESUME_FREE_KEYS = ("resume", "device", "outdir")


def checkpoint_tag(config):
    """Short hash of every config value that determines the run's results."""
    import hashlib
    key = {k: v for k, v in config.items() if k not in _RESUME_FREE_KEYS}
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def _replace_atomically(path, write):
    """write(tmp_path), then rename over `path`: a crash mid-write never leaves a
    truncated checkpoint, only the previous complete one."""
    import os
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


class RunCheckpoint:
    def __init__(self, outdir, config, prov, resume):
        tag = checkpoint_tag(config)
        self.dir = Path(outdir) / f"checkpoint_{tag}"
        self.partial_path = self.dir / "partial.json"
        if resume:
            if not self.partial_path.exists():
                raise RuntimeError(
                    f"--resume: no checkpoint for this configuration at {self.dir}. "
                    "A resume must repeat the original command (only --device and "
                    "--outdir may change).")
            self.partial = json.loads(self.partial_path.read_text())
            start = self.partial["provenance_at_start"]["git_commit"]
            if start and prov["git_commit"] and start != prov["git_commit"]:
                raise RuntimeError(
                    f"--resume: the checkpoint was written at commit {start}, this "
                    f"process runs {prov['git_commit']}. Check out {start} to resume.")
            if not (start and prov["git_commit"]):
                print("WARNING: git commit unknown at the start or at this resume; "
                      "cannot check that the code is unchanged.")
            self.partial["resumes"].append({
                "timestamp_utc": prov["timestamp_utc"],
                "git_commit": prov["git_commit"], "git_dirty": prov["git_dirty"],
                "hostname": prov["hostname"]})
            print(f"[checkpoint] resuming from {self.dir} "
                  f"({len(self.partial['done'])} models finished)")
        else:
            if self.dir.exists():
                raise RuntimeError(
                    f"A checkpoint for this configuration already exists at {self.dir}. "
                    "Pass --resume to continue it, or delete the directory to start over.")
            self.dir.mkdir(parents=True)
            self.partial = {
                "status": "in progress: partial results, not a finished run",
                "tag": tag,
                "config": {k: v for k, v in config.items() if k not in _RESUME_FREE_KEYS},
                "provenance_at_start": prov, "resumes": [],
                "done": {}, "in_progress": None}
        self._write_partial()

    @staticmethod
    def model_key(kind, seed, lam):
        return f"{kind}_seed{seed}_lam{lam}"

    def _write_partial(self):
        text = json.dumps(self.partial, indent=2)
        _replace_atomically(self.partial_path, lambda p: p.write_text(text))

    def _save_pt(self, name, obj):
        import torch
        _replace_atomically(self.dir / name, lambda p: torch.save(obj, p))

    def _load_pt(self, name):
        import torch
        path = self.dir / name
        if not path.exists():
            return None
        # weights_only=False: the state holds the numpy RNG state and the history.
        return torch.load(path, map_location="cpu", weights_only=False)

    def load_done(self, key):
        return self._load_pt(f"done_{key}.pt")

    def load_progress(self, key):
        return self._load_pt(f"progress_{key}.pt")

    def save_progress(self, key, state, elapsed_s):
        self._save_pt(f"progress_{key}.pt", {**state, "elapsed_s": elapsed_s})
        h = state["history"]
        self.partial["in_progress"] = {
            "model": key, "epochs_done": state["epoch"],
            "best_epoch": h["best_epoch"], "elapsed_s": elapsed_s,
            "train_loss": h["train_loss"], "val_loss": h["val_loss"]}
        self._write_partial()

    def save_done(self, key, best_state, train_info):
        self._save_pt(f"done_{key}.pt", {"best_state": best_state,
                                         "train_info": train_info})
        (self.dir / f"progress_{key}.pt").unlink(missing_ok=True)
        self.partial["done"][key] = {"train": train_info}
        self.partial["in_progress"] = None
        self._write_partial()

    def record_eval(self, key, train_info, val_ler, test_ler):
        # Rebuilt in full: robust to a crash between save_done's two writes.
        self.partial["done"][key] = {"train": train_info, "val_ler": val_ler,
                                     "test_ler": test_ler}
        self._write_partial()

    def resume_record(self):
        """For the final JSON: None unless this run was resumed."""
        if not self.partial["resumes"]:
            return None
        return {"tag": self.partial["tag"],
                "provenance_at_start": self.partial["provenance_at_start"],
                "resumes": self.partial["resumes"]}

    def finish(self):
        import shutil
        shutil.rmtree(self.dir)


def run_arm(kind, d, p, shots, split_seed, model_seeds, lambdas, epochs, device,
            aux_data_qubits_only=False, aux_class_weight="none",
            train_shots=None, extra_train_seed=None,
            norm="none", readout="add-mean-max", checkpoint=None, big_test_ds=None,
            train_opts=None):
    """Run one arm across all seeds, sweeping lambda per seed (Option A) and
    selecting best-by-validation. Returns the full grid plus the selected-lambda
    test results, and keeps the per-example correctness mask on the shared test set
    for McNemar (keyed by model_seed).

    aux_data_qubits_only / aux_class_weight apply IDENTICALLY to Raw and ZX (only
    their target field differs) and are no-ops for GNN-A, which has no aux head.
    Their defaults reproduce the original behaviour exactly.

    train_shots / extra_train_seed resize the train split only (see
    make_datasets); None keeps the base split.

    norm / readout configure the shared backbone (gnn_models.GNNBackbone) and
    apply IDENTICALLY to every arm. Their defaults are the original backbone.

    train_opts: optimisation options passed to train() (lr, lr_schedule, lr_min,
    grad_clip, patience), applied IDENTICALLY to every arm; None or {} keeps
    train()'s defaults, the original training.

    checkpoint: a RunCheckpoint, passed through to train(). big_test_ds: optional
    extra test set; each seed's SELECTED model is also scored on it. Selection
    never looks at it.

    Returns (summary, test_correct, test_truth, big), where big maps each seed to
    (correct_mask, y_true, eval_time_s) on big_test_ds ({} without one)."""
    data_args = dict(train_shots=train_shots, extra_train_seed=extra_train_seed)
    model_args = dict(norm=norm, readout=readout)
    # GNN-A has no aux head: its lambda grid is a single no-op point.
    arm_lambdas = [0.0] if kind == "A" else list(lambdas)

    grid = []            # every (seed, lambda) -> val/test metrics
    selected = []        # per seed: the best-by-val lambda and its TEST ler
    test_correct = {}    # seed -> boolean per-example correctness on shared test set
    test_truth = {}      # seed -> y_true on shared test set (sanity: identical across arms)
    big = {}             # seed -> (correct_mask, y_true, eval_time_s) on big_test_ds

    num_qubits = _num_qubits_for(d)
    aux_qubits = _data_qubits_for(d) if aux_data_qubits_only else None

    # Class weights come from the TRAINING split only. The split is fixed by
    # split_seed, so they are identical across seeds -> compute once per arm.
    # Each arm weights its OWN target's class frequencies: same rule, applied to
    # each arm's own data, which is the fairness that matters here.
    aux_weights = aux_freq = None
    if kind != "A" and aux_class_weight != "none":
        train_ds0, _, _ = make_datasets(d, p, shots, split_seed, **data_args)
        aux_weights, aux_freq = compute_aux_class_weights(
            train_ds0, AUX_KEY[kind], num_qubits, aux_qubits,
            scheme=aux_class_weight)
        print(f"[{kind}] aux class freq (I,X,Y,Z) = "
              f"{[round(f, 4) for f in aux_freq]} -> weights "
              f"{[round(w, 3) for w in aux_weights]}")

    for seed in model_seeds:
        # Split is fixed by split_seed, so the test set is identical across arms & seeds.
        train_ds, val_ds, test_ds = make_datasets(d, p, shots, split_seed, **data_args)

        best = None  # (val_ler, lambda, test_ler, correct_mask, y_true, aux, model)
        for lam in arm_lambdas:
            # Seed BEFORE building the model so `seed` controls the initial
            # weights too. Previously the model was constructed from whatever
            # RNG state the process happened to be in, so initial weights were
            # not reproducible from (seed, config). train() seeds again at its
            # start, so the RNG state entering the data loaders is unchanged.
            _seed_all(seed)
            model = build_model(kind, d=d, aux_qubits=aux_qubits, **model_args)
            n_params = sum(t.numel() for t in model.parameters())
            key = RunCheckpoint.model_key(kind, seed, lam)
            model, train_info = train(model, train_ds, val_ds, lambda_aux=lam,
                                      epochs=epochs, seed=seed, device=device,
                                      aux_class_weight=aux_weights,
                                      checkpoint=checkpoint, ckpt_key=key,
                                      **(train_opts or {}))

            # Selection metric: LER on VAL. Reported metric: LER on TEST.
            vp, vt = predict_flips(model, val_ds, device)
            val_ler = logical_error_rate(vp, vt)
            tp, tt = predict_flips(model, test_ds, device)
            test_ler = logical_error_rate(tp, tt)
            if checkpoint is not None:
                checkpoint.record_eval(key, train_info, val_ler, test_ler)

            grid.append({"seed": seed, "lambda": lam,
                         "val_ler": val_ler, "test_ler": test_ler,
                         "train": train_info})

            if best is None or val_ler < best[0]:
                # Aux diagnostics on TEST for this lambda: reporting only, never
                # used for selection (selection above is val LER).
                aux = aux_diagnostics(model, test_ds, device, num_qubits)
                best = (val_ler, lam, test_ler, (tp == tt), tt, aux, model)

        val_ler, lam, test_ler, correct_mask, y_true, aux, best_model = best
        if big_test_ds is not None:
            t0 = time.perf_counter()
            bp, bt = predict_flips(best_model, big_test_ds, device)
            big[seed] = ((bp == bt), bt, time.perf_counter() - t0)
        # test_wrong_idx: positions in the shared test split that this model got
        # wrong. Lets runs in separate processes (e.g. different train sizes) be
        # compared with a paired McNemar later.
        selected.append({"seed": seed, "selected_lambda": lam,
                         "val_ler": val_ler, "test_ler": test_ler,
                         "aux": aux,
                         "test_wrong_idx": np.flatnonzero(~correct_mask).tolist()})
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
        # aux configuration actually used by this arm (None for GNN-A)
        "aux_slots": (len(aux_qubits) if aux_qubits is not None else num_qubits)
                     if kind != "A" else None,
        "aux_class_weights": aux_weights,
        "aux_class_freq_train": aux_freq,
        # backbone actually built for this arm (identical across seeds and lambdas)
        "model": {**model_args, "n_params": n_params},
    }
    return summary, test_correct, test_truth, big

# Paired McNemar tests reported per run: (results key, arm a, arm b), b = None = MWPM.
_PAIRS = [("mcnemar_zx_vs_raw", "ZX", "Raw"),
          ("mcnemar_zx_vs_mwpm", "ZX", None),
          ("mcnemar_a_vs_mwpm", "A", None)]


def paired_mcnemar(arms, seeds, correctness, mwpm_correct, label=""):
    """McNemar per seed (primary = first seed) for each pair whose arms were run,
    on one shared test set. correctness: arm -> {seed -> mask}."""
    out = {}
    primary = str(seeds[0])
    for key, a, b in _PAIRS:
        if a not in arms or (b is not None and b not in arms):
            continue
        out[key] = {}
        for seed in seeds:
            out[key][str(seed)] = mcnemar(
                correctness[a][seed],
                mwpm_correct if b is None else correctness[b][seed])
        out[key]["_primary_seed"] = primary
        print(f"[McNemar {a} vs {b or 'MWPM'}{label}, seed {primary}] "
              f"p = {out[key][primary]['p_value']:.4g}")
    return out


def mwpm_on_test(d, p, test_ds):
    """Decode the shared test split with MWPM. Deterministic, so computed once per run."""
    import pymatching
    from qec_zx_dataset import build_circuit

    circ = build_circuit(d, p)
    dem = circ.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)

    # Nodes 0..num_detectors-1 are detectors, the last node is the boundary.
    # Column 0 of x is the 'fired' bit, so this rebuilds each shot's syndrome.
    dets = np.stack([g.x[:-1, 0].numpy() for g in test_ds]).astype(np.uint8)
    assert dets.shape[1] == circ.num_detectors, "syndrome length mismatch"
    y_true = np.array([int(g.y.item()) for g in test_ds])

    pred = matching.decode_batch(dets)[:, 0].astype(int)
    ler = float(np.mean(pred != y_true))
    return ler, (pred == y_true), y_true


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
    # ---- aux-head options. Defaults reproduce the ORIGINAL behaviour exactly. ----
    ap.add_argument("--aux-data-qubits-only", action="store_true",
                    help="restrict the aux head to DATA qubits (d*d slots) instead "
                         "of every qubit index. Applies to Raw and ZX identically; "
                         "no-op for GNN-A. Default: off (all slots, as before).")
    ap.add_argument("--aux-class-weight", choices=["none", "inv-freq"],
                    default="none",
                    help="per-Pauli weighting of the aux cross-entropy, computed "
                         "from the TRAINING split only. 'inv-freq' stops the head "
                         "collapsing to predicting identity everywhere. "
                         "Default: none (unweighted, as before).")
    # ---- run-shape options. Defaults reproduce the ORIGINAL behaviour exactly. ----
    ap.add_argument("--arms", nargs="+", choices=["A", "Raw", "ZX"],
                    default=["A", "Raw", "ZX"],
                    help="which arms to train. McNemar tests run only for the pairs "
                         "present. Default: all three, as before.")
    ap.add_argument("--train-shots", type=int, default=None,
                    help="resize the TRAIN split to this many shots; val and test "
                         "stay the --shots / --split-seed split, unchanged. Up to the "
                         "base train size it takes the first N base train shots; "
                         "beyond it, adds fresh shots from --extra-train-seed. Train "
                         "sets are nested across N. Default: the base split, as before.")
    ap.add_argument("--extra-train-seed", type=int, default=None,
                    help="sampler seed for train shots beyond the base split. Must "
                         "differ from --split-seed. Default: split seed + 1.")
    ap.add_argument("--expect-mwpm-errors", type=int, default=None,
                    help="stop before training unless MWPM makes exactly this many "
                         "errors on the test split: a fingerprint that the test set "
                         "is the expected one. Default: no check.")
    # ---- backbone options (all arms). Defaults reproduce the ORIGINAL backbone. ----
    from gnn_models import NORM_CHOICES, READOUT_CHOICES
    ap.add_argument("--norm", choices=NORM_CHOICES, default="none",
                    help="normalization after each GINEConv: 'layer' = LayerNorm "
                         "before the ReLU. Applies to every arm. Default: none, as before.")
    ap.add_argument("--readout", choices=READOUT_CHOICES, default="add-mean-max",
                    help="graph pooling: 'mean-max' drops the add pool, which on this "
                         "fixed-size graph is num_nodes x the mean pool. Applies to "
                         "every arm. Default: add-mean-max, as before.")
    # ---- resume and extra evaluation. Defaults reproduce the ORIGINAL results. ----
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from its per-epoch checkpoint "
                         "in --outdir. Repeat the original command exactly (only "
                         "--device and --outdir may change) and add --resume.")
    ap.add_argument("--big-test-shots", type=int, default=None,
                    help="also score each arm's selected model and MWPM on this many "
                         "fresh test shots from --big-test-seed, reported under "
                         "'big_test'. The 7,500-shot test split stays the primary one "
                         "and is unchanged. Default: off.")
    ap.add_argument("--big-test-seed", type=int, default=None,
                    help="sampler seed for the big test set. Must differ from "
                         "--split-seed and the extra-train seed. Default: split seed + 2.")
    # ---- optimisation (all arms). Defaults reproduce the ORIGINAL training. ----
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Adam learning rate (the peak lr with --lr-schedule cosine). "
                         "Default: 1e-3, as before.")
    ap.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                    help="'cosine' decays the lr from --lr to --lr-min over --epochs, "
                         "stepped once per epoch. Default: constant, as before.")
    ap.add_argument("--lr-min", type=float, default=1e-5,
                    help="final lr of the cosine schedule (unused with constant). "
                         "Default: 1e-5.")
    ap.add_argument("--grad-clip", type=float, default=None,
                    help="clip the global gradient norm to this value before each "
                         "optimizer step; records per-epoch grad-norm statistics. "
                         "Default: no clipping, as before.")
    ap.add_argument("--patience", type=int, default=10,
                    help="early-stopping patience in epochs (val loss). With a cosine "
                         "schedule, set it high enough that the low-lr phase is "
                         "reached. Default: 10, as before.")
    args = ap.parse_args()
    if args.lr <= 0:
        ap.error("--lr must be > 0")
    if args.lr_schedule == "cosine" and not 0 <= args.lr_min <= args.lr:
        ap.error("--lr-min must satisfy 0 <= --lr-min <= --lr")
    if args.grad_clip is not None and args.grad_clip <= 0:
        ap.error("--grad-clip must be > 0")
    if args.patience < 1:
        ap.error("--patience must be >= 1")
    train_opts = dict(lr=args.lr, lr_schedule=args.lr_schedule, lr_min=args.lr_min,
                      grad_clip=args.grad_clip, patience=args.patience)
    if args.extra_train_seed is not None and args.train_shots is None:
        ap.error("--extra-train-seed only applies with --train-shots")
    big_seed = None
    if args.big_test_seed is not None and args.big_test_shots is None:
        ap.error("--big-test-seed only applies with --big-test-shots")
    if args.big_test_shots is not None:
        if args.big_test_shots < 1:
            ap.error("--big-test-shots must be >= 1")
        big_seed = resolve_big_test_seed(args.split_seed, args.big_test_seed)
        extra_seed = resolve_extra_train_seed(args.split_seed, args.extra_train_seed)
        if big_seed in (args.split_seed, extra_seed):
            ap.error(f"--big-test-seed {big_seed} must differ from the split seed "
                     f"({args.split_seed}) and the extra-train seed ({extra_seed}): "
                     "the big test set would repeat train or test shots.")
    arms = [k for k in ("A", "Raw", "ZX") if k in args.arms]   # canonical order
    data_args = dict(train_shots=args.train_shots,
                     extra_train_seed=args.extra_train_seed)

    config = vars(args).copy()
    prov = provenance(config)

    results = {"provenance": prov, "arms": {}}

    # MWPM on the SAME test shots as the GNNs. No training, no seeds: computed once,
    # BEFORE training so a wrong test set (--expect-mwpm-errors) fails fast. It uses
    # no torch or numpy global RNG, and every model is seeded right before it is
    # built, so running it first does not change any GNN result.
    train_ds, val_ds, test_ds = make_datasets(args.d, args.p, args.shots,
                                              args.split_seed, **data_args)
    mwpm_ler, mwpm_correct, mwpm_truth = mwpm_on_test(args.d, args.p, test_ds)
    mwpm_wrong = np.flatnonzero(~mwpm_correct)
    results["mwpm"] = {"test_ler": mwpm_ler, "n_test": len(test_ds),
                       "n_errors": int(mwpm_wrong.size),
                       "test_wrong_idx": mwpm_wrong.tolist()}
    print(f"[MWPM] test LER = {mwpm_ler:.4f} ({mwpm_wrong.size} errors) on "
          f"{len(test_ds)} test shots")
    if (args.expect_mwpm_errors is not None
            and mwpm_wrong.size != args.expect_mwpm_errors):
        raise RuntimeError(
            f"MWPM made {mwpm_wrong.size} errors on the test split, expected "
            f"{args.expect_mwpm_errors}. This is not the expected test set (seeded "
            "stim sampling can differ between machines). Stopping before training.")

    n_base_train = len(make_datasets(args.d, args.p, args.shots, args.split_seed)[0])
    results["splits"] = {
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "n_train_base": min(len(train_ds), n_base_train),
        "n_train_extra": max(0, len(train_ds) - n_base_train),
        "extra_train_seed": (None if args.train_shots is None else
                             resolve_extra_train_seed(args.split_seed,
                                                      args.extra_train_seed)),
    }
    print(f"[splits] train {len(train_ds)} (base {results['splits']['n_train_base']}"
          f" + extra {results['splits']['n_train_extra']}), val {len(val_ds)}, "
          f"test {len(test_ds)}")

    # Big test set: fresh shots from their own sampler seed, never used for training,
    # selection or early stopping. Built and decoded by MWPM before training, so a
    # memory problem shows up before hours of training rather than after.
    big_ds = None
    if args.big_test_shots is not None:
        t0 = time.perf_counter()
        big_ds = _fresh_shots(args.d, args.p, args.big_test_shots, big_seed)
        gen_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        big_mwpm_ler, big_mwpm_correct, big_mwpm_truth = mwpm_on_test(
            args.d, args.p, big_ds)
        big_mwpm_s = time.perf_counter() - t0
        big_mwpm_wrong = np.flatnonzero(~big_mwpm_correct)
        print(f"[big test] {len(big_ds)} shots from seed {big_seed}, built in "
              f"{gen_s:.1f} s. MWPM LER = {big_mwpm_ler:.5f} "
              f"({big_mwpm_wrong.size} errors), decoded in {big_mwpm_s:.1f} s")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = RunCheckpoint(outdir, config, prov, args.resume)

    correctness = {}  # arm -> {seed -> mask}
    truths = {}       # arm -> {seed -> y_true}
    big = {}          # arm -> {seed -> (mask, y_true, eval_time_s)} on the big test set
    for kind in arms:
        summary, test_correct, test_truth, big[kind] = run_arm(
            kind, args.d, args.p, args.shots, args.split_seed,
            args.seeds, args.lambdas, args.epochs, args.device,
            aux_data_qubits_only=args.aux_data_qubits_only,
            aux_class_weight=args.aux_class_weight, **data_args,
            norm=args.norm, readout=args.readout,
            checkpoint=checkpoint, big_test_ds=big_ds, train_opts=train_opts)
        results["arms"][kind] = summary
        correctness[kind] = test_correct
        truths[kind] = test_truth
        print(f"[{kind}] test LER = {summary['test_ler_mean']:.4f} "
              f"+/- {summary['test_ler_std']:.4f} over {summary['n_seeds']} seeds")
        aux0 = summary["selected_per_seed"][0].get("aux")
        if aux0 is not None:
            print(f"[{kind}] aux head: non-identity acc = "
                  f"{aux0['aux_acc_nonidentity']:.4f}  predicts 'I' for "
                  f"{aux0['frac_pred_identity']:.3f} of {aux0['aux_slots']} slots"
                  f"  -> {'COLLAPSED' if aux0['collapsed_to_identity'] else 'alive'}")

    # Sanity: every arm, at every seed, must have seen MWPM's test set (hence each
    # other's) for the paired McNemar tests to be valid.
    for kind in arms:
        for seed in args.seeds:
            if not np.array_equal(truths[kind][seed], mwpm_truth):
                raise RuntimeError(
                    f"Test set of {kind} at seed {seed} differs from MWPM's. The split "
                    "is not independent of model seed -- McNemar would be invalid.")

    # McNemar per seed on the shared test set (primary = first seed), for each
    # pair whose arms were run.
    results.update(paired_mcnemar(arms, args.seeds, correctness, mwpm_correct))

    # Big test set: secondary, reported separately; the primary results above are
    # computed exactly as without it.
    if big_ds is not None:
        bt = {"n_test": len(big_ds), "seed": big_seed, "build_time_s": gen_s,
              "mwpm": {"test_ler": big_mwpm_ler, "n_errors": int(big_mwpm_wrong.size),
                       "test_wrong_idx": big_mwpm_wrong.tolist(),
                       "decode_time_s": big_mwpm_s},
              "arms": {}}
        big_correct = {}
        for kind in arms:
            bt["arms"][kind] = {}
            big_correct[kind] = {}
            for seed in args.seeds:
                mask, y_true, eval_s = big[kind][seed]
                if not np.array_equal(y_true, big_mwpm_truth):
                    raise RuntimeError(f"Big test set of {kind} at seed {seed} differs "
                                       "from MWPM's.")
                wrong = np.flatnonzero(~mask)
                bt["arms"][kind][str(seed)] = {
                    "test_ler": float(np.mean(~mask)), "n_errors": int(wrong.size),
                    "test_wrong_idx": wrong.tolist(), "eval_time_s": eval_s}
                big_correct[kind][seed] = mask
            first = bt["arms"][kind][str(args.seeds[0])]
            print(f"[big test] [{kind}] LER = {first['test_ler']:.5f} "
                  f"({first['n_errors']} errors), seed {args.seeds[0]}")
        bt.update(paired_mcnemar(arms, args.seeds, big_correct, big_mwpm_correct,
                                 label=", big test"))
        results["big_test"] = bt

    resumed = checkpoint.resume_record()
    if resumed is not None:
        results["checkpoint"] = resumed

    # Write ONE timestamped, provenance-carrying artifact. Never overwrites.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outpath = outdir / f"run_{stamp}_d{args.d}_p{args.p}.json"
    outpath.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {outpath}")
    # Only now is the checkpoint redundant: the JSON holds everything it did.
    checkpoint.finish()
    print("This file is now the source of truth: it carries its own git commit, "
          "config, seeds, and library versions. The number cannot exist without them.")


if __name__ == "__main__":
    main()