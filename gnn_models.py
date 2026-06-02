"""
gnn_models.py
=============
Three GNN decoders that share an IDENTICAL message-passing backbone and differ
only in auxiliary supervision, for the GNN-A / GNN-Raw / GNN-ZX study driven by
the data pipeline in ``qec_zx_dataset.py``.

    GNN_A   : backbone -> logical-flip head only            (no aux head)
    GNN_Raw : backbone -> flip head + aux head on raw_target (un-propagated Pauli)
    GNN_ZX  : backbone -> flip head + aux head on zx_target  (ZX-propagated Pauli)

The aux head exists only at training time; it is discarded at inference, so all
three models use the SAME forward path (syndrome graph -> flip logit) at eval.

Inputs (batched torch_geometric.data.Batch from qec_zx_dataset.to_pyg_list):
    x          (num_nodes, 4)            detector coords [x, y, t, is_virtual]
    edge_index (2, num_edges)
    edge_attr  (num_edges, 4)            [dx, dy, dt, |d|]
    batch      (num_nodes,)              graph assignment
    y          (batch_size,) or (B, 1)   logical flip label
    raw_target (B, num_qubits, 4)        one-hot Pauli {I,X,Y,Z}
    zx_target  (B, num_qubits, 4)        one-hot Pauli {I,X,Y,Z}

CLI:
    python gnn_models.py --dry-run
    python gnn_models.py --d 3 --p 0.01 --shots 20000 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GINEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from qec_zx_dataset import (
    build_circuit,
    build_fault_tables,
    sample_tuples,
    to_pyg_list,
)

NODE_IN_DIM = 4
EDGE_IN_DIM = 5        # [dx, dy, dt, |d|, is_boundary] (see qec_zx_dataset.build_graph)
HIDDEN = 96
NUM_PAULI = 4          # {I, X, Y, Z}
NUM_MP_LAYERS = 4
READOUT_DIM = HIDDEN * 3   # add + mean + max -> 288


# --------------------------------------------------------------------------- #
# Shared backbone
# --------------------------------------------------------------------------- #
class GNNBackbone(nn.Module):
    """4x GINEConv (hidden=96) + 3-way global pooling -> 288-dim graph vector."""

    def __init__(self, hidden: int = HIDDEN, num_layers: int = NUM_MP_LAYERS,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout
        self.node_encoder = nn.Linear(NODE_IN_DIM, hidden)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # Each GINEConv uses a 2-layer MLP with ReLU.
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            # edge_dim projects the 4-dim edge features to `hidden` internally.
            self.convs.append(GINEConv(mlp, train_eps=True, edge_dim=EDGE_IN_DIM))

    def forward(self, x, edge_index, edge_attr, batch) -> torch.Tensor:
        x = self.node_encoder(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        graph = torch.cat(
            [global_add_pool(x, batch),
             global_mean_pool(x, batch),
             global_max_pool(x, batch)],
            dim=1,
        )
        return F.dropout(graph, p=self.dropout, training=self.training)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class GNN_A(nn.Module):
    """Backbone -> logical-flip head. No auxiliary supervision."""

    has_aux = False

    def __init__(self, num_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.num_qubits = num_qubits
        self.backbone = GNNBackbone(dropout=dropout)
        self.flip_head = nn.Linear(READOUT_DIM, 1)

    def forward(self, data):
        g = self.backbone(data.x, data.edge_index, data.edge_attr, data.batch)
        flip_logit = self.flip_head(g).view(-1)        # raw logit (no sigmoid)
        return flip_logit, None


class _GNNWithAux(nn.Module):
    """Backbone + flip head + per-qubit Pauli auxiliary head."""

    has_aux = True
    aux_key = ""    # "raw_target" or "zx_target"

    def __init__(self, num_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.num_qubits = num_qubits
        self.backbone = GNNBackbone(dropout=dropout)
        self.flip_head = nn.Linear(READOUT_DIM, 1)
        self.aux_head = nn.Linear(READOUT_DIM, num_qubits * NUM_PAULI)

    def forward(self, data):
        g = self.backbone(data.x, data.edge_index, data.edge_attr, data.batch)
        flip_logit = self.flip_head(g).view(-1)
        aux_logits = self.aux_head(g).view(-1, self.num_qubits, NUM_PAULI)
        return flip_logit, aux_logits


class GNN_Raw(_GNNWithAux):
    aux_key = "raw_target"


class GNN_ZX(_GNNWithAux):
    aux_key = "zx_target"


MODEL_REGISTRY = {"A": GNN_A, "Raw": GNN_Raw, "ZX": GNN_ZX}


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def _aux_loss(aux_logits: torch.Tensor, target_onehot: torch.Tensor,
              num_qubits: int) -> torch.Tensor:
    """(1 / num_qubits) * sum_q CrossEntropy(qubit q), averaged over the batch.

    Equivalent to the mean per-qubit CE; the lambda scaling is applied by the
    caller. target_onehot is (B, num_qubits, 4) one-hot -> argmax to class idx.
    """
    B = aux_logits.size(0)
    target_idx = target_onehot.view(B, num_qubits, NUM_PAULI).argmax(dim=-1)  # (B,Nq)
    # reduction='mean' over B*num_qubits == (1/B) * (1/num_qubits) * sum_b sum_q CE
    return F.cross_entropy(aux_logits.reshape(-1, NUM_PAULI), target_idx.reshape(-1))


def compute_loss(model, data, bce: nn.BCEWithLogitsLoss,
                 lambda_aux: float) -> torch.Tensor:
    y = data.y.view(-1).float()
    flip_logit, aux_logits = model(data)
    loss = bce(flip_logit, y)
    if model.has_aux and lambda_aux > 0:
        target = getattr(data, model.aux_key)
        loss = loss + lambda_aux * _aux_loss(aux_logits, target, model.num_qubits)
    return loss


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _binary_counts(pred: torch.Tensor, y: torch.Tensor) -> Tuple[int, int, int, int]:
    pred = pred.bool(); y = y.bool()
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum()); tn = int((~pred & ~y).sum())
    return tp, fp, fn, tn


@torch.no_grad()
def evaluate(model, loader, model_type: str) -> Dict[str, float]:
    """Returns logical_error_rate, recall, precision. Aux head is ignored:
    all model types decode from the syndrome graph alone."""
    model.eval()
    device = next(model.parameters()).device
    tp = fp = fn = tn = 0
    wrong = total = 0
    for data in loader:
        data = data.to(device)
        flip_logit, _ = model(data)
        prob = torch.sigmoid(flip_logit)
        pred = (prob > 0.5).long()
        y = data.y.view(-1).long()
        a, b, c, d = _binary_counts(pred, y)
        tp += a; fp += b; fn += c; tn += d
        wrong += int((pred != y).sum()); total += y.numel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    logical_error_rate = wrong / total if total else 0.0
    return {"logical_error_rate": logical_error_rate,
            "recall": recall, "precision": precision}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_model(model, train_loader, val_loader, config: Dict):
    """Adam lr=1e-3, early stopping (patience 10) on val loss.

    config: epochs, lambda_aux, pos_weight (None -> 1.0), model_type.
    Returns (history, best_state_dict).

    NOTE: pos_weight defaults to 1.0 (plain BCE). The logical error rate weighs
    false positives and false negatives EQUALLY (it is a 0-1 loss), so the
    Bayes-optimal rule is to estimate P(flip|syndrome) and threshold at 0.5 --
    which plain BCE learns. A class-balancing pos_weight (~neg/pos) instead
    optimizes a recall-weighted objective and makes the model over-predict
    flips, inflating LER (empirically above the trivial baseline at d=3).
    """
    device = next(model.parameters()).device
    epochs = config["epochs"]
    lambda_aux = config.get("lambda_aux", 0.0)
    patience = config.get("patience", 10)

    pos_weight = config.get("pos_weight")
    if pos_weight is None:
        pos_weight = 1.0
    bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(pos_weight), device=device))
    print(f"  [{config['model_type']:>3}] pos_weight in use = {float(pos_weight)}  "
          f"| balanced sampler = {config.get('balanced_sampler', True)}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = {"train_loss": [], "val_loss": [], "val_recall": [],
               "val_precision": [], "val_logical_error_rate": []}
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(epochs):
        # ---- train ----
        model.train()
        running = 0.0; nb = 0
        for data in train_loader:
            data = data.to(device)
            opt.zero_grad()
            loss = compute_loss(model, data, bce, lambda_aux)
            loss.backward()
            opt.step()
            running += loss.item(); nb += 1
        train_loss = running / max(nb, 1)

        # ---- validation loss (same objective) ----
        model.eval()
        vrun = 0.0; vnb = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                vrun += compute_loss(model, data, bce, lambda_aux).item(); vnb += 1
        val_loss = vrun / max(vnb, 1)

        m = evaluate(model, val_loader, config["model_type"])
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_recall"].append(m["recall"])
        history["val_precision"].append(m["precision"])
        history["val_logical_error_rate"].append(m["logical_error_rate"])

        print(f"  [{config['model_type']:>3}] epoch {epoch + 1:>3}/{epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_recall={m['recall']:.3f}  val_prec={m['precision']:.3f}")

        # ---- early stopping on val loss ----
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  [{config['model_type']:>3}] early stop at epoch "
                      f"{epoch + 1} (no val-loss improvement for {patience}).")
                break

    return history, best_state


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_loaders(d: int, p: float, num_shots: int, seed: int,
                  batch_size: int = 256, val_frac: float = 0.2,
                  balanced_sampler: bool = True):
    """Build a fresh dataset for one seed and split into train/val loaders.

    With balanced_sampler=True (default), the TRAIN loader draws samples with a
    WeightedRandomSampler so each batch is ~50/50 flip/no-flip, the standard
    remedy for class imbalance (here ~16% positive at d=5, p=0.003). The loss
    keeps pos_weight=1.0 -- the rebalancing happens in the data, not the loss,
    so we don't double-count the minority class. The VAL loader is left at the
    natural class prior so val loss / metrics reflect the true distribution.

    Returns (train_loader, val_loader, num_qubits).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    circ = build_circuit(d, p)
    tables, sampler = build_fault_tables(circ)        # reference == SpiderTrace here
    tuples = list(sample_tuples(circ, tables, sampler, num_shots, seed=seed))
    data_list = to_pyg_list(tuples)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(data_list), generator=g).tolist()
    data_list = [data_list[i] for i in perm]
    n_val = max(1, int(len(data_list) * val_frac))
    val_list, train_list = data_list[:n_val], data_list[n_val:]

    if balanced_sampler:
        labels = np.array([int(dp.y.item()) for dp in train_list])
        class_count = np.bincount(labels, minlength=2)          # [n_neg, n_pos]
        # per-sample weight = inverse class frequency -> ~uniform class draw
        class_weight = 1.0 / np.maximum(class_count, 1)
        sample_weights = torch.as_tensor(class_weight[labels], dtype=torch.double)
        wrs_gen = torch.Generator().manual_seed(seed)
        sampler_obj = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_list),
            replacement=True, generator=wrs_gen)
        train_loader = DataLoader(train_list, batch_size=batch_size,
                                  sampler=sampler_obj)
    else:
        train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, tables.num_qubits


# --------------------------------------------------------------------------- #
# Experiment driver
# --------------------------------------------------------------------------- #
def _train_one(model_type: str, num_qubits: int, train_loader, val_loader,
               epochs: int, lambda_aux: float, pos_weight: Optional[float],
               device) -> Dict:
    model = MODEL_REGISTRY[model_type](num_qubits).to(device)
    config = {"epochs": epochs, "lambda_aux": lambda_aux,
              "pos_weight": pos_weight, "model_type": model_type}
    history, best_state = train_model(model, train_loader, val_loader, config)
    model.load_state_dict(best_state)
    metrics = evaluate(model, val_loader, model_type)
    best_val_loss = min(history["val_loss"]) if history["val_loss"] else float("inf")
    return {"metrics": metrics, "best_val_loss": best_val_loss,
            "lambda_aux": lambda_aux, "history": history}


def run_experiment(d: int, p: float, num_shots: int, seeds: List[int],
                   lambda_values: List[float], epochs: int = 100,
                   pos_weight: Optional[float] = None,
                   out_dir: str = "results") -> Dict:
    """Train GNN-A, GNN-Raw, GNN-ZX for each seed. GNN-ZX sweeps lambda and the
    best (by val loss) is kept. Saves mean+/-std and per-seed raw results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_seed = {"A": [], "Raw": [], "ZX": []}
    raw_records = []

    for seed in seeds:
        print(f"\n=== seed {seed} (d={d}, p={p}, shots={num_shots}) ===")
        train_loader, val_loader, num_qubits = build_loaders(d, p, num_shots, seed)

        # GNN-A (no aux)
        res_a = _train_one("A", num_qubits, train_loader, val_loader,
                           epochs, 0.0, pos_weight, device)
        # GNN-Raw (single lambda; use the first of the sweep as its weight)
        lam_raw = lambda_values[0]
        res_raw = _train_one("Raw", num_qubits, train_loader, val_loader,
                             epochs, lam_raw, pos_weight, device)
        # GNN-ZX (sweep lambda, pick best by val loss)
        zx_runs = []
        for lam in lambda_values:
            print(f"  -- GNN-ZX lambda={lam} --")
            zx_runs.append(_train_one("ZX", num_qubits, train_loader, val_loader,
                                      epochs, lam, pos_weight, device))
        res_zx = min(zx_runs, key=lambda r: r["best_val_loss"])
        print(f"  GNN-ZX best lambda={res_zx['lambda_aux']} "
              f"(val_loss={res_zx['best_val_loss']:.4f})")

        for key, res in (("A", res_a), ("Raw", res_raw), ("ZX", res_zx)):
            per_seed[key].append(res["metrics"])
        raw_records.append({
            "seed": seed,
            "A": {"metrics": res_a["metrics"], "lambda_aux": 0.0},
            "Raw": {"metrics": res_raw["metrics"], "lambda_aux": lam_raw},
            "ZX": {"metrics": res_zx["metrics"], "lambda_aux": res_zx["lambda_aux"],
                   "lambda_sweep": {str(r["lambda_aux"]): r["best_val_loss"]
                                    for r in zx_runs}},
        })

    # aggregate
    def agg(metric_list, field):
        vals = [m[field] for m in metric_list]
        mean = statistics.fmean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return {"mean": mean, "std": std, "values": vals}

    summary = {}
    for key in ("A", "Raw", "ZX"):
        summary[key] = {f: agg(per_seed[key], f)
                        for f in ("logical_error_rate", "recall", "precision")}

    config = {"d": d, "p": p, "num_shots": num_shots, "seeds": seeds,
              "lambda_values": lambda_values, "epochs": epochs}
    results = {
        "config": config,
        "summary": summary,
        "per_seed": raw_records,
    }

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Per-seed results in their own files so concurrent runs don't clobber.
    for record in raw_records:
        seed_path = Path(out_dir) / f"gnn_d{d}_p{p}_seed{record['seed']}.json"
        with open(seed_path, "w") as f:
            json.dump({"config": config, "seed": record["seed"],
                       "results": record}, f, indent=2)
        print(f"\nSaved per-seed results to {seed_path}")
    summary_path = Path(out_dir) / f"gnn_d{d}_p{p}_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"config": config, "summary": summary}, f, indent=2)
    print(f"Saved summary to {summary_path}")
    return results


# --------------------------------------------------------------------------- #
# Paired McNemar analysis: GNN-ZX vs GNN-Raw
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _per_shot_correct(model, loader) -> np.ndarray:
    """Boolean array (one entry per shot, in loader order) of (pred == label).
    loader must be shuffle=False so two models align shot-for-shot."""
    model.eval()
    device = next(model.parameters()).device
    out = []
    for data in loader:
        data = data.to(device)
        flip_logit, _ = model(data)
        pred = (torch.sigmoid(flip_logit) > 0.5).long()
        y = data.y.view(-1).long()
        out.append((pred == y).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=bool)


def _train_to_best(model_type: str, num_qubits: int, train_loader, val_loader,
                   epochs: int, lambda_aux: float, device):
    """Train one model and load its best-by-val-loss checkpoint. Returns
    (model, best_val_loss)."""
    model = MODEL_REGISTRY[model_type](num_qubits).to(device)
    config = {"epochs": epochs, "lambda_aux": lambda_aux,
              "pos_weight": None, "model_type": model_type}
    history, best_state = train_model(model, train_loader, val_loader, config)
    model.load_state_dict(best_state)
    best_val = min(history["val_loss"]) if history["val_loss"] else float("inf")
    return model, best_val


def mcnemar_test(b: int, c: int) -> Dict[str, float]:
    """McNemar's test on discordant counts b, c (b + c discordant pairs).

    Returns the continuity-corrected chi-square statistic with its p-value
    (chi-square, df=1) and the exact two-sided binomial p-value (preferred when
    b + c is small). b and c are the two discordant cells.
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0,
                "chi2_cc": 0.0, "p_chi2_cc": 1.0, "p_exact": 1.0}
    # Edwards continuity-corrected chi-square, df = 1.
    chi2 = (abs(b - c) - 1.0) ** 2 / n
    p_chi2 = math.erfc(math.sqrt(chi2 / 2.0))            # survival fn of chi2_1
    # Exact two-sided binomial (n trials, prob 0.5).
    k = min(b, c)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    p_exact = min(1.0, 2.0 * cdf)
    return {"b": b, "c": c, "n_discordant": n,
            "chi2_cc": chi2, "p_chi2_cc": p_chi2, "p_exact": p_exact}


def run_mcnemar_zx_vs_raw(d: int, p: float, num_shots: int, seeds: List[int],
                          lambda_values: List[float], epochs: int = 100,
                          out_dir: str = "results") -> Dict:
    """Paired McNemar comparison of GNN-ZX vs GNN-Raw.

    For each seed: build ONE dataset, train GNN-Raw (lambda=lambda_values[0]) and
    GNN-ZX (best lambda by val loss, as in run_experiment), then score BOTH on
    the identical natural-distribution val shots. Discordant pairs (exactly one
    model correct) are pooled across all seeds and fed to McNemar's test.

    Cell convention: b = Raw correct & ZX wrong; c = ZX correct & Raw wrong.
    c > b favors ZX.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b_total = c_total = both = neither = 0
    per_seed = []

    for seed in seeds:
        print(f"\n=== McNemar seed {seed} (d={d}, p={p}, shots={num_shots}) ===")
        train_loader, val_loader, num_qubits = build_loaders(d, p, num_shots, seed)

        print("  -- training GNN-Raw --")
        raw_model, _ = _train_to_best("Raw", num_qubits, train_loader, val_loader,
                                      epochs, lambda_values[0], device)
        zx_best_model, zx_best_val, zx_best_lam = None, float("inf"), None
        for lam in lambda_values:
            print(f"  -- training GNN-ZX lambda={lam} --")
            m, v = _train_to_best("ZX", num_qubits, train_loader, val_loader,
                                  epochs, lam, device)
            if v < zx_best_val:
                zx_best_val, zx_best_model, zx_best_lam = v, m, lam

        raw_ok = _per_shot_correct(raw_model, val_loader)
        zx_ok = _per_shot_correct(zx_best_model, val_loader)
        assert raw_ok.shape == zx_ok.shape, "paired shots misaligned"

        b = int(np.sum(raw_ok & ~zx_ok))      # Raw correct, ZX wrong
        c = int(np.sum(~raw_ok & zx_ok))      # ZX correct, Raw wrong
        bb = int(np.sum(raw_ok & zx_ok))
        nn_ = int(np.sum(~raw_ok & ~zx_ok))
        b_total += b; c_total += c; both += bb; neither += nn_
        print(f"  seed {seed}: n={raw_ok.size}  both_correct={bb}  neither={nn_}  "
              f"Raw-only(b)={b}  ZX-only(c)={c}  zx_lambda={zx_best_lam}")
        per_seed.append({"seed": seed, "n": int(raw_ok.size), "both_correct": bb,
                         "neither_correct": nn_, "b_raw_only": b, "c_zx_only": c,
                         "zx_lambda": zx_best_lam})

    stats = mcnemar_test(b_total, c_total)
    favored = "ZX" if c_total > b_total else ("Raw" if b_total > c_total else "tie")
    print("\n=== McNemar (ZX vs Raw), pooled across seeds ===")
    print(f"  contingency: both_correct={both}  neither_correct={neither}  "
          f"Raw-only(b)={b_total}  ZX-only(c)={c_total}  discordant={stats['n_discordant']}")
    print(f"  chi2 (cont. corrected, df=1) = {stats['chi2_cc']:.4f}  "
          f"p = {stats['p_chi2_cc']:.4g}")
    print(f"  exact two-sided binomial p   = {stats['p_exact']:.4g}")
    print(f"  favored model: {favored}")

    results = {
        "config": {"d": d, "p": p, "num_shots": num_shots, "seeds": seeds,
                   "lambda_values": lambda_values, "epochs": epochs},
        "comparison": "ZX_vs_Raw",
        "cell_convention": "b = Raw correct & ZX wrong; c = ZX correct & Raw wrong",
        "pooled": {"both_correct": both, "neither_correct": neither, **stats,
                   "favored": favored},
        "per_seed": per_seed,
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"mcnemar_zx_vs_raw_d{d}_p{p}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved McNemar results to {out_path}")
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _dry_run():
    # Budget chosen so the gate (recall>0.3 AND precision>0.3) is MEANINGFUL:
    # precision>0.3 exceeds the ~0.17 base rate, which requires the model to
    # actually learn discrimination. A 2-epoch/500-shot run (~4 grad steps) can
    # only reproduce the prior (precision ~= base rate), so it can't clear that
    # bar regardless of the fix -- it's a training-budget floor, not a bug.
    print("=== DRY RUN: 20 epochs, d=3, p=0.01, 2000 shots, 1 seed, lambda=0.1, "
          "balanced sampler ===")
    device = torch.device("cpu")
    d, p, shots, seed, lam, epochs = 3, 0.01, 2000, 0, 0.1, 20
    train_loader, val_loader, num_qubits = build_loaders(
        d, p, shots, seed, batch_size=128, balanced_sampler=True)
    print(f"num_qubits={num_qubits}  "
          f"train_batches={len(train_loader)}  val_batches={len(val_loader)}")

    # Report best-by-val-loss-checkpoint metrics (what run_experiment uses).
    final = {}
    finite = True
    epochs_ok = True
    for mt in ("A", "Raw", "ZX"):
        print(f"\n--- training GNN_{mt} ---")
        lam_mt = 0.0 if mt == "A" else lam
        res = _train_one(mt, num_qubits, train_loader, val_loader,
                         epochs, lam_mt, None, device)
        hist = res["history"]
        finite = finite and all(
            np.isfinite(hist[k]).all() for k in ("train_loss", "val_loss"))
        epochs_ok = epochs_ok and len(hist["val_loss"]) >= 1
        final[mt] = res["metrics"]

    print("\n=== DRY-RUN VAL METRICS (best checkpoint) ===")
    recall_ok = prec_ok = True
    for mt in ("A", "Raw", "ZX"):
        m = final[mt]
        print(f"GNN_{mt:>3}:  logical_error_rate={m['logical_error_rate']:.4f}  "
              f"recall={m['recall']:.4f}  precision={m['precision']:.4f}")
        if m["recall"] <= 0.3:
            recall_ok = False
        if m["precision"] <= 0.3:
            prec_ok = False
    print(f"\nrecall>0.3 (all 3): {recall_ok}  | precision>0.3 (all 3): {prec_ok}  "
          f"| no NaN losses: {finite}")
    print(f"PASS: {recall_ok and prec_ok and finite}")


def main():
    ap = argparse.ArgumentParser(description="GNN-A / GNN-Raw / GNN-ZX experiment")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--p", type=float, default=0.01)
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.05, 0.1, 0.5, 1.0])
    ap.add_argument("--mcnemar", action="store_true",
                    help="paired McNemar test (GNN-ZX vs GNN-Raw) instead of the full experiment")
    args = ap.parse_args()

    if args.dry_run:
        _dry_run()
        return

    if args.mcnemar:
        run_mcnemar_zx_vs_raw(args.d, args.p, args.shots, args.seeds, args.lambdas,
                              epochs=args.epochs)
        return

    run_experiment(args.d, args.p, args.shots, args.seeds, args.lambdas,
                   epochs=args.epochs)


if __name__ == "__main__":
    main()
