#!/usr/bin/env python3
"""
Neural decoder training with ZX knowledge distillation.

Models
  BaselineA      – syndrome bits → MLP → logical flip
  BaselineB      – ZX features  → MLP → logical flip
  ProposedModel  – syndrome bits → shared backbone → flip head + ZX reconstruction head

Usage
  python train.py --dry-run              # 5 epochs, d=3, prints val metrics
  python train.py --d 3                  # full experiment for d=3
  python train.py                        # full sweep d=3,5,7
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Data loading ─────────────────────────────────────────────────────────────

class QECDataset(Dataset):
    """Wraps one .npz file; returns (syndrome_bits, zx_onehot, zx_int, logical_flip)."""

    def __init__(self, path):
        data = np.load(path)
        self.syndrome = torch.tensor(data["syndrome_bits"], dtype=torch.float32)
        self.flip     = torch.tensor(data["logical_flip"],   dtype=torch.float32)
        self.p        = float(Path(path).stem.split("_p")[1])

        self.zx_int = torch.tensor(data["zx_features"], dtype=torch.long)  # (n, d*d)
        n, m = self.zx_int.shape
        onehot = torch.zeros(n, m, 4, dtype=torch.float32)
        onehot.scatter_(2, self.zx_int.unsqueeze(2), 1.0)
        self.zx_onehot = onehot.reshape(n, m * 4)                          # (n, d*d*4)

    def __len__(self):
        return len(self.flip)

    def __getitem__(self, idx):
        return self.syndrome[idx], self.zx_onehot[idx], self.zx_int[idx], self.flip[idx]


def load_datasets(d: int, data_dir: str = "data", batch_size: int = 512):
    """
    Load all d{d}_p*.npz files, split each 80/10/10, and return:
      train_loader  : all p values shuffled together
      val_loader    : all p values together
      test_loaders  : {p_float: DataLoader}  one per error rate
      n_syndrome    : syndrome vector width
      n_zx          : ZX feature vector width
    """
    files = sorted(Path(data_dir).glob(f"d{d}_p*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found for d={d} in '{data_dir}'. "
                                f"Run generate_dataset.py first.")

    train_splits, val_splits = [], []
    test_loaders = {}
    n_syndrome = n_zx = None

    for f in files:
        ds = QECDataset(f)
        if n_syndrome is None:
            n_syndrome = ds.syndrome.shape[1]
            n_zx       = ds.zx_onehot.shape[1]

        n       = len(ds)
        n_train = int(0.8 * n)
        n_val   = int(0.1 * n)
        n_test  = n - n_train - n_val

        train_ds, val_ds, test_ds = random_split(
            ds, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )
        train_splits.append(train_ds)
        val_splits.append(val_ds)
        test_loaders[ds.p] = DataLoader(test_ds, batch_size=batch_size)

    train_loader = DataLoader(ConcatDataset(train_splits), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(ConcatDataset(val_splits),   batch_size=batch_size)

    log.info("d=%d  files=%d  n_syndrome=%d  n_zx=%d", d, len(files), n_syndrome, n_zx)
    return train_loader, val_loader, test_loaders, n_syndrome, n_zx


# ─── Models ───────────────────────────────────────────────────────────────────

def _backbone(n_in: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_in, 256), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.1),
    )


class BaselineA(nn.Module):
    """Syndrome bits → flip logit."""

    def __init__(self, n_syndrome: int):
        super().__init__()
        self.net = nn.Sequential(_backbone(n_syndrome), nn.Linear(128, 1))

    def forward(self, syndrome, zx=None):
        return self.net(syndrome).squeeze(-1)


class BaselineB(nn.Module):
    """ZX features → flip logit."""

    def __init__(self, n_zx: int):
        super().__init__()
        self.net = nn.Sequential(_backbone(n_zx), nn.Linear(128, 1))

    def forward(self, syndrome, zx):
        return self.net(zx).squeeze(-1)


class ProposedModel(nn.Module):
    """
    Syndrome bits → shared backbone → two heads:
      flip_head : Linear(128,1)     — logical flip logit (no sigmoid; use BCEWithLogitsLoss)
      zx_head   : Linear(128, n_zx) — ZX feature reconstruction (auxiliary)
    """

    def __init__(self, n_syndrome: int, n_zx: int):
        super().__init__()
        self.backbone  = _backbone(n_syndrome)
        self.flip_head = nn.Linear(128, 1)
        self.zx_head   = nn.Linear(128, n_zx)

    def forward(self, syndrome, zx=None):
        h = self.backbone(syndrome)
        flip_pred = self.flip_head(h).squeeze(-1)               # raw logit
        zx_logits = self.zx_head(h).reshape(h.shape[0], -1, 4)  # (batch, d*d, 4)
        return flip_pred, zx_logits


# ─── Training ─────────────────────────────────────────────────────────────────

def _forward_and_loss(model, syndrome, zx_onehot, zx_int, flip, bce, ce, lam):
    if isinstance(model, ProposedModel):
        flip_pred, zx_pred = model(syndrome)            # zx_pred: (batch, d*d, 4)
        zx_pred_flat = zx_pred.reshape(-1, 4)           # (batch*d*d, 4)
        zx_target    = zx_int.reshape(-1)               # (batch*d*d,) long
        n_qubits     = zx_int.shape[1]
        return bce(flip_pred, flip) + lam * ce(zx_pred_flat, zx_target) / n_qubits, flip_pred
    flip_pred = model(syndrome, zx_onehot)
    return bce(flip_pred, flip), flip_pred


def train_model(model: nn.Module, dataloaders: dict, config: dict) -> list:
    """
    Train with Adam, BCE loss (+ lambda*MSE for ProposedModel).
    Early stopping on val loss with patience.

    config keys:
      lr         float  (default 1e-3)
      epochs     int    (default 50)
      patience   int    (default 10)
      lambda_zx  float  ProposedModel auxiliary weight (default 0.1)

    Returns list of per-epoch dicts with train_loss, val_loss, val_recall, val_precision.
    """
    train_loader = dataloaders["train"]
    val_loader   = dataloaders["val"]
    model        = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))

    # pos_weight=1.0 (plain BCE). The logical error rate weighs false positives
    # and false negatives EQUALLY, so the Bayes-optimal rule is to estimate
    # P(flip|syndrome) and threshold at 0.5 -- which plain BCE learns. A
    # class-balancing pos_weight (~neg/pos) instead optimizes a recall-weighted
    # objective, making the model over-predict flips and inflating LER.
    pos_weight = torch.tensor([1.0]).to(DEVICE)

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ce  = nn.CrossEntropyLoss()
    lam      = config.get("lambda_zx", 0.1)
    n_epochs = config.get("epochs",    50)
    patience = config.get("patience",  10)

    best_val  = float("inf")
    best_state = None
    no_improve = 0
    history    = []

    for epoch in range(1, n_epochs + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        running = 0.0
        for syndrome, zx_onehot, zx_int, flip in train_loader:
            syndrome  = syndrome.to(DEVICE)
            zx_onehot = zx_onehot.to(DEVICE)
            zx_int    = zx_int.to(DEVICE)
            flip      = flip.to(DEVICE)
            optimizer.zero_grad()
            loss, _ = _forward_and_loss(model, syndrome, zx_onehot, zx_int, flip, bce, ce, lam)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(flip)
        train_loss = running / len(train_loader.dataset)

        # ── validate ───────────────────────────────────────────────────────
        model.eval()
        running = 0.0
        preds_list, targets_list = [], []
        with torch.no_grad():
            for syndrome, zx_onehot, zx_int, flip in val_loader:
                syndrome  = syndrome.to(DEVICE)
                zx_onehot = zx_onehot.to(DEVICE)
                zx_int    = zx_int.to(DEVICE)
                flip      = flip.to(DEVICE)
                loss, flip_pred = _forward_and_loss(model, syndrome, zx_onehot, zx_int, flip, bce, ce, lam)
                running += loss.item() * len(flip)
                preds_list.append((flip_pred > 0.0).float().cpu())
                targets_list.append(flip.cpu())

        val_loss = running / len(val_loader.dataset)
        preds    = torch.cat(preds_list)
        targets  = torch.cat(targets_list)

        tp = ((preds == 1) & (targets == 1)).sum().item()
        fp = ((preds == 1) & (targets == 0)).sum().item()
        fn = ((preds == 0) & (targets == 1)).sum().item()
        recall    = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)

        log.info("Epoch %3d/%d  train=%.4f  val=%.4f  recall=%.3f  prec=%.3f",
                 epoch, n_epochs, train_loss, val_loss, recall, precision)

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_recall": recall, "val_precision": precision,
        })

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("Early stopping at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    return history


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, test_loaders: dict) -> dict:
    """
    Evaluate on each per-p test set.

    Returns {p_str: {logical_error_rate, recall, precision}} sorted by p.
    """
    model = model.to(DEVICE)
    model.eval()
    results = {}

    with torch.no_grad():
        for p, loader in sorted(test_loaders.items()):
            preds_list, targets_list = [], []
            for syndrome, zx_onehot, _, flip in loader:
                syndrome  = syndrome.to(DEVICE)
                zx_onehot = zx_onehot.to(DEVICE)
                flip      = flip.to(DEVICE)
                if isinstance(model, ProposedModel):
                    flip_pred, _ = model(syndrome)
                else:
                    flip_pred = model(syndrome, zx_onehot)
                preds_list.append((flip_pred > 0.0).float().cpu())
                targets_list.append(flip.cpu())

            preds   = torch.cat(preds_list)
            targets = torch.cat(targets_list)
            n  = len(targets)
            tp = ((preds == 1) & (targets == 1)).sum().item()
            fp = ((preds == 1) & (targets == 0)).sum().item()
            fn = ((preds == 0) & (targets == 1)).sum().item()

            results[str(p)] = {
                "logical_error_rate": (fp + fn) / n,
                "recall":             tp / (tp + fn + 1e-8),
                "precision":          tp / (tp + fp + 1e-8),
            }

    return results


# ─── Experiment runner ────────────────────────────────────────────────────────

def run_experiment(
    d:           int,
    data_dir:    str = "data",
    results_dir: str = "results",
    n_epochs:    int = 50,
) -> dict:
    """Train all models for distance d and save d{d}_results.json."""
    Path(results_dir).mkdir(exist_ok=True)
    train_loader, val_loader, test_loaders, n_syndrome, n_zx = load_datasets(d, data_dir)
    loaders = {"train": train_loader, "val": val_loader}
    base_cfg = {"epochs": n_epochs, "patience": 10}
    all_results = {}

    log.info("=== BaselineA  d=%d ===", d)
    model_a = BaselineA(n_syndrome)
    train_model(model_a, loaders, base_cfg)
    all_results["BaselineA"] = evaluate(model_a, test_loaders)

    log.info("=== BaselineB  d=%d ===", d)
    model_b = BaselineB(n_zx)
    train_model(model_b, loaders, base_cfg)
    all_results["BaselineB"] = evaluate(model_b, test_loaders)

    log.info("=== ProposedModel lambda sweep  d=%d ===", d)
    lambda_sweep  = {}
    best_lam      = None
    best_val_loss = float("inf")
    best_model_p  = None

    for lam in (0.01, 0.1, 0.5, 1.0):
        log.info("--- lambda=%.2f  d=%d ---", lam, d)
        model_p = ProposedModel(n_syndrome, n_zx)
        history = train_model(model_p, loaders, {**base_cfg, "lambda_zx": lam})
        min_val = min(h["val_loss"] for h in history)
        lambda_sweep[str(lam)] = {
            "best_val_loss": min_val,
            "test": evaluate(model_p, test_loaders),
        }
        if min_val < best_val_loss:
            best_val_loss = min_val
            best_lam      = lam
            best_model_p  = model_p

    all_results["ProposedModel"]              = evaluate(best_model_p, test_loaders)
    all_results["ProposedModel_lambda_sweep"] = lambda_sweep
    all_results["best_lambda"]                = best_lam

    out = Path(results_dir) / f"d{d}_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Results saved to %s", out)
    return all_results


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QEC neural decoder training")
    parser.add_argument("--d",           type=int, default=None, help="Single distance (runs all if omitted)")
    parser.add_argument("--dry-run",     action="store_true",    help="5 epochs on d=3, print val metrics")
    parser.add_argument("--data-dir",    default="data",         help="Directory with .npz files")
    parser.add_argument("--results-dir", default="results",      help="Output directory for JSON results")
    args = parser.parse_args()

    if args.dry_run:
        train_loader, val_loader, test_loaders, n_syndrome, n_zx = load_datasets(
            3, data_dir=args.data_dir
        )
        loaders = {"train": train_loader, "val": val_loader}
        log.info("Dry run: ProposedModel  d=3  5 epochs  lambda=0.1  device=%s", DEVICE)
        model   = ProposedModel(n_syndrome, n_zx)
        history = train_model(model, loaders, {"epochs": 5, "patience": 10, "lambda_zx": 0.1})
        print("\nFinal epoch metrics:", history[-1])
        print("\nTest metrics per p:")
        for p_str, m in evaluate(model, test_loaders).items():
            print(f"  p={p_str}:  logical_error_rate={m['logical_error_rate']:.4f}"
                  f"  recall={m['recall']:.3f}  precision={m['precision']:.3f}")
    elif args.d is not None:
        run_experiment(args.d, data_dir=args.data_dir, results_dir=args.results_dir)
    else:
        for d in (3, 5, 7):
            run_experiment(d, data_dir=args.data_dir, results_dir=args.results_dir)
