"""Phase 1, task 3 (oversmoothing) and task 4 (empirical boundary share), d=3, p=0.003.

    python analysis/p3_oversmoothing.py --shots 10000 --epochs 150 --tag 10k

Trains the repo's default GNN-A on CPU with qec_run.train (constant lr 1e-3,
early stopping on val loss) on the qec_run split of --shots shots (70/15/15,
split seed 12345), then reports:
  * how good it got: LER on the split's test shots and on 20,000 fresh shots
    (seed 777), each with exact 95% CI, next to MWPM on the same shots and the
    majority ("never flip") baseline;
  * per layer k = 0 (node encoder) .. 4 (after conv k + ReLU), over the fresh
    test shots, in eval mode (no dropout), for the TRAINED model and for the
    model AT INITIALISATION (same seed):
      - mean pairwise cosine similarity within fired detectors, within unfired
        detectors, fired vs unfired, boundary vs fired, boundary vs unfired,
        and over all nodes; fraction of node pairs with cosine > 0.99;
      - Dirichlet energy E_k = mean over directed edges (u, v) of
        ||x_u - x_v||^2, divided by the layer's mean ||x_i||^2 (scale-free);
        per group = edges whose source u is in the group;
      - mean embedding norm per group, and the effective rank (participation
        ratio of singular values) of the node-embedding matrix;
      - linear separability of fired vs unfired detectors from the layer-k
        embedding: balanced accuracy of a class-balanced ridge classifier fit
        on half the shots and scored on the other half;
  * boundary ablation (task 4): embeddings and logits recomputed with every
    edge touching the boundary node removed; per layer, the mean relative
    change ||x'_v - x_v|| / ||x_v|| over detectors, and the fraction of test
    predictions that change.
Writes phase1_results/p3_oversmoothing_<tag>.json, the training log and the
trained weights (gnn_a_d3_<tag>.pt).
"""
import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import binomtest
from torch_geometric.loader import DataLoader

from common import OUT, REPO, save
import qec_run

D, P, SPLIT_SEED, FRESH_SEED, N_FRESH = 3, 0.003, 12345, 777, 20000


def ci(k, n):
    c = binomtest(k, n).proportion_ci(0.95, method="exact")
    return [c.low, c.high]


def layer_embeddings(model, batch, drop_boundary=False):
    """Replicates GNNBackbone.forward in eval mode, returning [x0, x1, .., xL]
    and the flip logit. Asserts the replica equals the model's own output."""
    bb = model.backbone
    ei, ea = batch.edge_index, batch.edge_attr
    if drop_boundary:
        is_b = batch.x[:, 4] == 1
        keep = ~(is_b[ei[0]] | is_b[ei[1]])
        ei, ea = ei[:, keep], ea[keep]
    xs = [bb.node_encoder(batch.x)]
    for i, conv in enumerate(bb.convs):
        x = conv(xs[-1], ei, ea)
        if bb.norms:
            x = bb.norms[i](x)
        xs.append(F.relu(x))
    from gnn_models import _POOLS
    g = torch.cat([pool(xs[-1], batch.batch) for pool in _POOLS[bb.readout]], dim=1)
    logit = model.flip_head(g).view(-1)
    if not drop_boundary:
        ref, _ = model(batch)
        assert torch.allclose(ref, logit, atol=1e-4, rtol=1e-4), "replica mismatch"
    return xs, logit


def cos_matrix(x):
    xn = F.normalize(x, dim=1, eps=1e-12)
    return xn @ xn.T


def mean_offdiag(M):
    n = M.shape[0]
    if n < 2:
        return None
    return float((M.sum() - M.diagonal().sum()) / (n * (n - 1)))


def effective_rank(x):
    s = torch.linalg.svdvals(x - x.mean(0, keepdim=True))
    p = s ** 2 / (s ** 2).sum().clamp_min(1e-30)
    return float(1.0 / (p ** 2).sum())


def group_metrics(xs, batch):
    """Per-layer metrics for ONE graph (batch of one). Returns dict of lists."""
    fired = batch.x[:, 0] == 1
    is_b = batch.x[:, 4] == 1
    unf = (~fired) & (~is_b)
    ei = batch.edge_index
    out = {}
    for k, x in enumerate(xs):
        C = cos_matrix(x)
        norms = x.norm(dim=1)
        mean_sq = float((norms ** 2).mean()).__class__(max(float((norms ** 2).mean()), 1e-30))
        diff = ((x[ei[0]] - x[ei[1]]) ** 2).sum(1)
        src = ei[0]
        rec = {
            "cos_all": mean_offdiag(C),
            "cos_fired_fired": mean_offdiag(C[fired][:, fired]) if fired.sum() >= 2 else None,
            "cos_unfired_unfired": mean_offdiag(C[unf][:, unf]),
            "cos_fired_unfired": float(C[fired][:, unf].mean()) if fired.any() else None,
            "cos_boundary_fired": float(C[is_b][:, fired].mean()) if fired.any() else None,
            "cos_boundary_unfired": float(C[is_b][:, unf].mean()),
            "frac_pairs_cos_gt_0.99": float(((C > 0.99).sum() - C.shape[0])
                                            / (C.shape[0] * (C.shape[0] - 1))),
            "dirichlet_all": float(diff.mean() / mean_sq),
            "dirichlet_fired": float(diff[fired[src]].mean() / mean_sq) if fired.any() else None,
            "dirichlet_unfired": float(diff[unf[src]].mean() / mean_sq),
            "dirichlet_boundary": float(diff[is_b[src]].mean() / mean_sq),
            "norm_fired": float(norms[fired].mean()) if fired.any() else None,
            "norm_unfired": float(norms[unf].mean()),
            "norm_boundary": float(norms[is_b].mean()),
            "effective_rank": effective_rank(x),
        }
        out[k] = rec
    return out


def aggregate(per_shot):
    """Mean over shots of each metric, ignoring None; also the count used."""
    layers = sorted(per_shot[0])
    agg = {}
    for k in layers:
        agg[k] = {}
        for key in per_shot[0][k]:
            vals = [s[k][key] for s in per_shot if s[k][key] is not None]
            agg[k][key] = float(np.mean(vals)) if vals else None
            agg[k][key + "_n"] = len(vals)
    return agg


def probe(features, labels, split):
    """Class-balanced ridge classifier, fit on split==0 shots, balanced accuracy on split==1."""
    X, y = features.numpy().astype(np.float64), labels.numpy().astype(int)
    tr, te = split == 0, split == 1
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Z = (X - mu) / sd
    Ztr, ytr = Z[tr], y[tr]
    w = np.where(ytr == 1, 0.5 / max(ytr.sum(), 1), 0.5 / max((ytr == 0).sum(), 1))
    t = np.where(ytr == 1, 1.0, -1.0)
    A = (Ztr * w[:, None]).T @ Ztr + 1e-3 * np.eye(Z.shape[1])
    beta = np.linalg.solve(A, (Ztr * w[:, None]).T @ t)
    b = -float((Ztr @ beta * w).sum())
    pred = (Z[te] @ beta + b > 0).astype(int)
    yt = y[te]
    tpr = (pred[yt == 1] == 1).mean() if (yt == 1).any() else np.nan
    tnr = (pred[yt == 0] == 0).mean() if (yt == 0).any() else np.nan
    return {"balanced_accuracy": float((tpr + tnr) / 2), "tpr": float(tpr), "tnr": float(tnr),
            "n_fit": int(tr.sum()), "n_score": int(te.sum()),
            "fired_rate_score": float(yt.mean())}


def analyse_model(model, graphs, tag):
    model.eval()
    per_shot, abl, feats, labels, split = [], [], {}, [], []
    preds, preds_nb = [], []
    with torch.no_grad():
        for i, g in enumerate(graphs):
            batch = next(iter(DataLoader([g], batch_size=1)))
            xs, logit = layer_embeddings(model, batch)
            xs_nb, logit_nb = layer_embeddings(model, batch, drop_boundary=True)
            per_shot.append(group_metrics(xs, batch))
            is_b = batch.x[:, 4] == 1
            rel = [float(((a[~is_b] - c[~is_b]).norm(dim=1) / a[~is_b].norm(dim=1).clamp_min(1e-12)).mean())
                   for a, c in zip(xs, xs_nb)]
            abl.append(rel)
            preds.append(int(logit > 0)); preds_nb.append(int(logit_nb > 0))
            for k, x in enumerate(xs):
                feats.setdefault(k, []).append(x[~is_b])
            labels.append(batch.x[~is_b, 0])
            split.append(np.full(int((~is_b).sum()), i % 2))
    labels = torch.cat(labels)
    split = np.concatenate(split)
    probes = {k: probe(torch.cat(v), labels, split) for k, v in feats.items()}
    abl = np.array(abl)
    return {"tag": tag,
            "layers": aggregate(per_shot),
            "linear_probe_fired_vs_unfired": probes,
            "boundary_ablation": {
                "mean_rel_change_detector_embeddings_per_layer": abl.mean(0).tolist(),
                "frac_predictions_changed": float(np.mean(np.array(preds) != np.array(preds_nb))),
                "frac_predict_flip_with_boundary": float(np.mean(preds)),
                "frac_predict_flip_without_boundary": float(np.mean(preds_nb))}}


ap = argparse.ArgumentParser()
ap.add_argument("--shots", type=int, default=10000)
ap.add_argument("--epochs", type=int, default=150)
ap.add_argument("--patience", type=int, default=20)
ap.add_argument("--tag", default=None)
ap.add_argument("--n-analyse", type=int, default=4000, help="fresh test shots used for metrics")
a = ap.parse_args()
tag = a.tag or f"{a.shots // 1000}k"
OUT.mkdir(exist_ok=True)
torch.set_num_threads(4)

train_ds, val_ds, test_ds = qec_run.make_datasets(D, P, a.shots, SPLIT_SEED)
fresh = qec_run._fresh_shots(D, P, N_FRESH, FRESH_SEED)
qec_run._seed_all(0)
model = qec_run.build_model("A", d=D)
init_state = {k: v.clone() for k, v in model.state_dict().items()}
n_params = sum(t.numel() for t in model.parameters())

log_path = OUT / f"train_log_{tag}.txt"
t0 = time.perf_counter()
with open(log_path, "w") as log, contextlib.redirect_stdout(log):
    model, info = qec_run.train(model, train_ds, val_ds, lambda_aux=0.0, epochs=a.epochs,
                                seed=0, device="cpu", patience=a.patience)
train_s = time.perf_counter() - t0
torch.save(model.state_dict(), OUT / f"gnn_a_d3_{tag}.pt")


def evaluate(ds, name):
    pred, true = qec_run.predict_flips(model, ds, "cpu")
    k = int((pred != true).sum())
    mw_ler, mw_ok, _ = qec_run.mwpm_on_test(D, P, ds)
    km = int((~mw_ok).sum())
    maj = int(true.sum())          # "never flip" errors = number of flips
    return {"set": name, "n": len(ds), "gnn_errors": k, "gnn_ler": k / len(ds), "gnn_ci95": ci(k, len(ds)),
            "mwpm_errors": km, "mwpm_ler": mw_ler, "mwpm_ci95": ci(km, len(ds)),
            "never_flip_errors": maj, "never_flip_ler": maj / len(ds),
            "gnn_pred_flip_rate": float(pred.mean()), "true_flip_rate": float(true.mean())}


results = {"d": D, "p": P, "shots_pool": a.shots, "n_train": len(train_ds), "n_val": len(val_ds),
           "n_test": len(test_ds), "split_seed": SPLIT_SEED, "model_seed": 0, "n_params": n_params,
           "training": {"epochs_run": info["epochs_run"], "best_epoch": info["best_epoch"],
                        "patience": a.patience, "epochs_max": a.epochs, "wall_s": train_s,
                        "train_loss_final": info["train_loss"][-1], "val_loss_final": info["val_loss"][-1],
                        "val_loss_best": min(info["val_loss"]), "lr": info["optim"]["lr"],
                        "train_loss": info["train_loss"], "val_loss": info["val_loss"]},
           "quality": [evaluate(test_ds, "split test"), evaluate(fresh, f"fresh {N_FRESH} (seed {FRESH_SEED})")]}
for q in results["quality"]:
    print(f"{q['set']}: GNN {q['gnn_errors']}/{q['n']} = {q['gnn_ler']:.4f} {q['gnn_ci95']}, "
          f"MWPM {q['mwpm_errors']} = {q['mwpm_ler']:.4f}, never-flip {q['never_flip_errors']}; "
          f"GNN predicts flip on {q['gnn_pred_flip_rate']:.4f} of shots")
print(f"training: {info['epochs_run']} epochs, best {info['best_epoch']}, {train_s:.0f} s")

graphs = fresh[:a.n_analyse]
results["n_analysed_shots"] = len(graphs)
results["fired_per_shot_analysed"] = float(np.mean([float(g.x[:, 0].sum()) for g in graphs]))
results["trained"] = analyse_model(model, graphs, "trained")
model.load_state_dict(init_state)
results["init"] = analyse_model(model, graphs, "init")
for which in ("init", "trained"):
    print(f"--- {which} ---")
    for k, m in results[which]["layers"].items():
        pr = results[which]["linear_probe_fired_vs_unfired"][k]["balanced_accuracy"]
        print(f"  layer {k}: cos all {m['cos_all']:.3f} | fired-fired {m['cos_fired_fired']} (n={m['cos_fired_fired_n']}) "
              f"| unf-unf {m['cos_unfired_unfired']:.3f} | fired-unf {m['cos_fired_unfired']} | bnd-unf {m['cos_boundary_unfired']:.3f} "
              f"| >0.99 {m['frac_pairs_cos_gt_0.99']:.3f} | Dirichlet all {m['dirichlet_all']:.3g} fired {m['dirichlet_fired']} "
              f"unf {m['dirichlet_unfired']:.3g} bnd {m['dirichlet_boundary']:.3g} | norms f/u/b "
              f"{m['norm_fired']}/{m['norm_unfired']:.3g}/{m['norm_boundary']:.3g} | eff.rank {m['effective_rank']:.2f} | probe bal.acc {pr:.3f}")
    print("  boundary ablation:", results[which]["boundary_ablation"])
save(f"p3_oversmoothing_{tag}.json", results)
