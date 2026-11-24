"""Oversmoothing in TRAINED d=5 models: default GNN vs residual + LayerNorm.

    python analysis/p4_trained_smoothing.py               # 4,000 shots, writes the JSON
    python analysis/p4_trained_smoothing.py --n-shots 500 # quick check

Question (C2 in the paper plan): the Phase 1 report measured oversmoothing at
initialisation (d=3: cosine 0.997 at layer 4) and on one trained d=3 default model.
Residual + LayerNorm was motivated by it. Do the TRAINED d=5 models differ the way
that motivation predicts: default node embeddings close to one direction by layer 4,
residual + LayerNorm ones not?

Models (saved by qec_run.py, reloaded on CPU, eval mode, so no dropout):
  base_35k   default GNN, recipe v2, fixed 35k training set, seeds 0 1 2
  resln_35k  residual + LayerNorm, same data and recipe, seeds 0 1 2
  final      residual + LayerNorm, fresh 300k shots every epoch, seeds 0 1 2
  init_base / init_resln: the same two architectures before training, built
             exactly as run_arm builds them (_seed_all(seed), then build_model).
             The CUDA run built its models on CPU from the same torch seed, so
             these should be its starting weights; this is not checked.
Shots: the first --n-shots shots of the stored eval set (the big test).

Per layer k = 0 (node encoder output) .. 4, on x_k, the node states the next layer
(or the readout, for k = 4) receives. Default: x_k = relu(conv_k(x_{k-1})).
Residual + LayerNorm: x_k = x_{k-1} + relu(LN(conv_k(x_{k-1}))).
Metrics, as in analysis/p3_oversmoothing.py (Phase 1) so the numbers compare:
  - mean pairwise cosine over all node pairs, fired-fired, unfired-unfired,
    fired-unfired, boundary-unfired; fraction of pairs with cosine > 0.99;
  - Dirichlet energy over the graph's edges divided by the mean squared norm;
  - mean norm of fired / unfired detectors and of the boundary node;
  - effective rank (participation ratio of singular values of the centred
    121 x 96 node matrix);
  - linear probe: can a class-balanced ridge classifier read the fired bit of a
    detector from x_k? Balanced accuracy, fit on even shots, scored on odd ones,
    on the first --n-probe shots.
Each metric is computed per shot, then averaged over shots (pairs that do not exist
in a shot, e.g. fired-fired with < 2 fired detectors, are skipped; counts recorded).
The script checks that its layer-by-layer replica gives the model's own logits.

Writes analysis/phase3_results/p4_trained_smoothing.json.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

import qec_run  # noqa: E402
from gnn_models import _POOLS  # noqa: E402

D, P = 5, 0.003
EVAL_SET = "data/eval_d5_p0.003_seed12347_200k.npz"
RUNS = {
    "base_35k": "results/run_20261124T074644Z_d5_p0.003.json",
    "resln_35k": "results/run_20261124T063211Z_d5_p0.003.json",
    "final": "results/run_20261123T232850Z_d5_p0.003.json",
}
SEEDS = (0, 1, 2)
BATCH = 250


def weights_path(run_json, seed):
    f = REPO / run_json
    return f.with_name(f.stem + "_weights") / f"A_seed{seed}.pt"


def load_trained(run_json, seed):
    ck = torch.load(weights_path(run_json, seed), weights_only=False)
    assert ck["arm"] == "A" and ck["seed"] == seed and ck["run_json"] == Path(run_json).name
    hp = {k: v for k, v in ck["model"].items() if k != "n_params"}
    model = qec_run.build_model("A", d=D, **hp)
    model.load_state_dict(ck["state_dict"])
    return model.eval(), hp


def build_init(hp, seed):
    qec_run._seed_all(seed)                      # as run_arm: seed, then build
    return qec_run.build_model("A", d=D, **hp).eval()


def layer_states(model, batch):
    """[x_0, .., x_L] as the backbone computes them in eval mode, and the flip logit."""
    bb = model.backbone
    xs = [bb.node_encoder(batch.x)]
    for i, conv in enumerate(bb.convs):
        h = conv(xs[-1], batch.edge_index, batch.edge_attr)
        if bb.norms:
            h = bb.norms[i](h)
        h = F.relu(h)
        xs.append(xs[-1] + h if bb.residual else h)
    g = torch.cat([pool(xs[-1], batch.batch) for pool in _POOLS[bb.readout]], dim=1)
    logit = model.flip_head(g).view(-1)
    ref, _ = model(batch)
    assert torch.allclose(ref, logit, atol=1e-4, rtol=1e-4), "replica mismatch"
    return xs, logit


def pair_mean(C, a, b, same):
    """Mean cosine over pairs (i in a, j in b), per shot; i != j when same. NaN if none."""
    m = a[:, :, None] & b[:, None, :]
    if same:
        m = m & ~torch.eye(C.shape[1], dtype=torch.bool)[None]
    n = m.sum((1, 2)).double()
    s = (C.double() * m).sum((1, 2))
    return torch.where(n > 0, s / n.clamp_min(1), torch.full_like(s, float("nan")))


def group_mean(v, mask):
    n = mask.sum(1).double()
    s = (v.double() * mask).sum(1)
    return torch.where(n > 0, s / n.clamp_min(1), torch.full_like(s, float("nan")))


def shot_metrics(X, fired, bnd, src, dst):
    """X: (B, N, H) node states of B shots. Returns {metric: (B,) tensor}."""
    B, N, _ = X.shape
    unf = ~fired & ~bnd[None]
    bndB = bnd[None].expand(B, N)
    allB = torch.ones(B, N, dtype=torch.bool)
    Xn = F.normalize(X, dim=2, eps=1e-12)
    C = Xn @ Xn.transpose(1, 2)
    off = ~torch.eye(N, dtype=torch.bool)[None]
    norms = X.norm(dim=2)
    mean_sq = (norms ** 2).mean(1).clamp_min(1e-30)
    Xc = X - X.mean(1, keepdim=True)
    s = torch.linalg.svdvals(Xc.double())
    p = s ** 2 / (s ** 2).sum(1, keepdim=True).clamp_min(1e-30)
    return {
        "cos_all": pair_mean(C, allB, allB, True),
        "cos_fired_fired": pair_mean(C, fired, fired, True),
        "cos_unfired_unfired": pair_mean(C, unf, unf, True),
        "cos_fired_unfired": pair_mean(C, fired, unf, False),
        "cos_boundary_unfired": pair_mean(C, bndB, unf, False),
        "frac_pairs_cos_gt_0.99": ((C > 0.99) & off).sum((1, 2)).double() / (N * (N - 1)),
        "dirichlet": ((X[:, src] - X[:, dst]) ** 2).sum(2).mean(1).double() / mean_sq.double(),
        "norm_fired": group_mean(norms, fired),
        "norm_unfired": group_mean(norms, unf),
        "norm_boundary": group_mean(norms, bndB),
        "effective_rank": 1.0 / (p ** 2).sum(1),
    }


def probe(X, y, split):
    """Class-balanced ridge classifier (as Phase 1): fit split==0, balanced accuracy on 1."""
    X, y = X.astype(np.float64), y.astype(int)
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
    tpr = (pred[yt == 1] == 1).mean()
    tnr = (pred[yt == 0] == 0).mean()
    return {"balanced_accuracy": float((tpr + tnr) / 2), "tpr": float(tpr), "tnr": float(tnr),
            "n_fit": int(tr.sum()), "n_score": int(te.sum()), "n_fired_score": int(yt.sum())}


def analyse(model, shots, n_probe, src, dst, bnd_idx):
    N = shots[0].num_nodes
    bnd = torch.zeros(N, dtype=torch.bool)
    bnd[bnd_idx] = True
    per = {}
    feats, labels, split = {}, [], []
    logits = []
    with torch.no_grad():
        for start in range(0, len(shots), BATCH):
            chunk = shots[start:start + BATCH]
            batch = Batch.from_data_list(chunk)
            xs, logit = layer_states(model, batch)
            logits.append(logit)
            B = len(chunk)
            fired = (batch.x[:, 0] > 0.5).view(B, N)
            for k, x in enumerate(xs):
                X = x.view(B, N, -1)
                for name, v in shot_metrics(X, fired, bnd, src, dst).items():
                    per.setdefault(k, {}).setdefault(name, []).append(v)
                if start < n_probe:
                    keep = min(B, n_probe - start)
                    feats.setdefault(k, []).append(X[:keep][:, ~bnd].reshape(-1, X.shape[2]).numpy())
            if start < n_probe:
                keep = min(B, n_probe - start)
                labels.append(fired[:keep][:, ~bnd].reshape(-1).numpy())
                idx = np.arange(start, start + keep)
                split.append(np.repeat(idx % 2, int((~bnd).sum())))
    labels, split = np.concatenate(labels), np.concatenate(split)
    layers = {}
    for k in sorted(per):
        layers[k] = {}
        for name, chunks in per[k].items():
            v = torch.cat(chunks).numpy()
            ok = ~np.isnan(v)
            layers[k][name] = float(v[ok].mean()) if ok.any() else None
            layers[k][name + "_n"] = int(ok.sum())
        layers[k]["probe_fired_bit"] = probe(np.concatenate(feats[k]), labels, split)
    return layers, torch.cat(logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shots", type=int, default=4000)
    ap.add_argument("--n-probe", type=int, default=1000)
    ap.add_argument("--out", default=str(HERE / "phase3_results" / "p4_trained_smoothing.json"))
    a = ap.parse_args()
    torch.set_num_threads(8)
    t0 = time.perf_counter()

    dets, obs, errs, meta, file_sha, checks = qec_run.load_eval_set_arrays(REPO / EVAL_SET, D, P)
    n = a.n_shots
    shots = qec_run._shots_from_samples(D, P, dets[:n], obs[:n], errs[:n])
    g0 = shots[0]
    for s in shots:                                      # one fixed graph, features differ
        assert s.num_nodes == g0.num_nodes and torch.equal(s.edge_index, g0.edge_index)
    bnd_idx = torch.nonzero(g0.x[:, 4] > 0.5).view(-1)
    assert len(bnd_idx) == 1
    src, dst = g0.edge_index
    fired_per_shot = float(np.mean([float(s.x[:, 0].sum()) for s in shots]))
    y = np.asarray(obs[:n, 0], dtype=int)

    results = {
        "what": "per-layer smoothing metrics of trained and initial d=5 models",
        "eval_set": {"path": EVAL_SET, "sha256": file_sha, "checks": checks},
        "n_shots": n, "n_probe_shots": a.n_probe,
        "mean_fired_detectors_per_shot": fired_per_shot,
        "zero_fired_shots": int(sum(float(s.x[:, 0].sum()) == 0 for s in shots)),
        "num_nodes": int(g0.num_nodes), "num_directed_edges": int(g0.edge_index.shape[1]),
        "boundary_index": int(bnd_idx[0]),
        "models": {},
    }
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=REPO, capture_output=True, text=True).stdout.strip())
    except OSError:
        commit, dirty = None, None
    results["provenance"] = {"git_commit": commit, "git_dirty_tracked": dirty,
                             "script": "analysis/p4_trained_smoothing.py",
                             "torch": torch.__version__, "argv": sys.argv}

    hps = {}
    for label, run_json in RUNS.items():
        for seed in SEEDS:
            model, hp = load_trained(run_json, seed)
            hps[label] = hp
            layers, logit = analyse(model, shots, a.n_probe, src, dst, bnd_idx)
            wrong_idx = np.flatnonzero((logit > 0).numpy().astype(int) != y).tolist()
            wrong = len(wrong_idx)
            rj = json.loads((REPO / run_json).read_text(encoding="utf-8"))
            ref = [i for i in rj["big_test"]["arms"]["A"][str(seed)]["test_wrong_idx"] if i < n]
            # CPU here vs GPU in the run: a shot whose logit sits at ~0 could flip.
            same = wrong_idx == ref
            print(f"  {label} seed {seed}: {wrong} wrong on these shots, run JSON says "
                  f"{len(ref)}, identical list: {same}")
            wf = weights_path(run_json, seed)
            results["models"][f"{label}_seed{seed}"] = {
                "kind": label, "seed": seed, "trained": True, "run_json": run_json,
                "weights": str(wf.relative_to(REPO)).replace("\\", "/"),
                "weights_sha256": hashlib.sha256(wf.read_bytes()).hexdigest(),
                "model": hp, "errors_on_these_shots": wrong,
                "errors_in_run_json_on_these_shots": len(ref),
                "wrong_list_identical_to_run_json": same, "layers": layers}
            print(f"{label} seed {seed}: layer 4 cos_all "
                  f"{layers[4]['cos_all']:.3f}, eff.rank {layers[4]['effective_rank']:.2f}, "
                  f"probe {layers[4]['probe_fired_bit']['balanced_accuracy']:.3f} "
                  f"({time.perf_counter() - t0:.0f} s)")
    for label, hp_from in (("init_base", "base_35k"), ("init_resln", "resln_35k")):
        for seed in SEEDS:
            model = build_init(hps[hp_from], seed)
            layers, logit = analyse(model, shots, a.n_probe, src, dst, bnd_idx)
            wrong = int(((logit > 0).numpy().astype(int) != y).sum())
            results["models"][f"{label}_seed{seed}"] = {
                "kind": label, "seed": seed, "trained": False, "model": hps[hp_from],
                "errors_on_these_shots": wrong, "layers": layers}
            print(f"{label} seed {seed}: layer 4 cos_all {layers[4]['cos_all']:.3f}, "
                  f"eff.rank {layers[4]['effective_rank']:.2f}, "
                  f"probe {layers[4]['probe_fired_bit']['balanced_accuracy']:.3f}")
    results["wall_s"] = time.perf_counter() - t0
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {out} in {results['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
