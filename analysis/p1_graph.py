"""Phase 1, tasks 1, 2 (theory), 4 (structure) and 5, at d=3 and d=5, p=0.003, rounds=d.

    python analysis/p1_graph.py   ->  analysis/phase1_results/p1_graph.json

1. Size, degree distribution, boundary-node degree, diameter (hops) with the
   boundary node and with it removed (components reported if disconnected).
2. Minimal logical operator on the decoding graph: the shortest closed walk
   through the boundary node with odd observable parity, found on the
   parity-doubled graph (v, parity). Reported in hops (fewest edges) and for the
   minimum-weight one (most likely logical, raw weights -log(p/(1-p))).
4. Structural share of a node's layer-k receptive field that passes through the
   boundary node: GINEConv aggregates (1+eps) x_i + sum_j msg_j with eps = 0 at
   init, so walk counts of W = A + I bound how inputs mix. Fraction of length-k
   walks ending at v that visit the boundary node at least once.
5. Fraction of shots with zero fired detectors (1M shots, seeded), with exact
   95% CI, plus the observable flip rate among those shots.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra, shortest_path
from scipy.stats import binomtest

from common import Graph, save


def diameter(A):
    D = shortest_path(A, unweighted=True, directed=False)
    finite = D[np.isfinite(D)]
    return int(finite.max()), D


def logical(G):
    """Shortest odd-parity closed walk through the boundary node."""
    N, b = G.N, G.b
    rows, cols, hops, wts = [], [], [], []
    for (u, v), f, w in zip(G.und, G.flips, G.w_raw):
        for par in (0, 1):
            a, c = u + par * N, v + (par ^ f) * N
            rows += [a, c]; cols += [c, a]; wts += [w, w]
    H = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(2 * N, 2 * N)).tocsr()
    Wm = sp.coo_matrix((wts, (rows, cols)), shape=(2 * N, 2 * N)).tocsr()
    dh = dijkstra(H, indices=b, unweighted=True)
    dw, pred = dijkstra(Wm, indices=b, return_predecessors=True)
    # walk back the min-weight path from (b, 1) to (b, 0)
    path, n = [], b + N
    while n != b and n >= 0:
        path.append(n); n = pred[n]
    path.append(b)
    nodes = [x % N for x in reversed(path)]
    return {"min_hops": int(dh[b + N]), "min_weight": float(dw[b + N]),
            "min_weight_path_hops": len(nodes) - 1,
            "min_weight_path_detectors": len(nodes) - 2,
            "min_weight_path_boundary_edges": sum(1 for x in nodes[1:-1] if x == b) + 2}


def walk_share(G, kmax=8):
    W = (G.A + sp.eye(G.N)).toarray()
    keep = np.arange(G.N) != G.b
    Wb = W[np.ix_(keep, keep)]
    tot, avo = np.ones(G.N), np.ones(G.N - 1)
    adj_b = np.zeros(G.N, bool)
    adj_b[G.A[G.b].indices] = True
    adj_b = adj_b[keep]
    out = []
    for k in range(1, kmax + 1):
        tot, avo = W @ tot, Wb @ avo          # walks of length k ending at each node
        share = 1 - avo / tot[keep]
        out.append({"k": k, "mean_all_detectors": float(share.mean()),
                    "mean_boundary_adjacent": float(share[adj_b].mean()),
                    "mean_not_boundary_adjacent": float(share[~adj_b].mean())
                    if (~adj_b).any() else None,
                    "min": float(share.min()), "max": float(share.max())})
    return out, int(adj_b.sum())


def zero_fired(G, shots=1_000_000, seed=2026):
    dem = G.circ.detector_error_model(decompose_errors=False, flatten_loops=True)
    dets, obs, _ = dem.compile_sampler(seed=seed).sample(shots=shots)
    z = ~dets.any(axis=1)
    k = int(z.sum())
    ci = binomtest(k, shots).proportion_ci(0.95, method="exact")
    kz1 = int(obs[z, 0].sum())
    p_err = np.array([inst.args_copy()[0] for inst in dem.flattened() if inst.type == "error"])
    return {"shots": shots, "seed": seed, "zero_fired": k, "fraction": k / shots,
            "ci95": [ci.low, ci.high],
            "zero_fired_with_obs_flip": kz1,
            "obs_flip_rate_given_zero_fired": kz1 / k if k else None,
            "p_no_dem_error_at_all": float(np.prod(1 - p_err)),
            "mean_fired_per_shot": float(dets.sum(1).mean())}


res = {}
for d in (3, 5):
    G = Graph(d)
    diam, D = diameter(G.A)
    keep = np.arange(G.N) != G.b
    Ab = G.A[keep][:, keep]
    ncomp, labels = connected_components(Ab, directed=False)
    comp_sizes = np.bincount(labels).tolist()
    Db = shortest_path(Ab, unweighted=True, directed=False)
    diam_nb = int(Db[np.isfinite(Db)].max())
    det_deg = G.deg[keep]
    ecc = D[G.b][keep]
    walks, n_adj = walk_share(G)
    res[f"d{d}"] = {
        "d": d, "p": G.p, "rounds": d, "nodes": G.N, "detectors": G.N - 1,
        "undirected_edges": int(len(G.und)), "boundary_edges": int(G.is_bnd_edge.sum()),
        "boundary_node_degree": int(G.deg[G.b]),
        "boundary_adjacent_detectors": n_adj,
        "detector_degree_hist": {int(k): int(v) for k, v in
                                 zip(*np.unique(det_deg, return_counts=True))},
        "detector_degree_mean": float(det_deg.mean()),
        "diameter_with_boundary": diam,
        "max_hops_detector_to_boundary_node": int(ecc.max()),
        "connected_without_boundary": ncomp == 1, "components_without_boundary": comp_sizes,
        "diameter_without_boundary": diam_nb,
        "mean_hops_between_detectors_with_boundary":
            float(D[np.ix_(keep, keep)][np.triu_indices(G.N - 1, 1)].mean()),
        "mean_hops_between_detectors_without_boundary":
            float(Db[np.triu_indices(G.N - 1, 1)].mean()),
        "logical": logical(G),
        "walk_share_through_boundary": walks,
        "zero_fired": zero_fired(G),
    }
    r = res[f"d{d}"]
    print(f"d={d}: N={r['nodes']} E={r['undirected_edges']} bnd_deg={r['boundary_node_degree']} "
          f"diam={r['diameter_with_boundary']} / no-bnd {r['diameter_without_boundary']} "
          f"(connected {r['connected_without_boundary']}) logical={r['logical']} "
          f"zero-fired={r['zero_fired']['fraction']:.4f}")
    for w in walks[:4]:
        print("   ", w)
save("p1_graph.json", res)
