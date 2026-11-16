"""Phase 1, task 2 (empirical): hop span of error chains, at d=3 and d=5, p=0.003.

    python analysis/p2_chains.py [--shots 200000]  ->  phase1_results/p2_chains.json

Shots come from the DECOMPOSED DEM sampler with return_errors=True, so each fired
error instruction is a known set of graphlike edges (1 or 2 detectors each, plus
an observable flag). Per shot, the fired edges (mod 2) are grouped into chains:
connected components over detectors, where boundary edges hang off their
detector and do NOT connect chains through the boundary node.

For each chain: number of edges, number of detectors, boundary edges, observable
parity (odd = would cause a logical error if left uncorrected), and its hop span
= max pairwise shortest-path distance between its detectors on the decoding graph
WITHOUT the boundary node (spatial extent). With the boundary node the span is
also reported (the hub shortens it).

MWPM failures: pymatching on the same decomposed DEM; for each shot where its
predicted observable differs from the truth, the edge set (true fired edges XOR
matched edges) forms closed loops; the odd-parity loops are the minimal logical
operators the decoder got wrong. Their edge counts and hop spans are reported.

"Compared with 4 layers": a node's k-layer receptive field is its k-hop
neighbourhood, so a chain spanning s hops is fully visible from its central
detector after ceil(s/2) layers and from an end detector after s layers.
"""
import argparse
from collections import Counter, defaultdict

import numpy as np
import pymatching
from scipy.sparse.csgraph import shortest_path

from common import Graph, save


def components(edges, G):
    """edges: iterable of (u, v) with v = -1 for boundary. Union-find over detectors."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        find(u)
        if v >= 0:
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
    comps = defaultdict(lambda: {"edges": 0, "bnd": 0, "dets": set(), "parity": 0})
    for u, v in edges:
        c = comps[find(u)]
        c["edges"] += 1
        c["dets"].add(u)
        if v < 0:
            c["bnd"] += 1
            c["parity"] ^= G.obs_of[frozenset((u, G.b))]
        else:
            c["dets"].add(v)
            c["parity"] ^= G.obs_of[frozenset((u, v))]
    return list(comps.values())


INF_SPAN = 99   # detectors in different components of the no-boundary graph


def span(dets, D):
    idx = sorted(dets)
    if len(idx) < 2:
        return 0
    m = D[np.ix_(idx, idx)].max()
    return INF_SPAN if not np.isfinite(m) else int(m)


def hist(values, cap=12):
    c = Counter(min(int(v), cap) for v in values)
    return {str(k) if k < cap else f">={cap}": c[k] for k in sorted(c)}


def analyse(d, shots, seed):
    G = Graph(d)
    dem = G.circ.detector_error_model(decompose_errors=True)
    # error instruction -> list of (u, v) edges (v = -1 boundary); 0-detector components counted
    instr_edges, n_undetectable = [], 0
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        comp, comps = [], []
        for t in inst.targets_copy():
            if t.is_separator():
                comps.append(comp); comp = []
            else:
                comp.append(t)
        comps.append(comp)
        edges = []
        for cc in comps:
            dets = sorted(t.val for t in cc if t.is_relative_detector_id())
            if not dets:
                n_undetectable += 1
                continue
            edges.append((dets[0], dets[1]) if len(dets) == 2 else (dets[0], -1))
        instr_edges.append(edges)
    assert len(instr_edges) == dem.num_errors

    keep = np.arange(G.N) != G.b
    D_nb = shortest_path(G.A[keep][:, keep], unweighted=True, directed=False)
    D_wb = shortest_path(G.A, unweighted=True, directed=False)[np.ix_(keep, keep)]
    # detector indices are 0..N-2, boundary is N-1, so D_nb indices == detector ids

    sampler = dem.compile_sampler(seed=seed)
    dets, obs, errs = sampler.sample(shots=shots, return_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)

    chain_stats = {"all": [], "odd": [], "even": []}
    shot_span, shot_fired = [], []
    fail_loops, n_fail, n_fail_multi_odd, n_fail_zero_odd = [], 0, 0, 0
    agree_check = 0
    for s in range(shots):
        cnt = Counter()
        for i in np.flatnonzero(errs[s]):
            for e in instr_edges[i]:
                cnt[e] += 1
        fired_edges = [e for e, c in cnt.items() if c % 2 == 1]
        fired_dets = np.flatnonzero(dets[s])
        shot_fired.append(len(fired_dets))
        shot_span.append(span(fired_dets, D_nb))
        for c in components(fired_edges, G):
            rec = (c["edges"], len(c["dets"]), c["bnd"], span(c["dets"], D_nb),
                   span(c["dets"], D_wb))
            chain_stats["all"].append(rec)
            chain_stats["odd" if c["parity"] else "even"].append(rec)

        matched = matching.decode_to_edges_array(dets[s])
        matched_edges = [(int(u), int(v)) if v < 0 or u < v else (int(v), int(u))
                         for u, v in matched]
        pred = 0
        for u, v in matched_edges:
            pred ^= G.obs_of[frozenset((u, G.b if v < 0 else v))]
        agree_check += int(pred == int(matching.decode(dets[s])[0]))
        if pred != int(obs[s, 0]):
            n_fail += 1
            loop_edges = list(set(fired_edges) ^ set(matched_edges))
            odd = [c for c in components(loop_edges, G) if c["parity"]]
            if len(odd) != 1:
                n_fail_multi_odd += len(odd) > 1
                n_fail_zero_odd += len(odd) == 0
            for c in odd:
                fail_loops.append((c["edges"], len(c["dets"]), c["bnd"],
                                   span(c["dets"], D_nb), span(c["dets"], D_wb)))

    def summarise(recs):
        if not recs:
            return {"count": 0}
        a = np.array(recs)
        return {"count": len(recs),
                "edges_hist": hist(a[:, 0]), "edges_mean": float(a[:, 0].mean()),
                "detectors_hist": hist(a[:, 1]), "detectors_mean": float(a[:, 1].mean()),
                "boundary_edges_hist": hist(a[:, 2]),
                "span_no_boundary_hist": hist(a[:, 3]),
                "frac_disconnected_without_boundary": float((a[:, 3] == INF_SPAN).mean()),
                "span_no_boundary_mean_connected": float(a[a[:, 3] < INF_SPAN, 3].mean()),
                "span_with_boundary_hist": hist(a[:, 4]),
                "frac_span_le_4": float((a[:, 3] <= 4).mean()),
                "frac_span_le_2": float((a[:, 3] <= 2).mean()),
                "frac_half_span_le_4": float((np.ceil(a[:, 3] / 2) <= 4).mean())}

    out = {"d": d, "p": G.p, "shots": shots, "seed": seed,
           "dem_error_instructions": dem.num_errors,
           "undetectable_components": n_undetectable,
           "mwpm_pred_matches_decode": agree_check == shots,
           "mwpm_failures": n_fail, "mwpm_ler": n_fail / shots,
           "failures_with_multiple_odd_loops": n_fail_multi_odd,
           "failures_with_zero_odd_loops": n_fail_zero_odd,
           "obs_flip_rate": float(obs[:, 0].mean()),
           "fired_detectors_per_shot": {"mean": float(np.mean(shot_fired)),
                                        "hist": hist(shot_fired, 16)},
           "shot_span_no_boundary": {"mean": float(np.mean(shot_span)),
                                     "hist": hist(shot_span)},
           "chains": {k: summarise(v) for k, v in chain_stats.items()},
           "mwpm_failure_loops": summarise(fail_loops),
           "diameter_no_boundary_finite": int(D_nb[np.isfinite(D_nb)].max()),
           "diameter_with_boundary": int(D_wb.max())}
    print(f"d={d}: chains {len(chain_stats['all'])} (odd {len(chain_stats['odd'])}), "
          f"MWPM failures {n_fail}/{shots}; odd-chain span hist {out['chains']['odd'].get('span_no_boundary_hist')}; "
          f"failure-loop span hist {out['mwpm_failure_loops'].get('span_no_boundary_hist')}")
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--shots", type=int, default=200000)
ap.add_argument("--seed", type=int, default=2026)
a = ap.parse_args()
save("p2_chains.json", {f"d{d}": analyse(d, a.shots, a.seed) for d in (3, 5)})
