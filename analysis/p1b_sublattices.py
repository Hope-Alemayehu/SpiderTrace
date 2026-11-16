"""Phase 1 follow-up: the two components of the decoding graph without the boundary
node. Which one carries the observable-flipping edges, and how many fired
detectors per shot fall in each (d=3 and d=5, p=0.003, 1M seeded shots).

    python analysis/p1b_sublattices.py  ->  phase1_results/p1b_sublattices.json
"""
import numpy as np
from scipy.sparse.csgraph import connected_components

from common import Graph, save

res = {}
for d in (3, 5):
    G = Graph(d)
    keep = np.arange(G.N) != G.b
    ncomp, lab = connected_components(G.A[keep][:, keep], directed=False)
    assert ncomp == 2
    # final-round detectors (largest t) sit in the component the observable lives in
    coords = G.circ.get_detector_coordinates()
    t = np.array([coords[i][2] for i in range(G.N - 1)])
    big = int(np.bincount(lab).argmax())
    comp_of_edge = []
    for (u, v), f in zip(G.und, G.flips):
        comp_of_edge.append(lab[u])          # u is always a detector
    comp_of_edge = np.array(comp_of_edge)
    dem = G.circ.detector_error_model(decompose_errors=False, flatten_loops=True)
    dets, obs, _ = dem.compile_sampler(seed=2026).sample(shots=1_000_000)
    fired_by_comp = [dets[:, lab == c].sum(1) for c in range(2)]
    res[f"d{d}"] = {
        "components": {str(c): {
            "detectors": int((lab == c).sum()),
            "detector_rounds_t": sorted({float(x) for x in t[lab == c]}),
            "edges_incl_boundary": int((comp_of_edge == c).sum()),
            "obs_flipping_edges": int((G.flips[comp_of_edge == c]).sum()),
            "mean_fired_per_shot": float(fired_by_comp[c].mean()),
            "frac_shots_with_a_fired_detector": float((fired_by_comp[c] > 0).mean()),
        } for c in range(2)},
        "frac_shots_fired_in_both_components": float(((fired_by_comp[0] > 0) & (fired_by_comp[1] > 0)).mean()),
        "obs_flip_rate": float(obs[:, 0].mean()),
    }
    print(d, res[f"d{d}"])
save("p1b_sublattices.json", res)
