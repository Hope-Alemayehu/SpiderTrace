"""Shared helpers for the Phase 1 architecture diagnosis. Read-only use of the
repo: imports qec_zx_dataset / qec_run / gnn_models, never modifies them.

graph(d, p)   the GNN's own input graph, qec_zx_dataset.build_dem_graph(
              build_circuit(d, p)) with rounds = d, plus an undirected adjacency
              matrix (boundary node included) and {frozenset(u, v): flips_obs}.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "phase1_results"
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import qec_zx_dataset  # noqa: E402

P = 0.003


class Graph:
    def __init__(self, d, p=P):
        self.d, self.p = d, p
        self.circ = qec_zx_dataset.build_circuit(d, p)          # rounds = d
        g = qec_zx_dataset.build_dem_graph(self.circ)
        self.g = g
        self.N = g.num_nodes
        self.b = g.boundary_index
        ei = g.edge_index
        # build_dem_graph stores each undirected edge twice (u->v then v->u)
        self.und = ei[:, 0::2].T                                  # (E, 2)
        self.flips = g.edge_attr[0::2, 5].astype(int)             # observable parity per edge
        self.is_bnd_edge = g.edge_attr[0::2, 4].astype(int)
        self.A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])),
                               shape=(self.N, self.N)).tocsr()
        self.A.data[:] = 1.0
        self.obs_of = {frozenset((int(u), int(v))): int(f)
                       for (u, v), f in zip(self.und, self.flips)}
        self.deg = np.asarray(self.A.sum(1)).ravel().astype(int)
        # Raw edge weights -log(p/(1-p)): build_dem_graph only keeps them z-scored.
        merged = merged_raw(self.circ)
        pc = np.clip(np.array([v[0] for v in merged.values()]), 1e-12, 1 - 1e-12)
        self.w_raw = -np.log(pc / (1 - pc))
        ends = np.array([sorted(k) if len(k) == 2 else [min(k), self.b] for k in merged]).T
        wn = ((self.w_raw - self.w_raw.mean()) / (self.w_raw.std() or 1.0)).astype(np.float32)
        assert np.array_equal(ends, self.und.T) and np.array_equal(wn, g.edge_attr[0::2, 0]), \
            "merge replica does not reproduce build_dem_graph"


def merged_raw(circ):
    """Verbatim copy of build_dem_graph's merge step (qec_zx_dataset.py:391-419);
    Graph() checks it reproduces build_dem_graph's edges and w_norm exactly."""
    dem = circ.detector_error_model(decompose_errors=True)
    merged = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        comp, comps = [], []
        for t in inst.targets_copy():
            if t.is_separator():
                comps.append(comp); comp = []
            else:
                comp.append(t)
        comps.append(comp)
        for cc in comps:
            dets = tuple(sorted(t.val for t in cc if t.is_relative_detector_id()))
            obs_parity = sum(1 for t in cc if t.is_logical_observable_id()) % 2
            if not dets:
                continue
            key = frozenset(dets)
            if key not in merged:
                merged[key] = [0.0, obs_parity]
            pc = merged[key][0]
            merged[key][0] = pc + p - 2 * pc * p
            merged[key][1] = obs_parity
    return merged


def save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float))
    print(f"wrote {OUT / name}")
