"""Variants of the DEM decoding graph for the architecture ablation (qec_run.py).

The default graph (qec_zx_dataset.build_dem_graph) is untouched; these options
rewrite a list of PyG Data objects into a variant graph. Every variant keeps the
fired bit in x column 0 and is_boundary in x column 4, keeps the labels and aux
targets as they are, and keeps every DEM edge with its original features.

boundary="global"     the original graph: one boundary node, the last node.
boundary="per-round"  one boundary node per detector round. Each boundary
                      half-edge (u, B) becomes (u, B_r) with r = the round (raw
                      detector t coordinate) of detector u. The new boundary
                      nodes are the last nodes, in increasing r, with static
                      features [0, 0, t_norm(r), 1]. Edge features are copied.
sublattice_types      appends one node feature and one edge feature:
                        x[:, 5]          1 for detectors of the label-carrying
                                         sublattice, 0 otherwise (other
                                         sublattice and boundary nodes)
                        edge_attr[:, 6]  1 for boundary edges whose detector is in
                                         the label-carrying sublattice
                      Without the boundary node the decoding graph splits into
                      two components (the two stabilizer types); the
                      label-carrying one is the one containing the observable-
                      flipping edges. Both sublattices are kept.
"""
from dataclasses import dataclass

import numpy as np
import torch

BOUNDARY_CHOICES = ("global", "per-round")


def feature_dims(sublattice_types: bool):
    """(node_in_dim, edge_in_dim) of graphs built with these options."""
    from gnn_models import EDGE_IN_DIM, NODE_IN_DIM
    return (NODE_IN_DIM + 1, EDGE_IN_DIM + 1) if sublattice_types else (NODE_IN_DIM, EDGE_IN_DIM)


def _components(n, edges):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    roots = [find(a) for a in range(n)]
    uniq = {r: i for i, r in enumerate(dict.fromkeys(roots))}
    return np.array([uniq[r] for r in roots])


def sublattice_labels(edge_index, edge_attr, num_detectors):
    """1 for detectors of the label-carrying sublattice, 0 for the other one.

    edge_index / edge_attr: the default graph (boundary node = num_detectors).
    Raises unless the graph without boundary has exactly two components and the
    observable-flipping edges all touch exactly one of them."""
    ei = np.asarray(edge_index)
    ea = np.asarray(edge_attr)
    nd = num_detectors
    inner = (ei[0] < nd) & (ei[1] < nd)
    comp = _components(nd, ei[:, inner].T)
    if comp.max() != 1:
        raise ValueError(f"expected 2 components without the boundary, got {comp.max() + 1}")
    flips = ea[:, 5] > 0.5
    det_end = np.where(ei[0] < nd, ei[0], ei[1])
    comps_with_obs = set(comp[det_end[flips]].tolist())
    if len(comps_with_obs) != 1:
        raise ValueError(f"observable-flipping edges touch components {comps_with_obs}")
    label = comps_with_obs.pop()
    return (comp == label).astype(np.int64)


@dataclass
class GraphSpec:
    boundary: str
    sublattice_types: bool
    num_detectors: int
    num_nodes: int
    boundary_nodes: list          # node indices of the boundary node(s)
    boundary_rounds: list         # raw detector round of each boundary node (per-round)
    static_x: torch.Tensor        # (num_nodes, node_dim) with x[:, 0] = 0
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    base_edge_index: np.ndarray   # the default graph's, to check inputs against
    sublattice: np.ndarray        # (num_detectors,) labels, see sublattice_labels


_SPEC_CACHE = {}


def build_spec(d, p, boundary="global", sublattice_types=False):
    if boundary not in BOUNDARY_CHOICES:
        raise ValueError(f"boundary must be one of {BOUNDARY_CHOICES}, got {boundary!r}")
    key = (d, p, boundary, sublattice_types)
    if key in _SPEC_CACHE:
        return _SPEC_CACHE[key]
    from qec_zx_dataset import build_circuit, build_dem_graph
    circ = build_circuit(d, p)
    g = build_dem_graph(circ)
    nd = circ.num_detectors
    b = g.boundary_index
    ei = g.edge_index.copy()
    ea = g.edge_attr.copy()
    static = g.static_node_feat.copy()           # (nd + 1, 4): x, y, t, is_boundary
    sub = sublattice_labels(g.edge_index, g.edge_attr, nd)

    bnodes, brounds = [b], []
    if boundary == "per-round":
        coords = circ.get_detector_coordinates()
        t = np.array([coords[i][2] for i in range(nd)])
        det_end = np.where(ei[0] == b, ei[1], ei[0])
        touches_b = (ei[0] == b) | (ei[1] == b)
        rounds = sorted({float(x) for x in t[det_end[touches_b]]})
        rank = {r: i for i, r in enumerate(rounds)}
        new_b = np.array([nd + rank[float(t[u])] for u in det_end])
        ei[0] = np.where(ei[0] == b, new_b, ei[0])
        ei[1] = np.where(ei[1] == b, new_b, ei[1])
        rows = np.zeros((len(rounds), 4), np.float32)
        for r, i in rank.items():
            rows[i, 2] = static[:nd, 2][t == r][0]       # normalised t of that round
            rows[i, 3] = 1.0
        static = np.concatenate([static[:nd], rows], axis=0)
        bnodes = list(range(nd, nd + len(rounds)))
        brounds = rounds

    n = static.shape[0]
    x = np.concatenate([np.zeros((n, 1), np.float32), static], axis=1)
    if sublattice_types:
        col = np.zeros((n, 1), np.float32)
        col[:nd, 0] = sub
        x = np.concatenate([x, col], axis=1)
        det_end = np.where(ei[0] < nd, ei[0], ei[1])
        ecol = (ea[:, 4] * sub[det_end]).astype(np.float32)[:, None]
        ea = np.concatenate([ea, ecol], axis=1)
    spec = GraphSpec(boundary, sublattice_types, nd, n, bnodes, brounds,
                     torch.from_numpy(x), torch.from_numpy(ei), torch.from_numpy(ea),
                     g.edge_index, sub)
    _SPEC_CACHE[key] = spec
    return spec


_DS_CACHE = {}


def transform(ds, d, p, boundary="global", sublattice_types=False):
    """Rewrite a list of default-graph Data objects into the variant graph.
    Returns `ds` itself when no option is set."""
    if boundary == "global" and not sublattice_types:
        return ds
    key = (id(ds), d, p, boundary, sublattice_types)
    if key in _DS_CACHE and _DS_CACHE[key][0] is ds:
        return _DS_CACHE[key][1]
    from torch_geometric.data import Data
    spec = build_spec(d, p, boundary, sublattice_types)
    nd = spec.num_detectors
    out = []
    for i, g in enumerate(ds):
        if i == 0 and not np.array_equal(g.edge_index.numpy(), spec.base_edge_index):
            raise ValueError("input graphs are not the default d/p decoding graph")
        x = spec.static_x.clone()
        x[:nd, 0] = g.x[:nd, 0]
        out.append(Data(x=x, edge_index=spec.edge_index, edge_attr=spec.edge_attr,
                        y=g.y, raw_target=g.raw_target, zx_target=g.zx_target))
    _DS_CACHE[key] = (ds, out)         # keep ds alive so its id is not reused
    return out
