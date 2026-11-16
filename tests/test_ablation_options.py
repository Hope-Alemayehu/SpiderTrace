"""Architecture-ablation options: --readout fired (C2), --head mlp (C5),
--residual / --depth (C4), --sublattice-types (C6), --boundary per-round (C3).

  * Defaults build the original model and graph (same parameters, same objects).
  * Every option: forward and backward pass for A, Raw and ZX at d=5, finite
    gradients on every parameter, sane logits at init (printed with -s).
  * C2: zero-fired shots get the learned empty vector; batches of only zero-fired
    shots, and batches ending in one, work; the pool equals a manual mean/max.
  * C6: the sublattice labels are the two components of the graph without the
    boundary node, the label-carrying one holds every observable-flipping edge.
  * C3: every merged DEM edge maps one-to-one onto the per-round graph with the
    same raw probability and observable flag; MWPM is unchanged.
  * qec_run end to end with each option: flags in provenance, records present.

    pytest tests/test_ablation_options.py -s
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch.nn as nn  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

import gnn_models  # noqa: E402
import graph_variants  # noqa: E402
import qec_run  # noqa: E402
from qec_zx_dataset import build_circuit, build_dem_graph  # noqa: E402

D, P, SHOTS, SPLIT_SEED = 5, 0.003, 2000, 12345

# name -> (model kwargs, graph kwargs)
CONFIGS = {
    "C2 readout fired": (dict(readout="fired"), {}),
    "C5 head mlp": (dict(head="mlp"), {}),
    "C4 residual + layer norm": (dict(residual=True, norm="layer"), {}),
    "C4b depth 8 + residual + layer norm": (dict(residual=True, norm="layer", num_layers=8), {}),
    "C6 sublattice types": ({}, dict(sublattice_types=True)),
    "C3 per-round boundary": ({}, dict(boundary="per-round")),
    "all combined": (dict(readout="fired", head="mlp", residual=True, norm="layer"),
                     dict(sublattice_types=True, boundary="per-round")),
}


@pytest.fixture(scope="module")
def base():
    train, _, _ = qec_run.make_datasets(D, P, SHOTS, SPLIT_SEED)
    return train[:256]


@pytest.fixture(scope="module")
def aux_qubits():
    return qec_run._data_qubits_for(D)


def build(kind, aux_qubits, model_kw, graph_kw):
    ni, ne = graph_variants.feature_dims(graph_kw.get("sublattice_types", False))
    qec_run._seed_all(0)
    return qec_run.build_model(kind, d=D, aux_qubits=aux_qubits, node_in_dim=ni,
                               edge_in_dim=ne, **model_kw)


# ---------------------------------------------------------------- defaults
def test_defaults_are_the_original_model_and_graph(base, aux_qubits):
    explicit = dict(head="linear", residual=False, num_layers=None, readout="add-mean-max",
                    norm="none", node_in_dim=5, edge_in_dim=6)
    for kind in ("A", "Raw", "ZX"):
        qec_run._seed_all(0)
        m = qec_run.build_model(kind, d=D, aux_qubits=aux_qubits)
        qec_run._seed_all(0)
        e = qec_run.build_model(kind, d=D, aux_qubits=aux_qubits, **explicit)
        assert list(m.state_dict()) == list(e.state_dict())
        assert all(torch.equal(v, e.state_dict()[k]) for k, v in m.state_dict().items())
    assert graph_variants.transform(base, D, P) is base


# ---------------------------------------------------------------- every option
@pytest.mark.parametrize("name", list(CONFIGS))
@pytest.mark.parametrize("kind", ["A", "Raw", "ZX"])
def test_forward_backward_and_init(name, kind, base, aux_qubits):
    model_kw, graph_kw = CONFIGS[name]
    ds = graph_variants.transform(base, D, P, **graph_kw)
    batch = Batch.from_data_list(ds)
    m = build(kind, aux_qubits, model_kw, graph_kw)
    m.eval()
    with torch.no_grad():
        logit, aux = m(batch)
    assert logit.shape == (len(ds),) and torch.isfinite(logit).all()
    if kind != "A":
        assert aux.shape == (len(ds), len(aux_qubits), gnn_models.NUM_PAULI)
    bce = nn.functional.binary_cross_entropy_with_logits(logit, batch.y.float())
    print(f"\n[{name} | {kind}] params {sum(t.numel() for t in m.parameters()):,} "
          f"init logit {logit.mean():.4g} +- {logit.std():.3g} BCE {bce:.4g}")
    m.train()
    loss = gnn_models.compute_loss(m, batch, nn.BCEWithLogitsLoss(), lambda_aux=0.1)
    loss.backward()
    for pname, t in m.named_parameters():
        assert t.grad is not None, f"no gradient for {pname}"
        assert torch.isfinite(t.grad).all(), f"non-finite gradient for {pname}"


# ---------------------------------------------------------------- C2 zero-fired
def fired_counts(ds):
    return np.array([int(g.x[:, 0].sum()) for g in ds])


def test_c2_zero_fired_shots_use_the_empty_vector(base, aux_qubits):
    counts = fired_counts(base)
    zero = [g for g, c in zip(base, counts) if c == 0]
    some = [g for g, c in zip(base, counts) if c > 0]
    assert len(zero) >= 3 and len(some) >= 3
    m = build("A", aux_qubits, dict(readout="fired"), {})
    with torch.no_grad():
        m.backbone.empty.copy_(torch.linspace(-1, 1, m.backbone.out_dim))
    m.eval()
    bb = m.backbone
    for graphs in (zero, some[:3] + zero[:1], zero[:1] + some[:3]):
        b = Batch.from_data_list(graphs)
        with torch.no_grad():
            g = bb(b.x, b.edge_index, b.edge_attr, b.batch)
        assert g.shape == (len(graphs), bb.out_dim)
        for i, gr in enumerate(graphs):
            if int(gr.x[:, 0].sum()) == 0:
                assert torch.equal(g[i], bb.empty)
            else:
                assert not torch.equal(g[i], bb.empty)
    # all-zero batch: every logit is the same, and training moves the empty vector
    b = Batch.from_data_list(zero)
    with torch.no_grad():
        logit, _ = m(b)
    assert torch.allclose(logit, logit[0].expand_as(logit))
    m.train()
    nn.functional.binary_cross_entropy_with_logits(m(b)[0], b.y.float()).backward()
    assert bb.empty.grad is not None and bb.empty.grad.abs().sum() > 0


def test_c2_pool_matches_manual_mean_max(base, aux_qubits):
    g = next(gr for gr in base if int(gr.x[:, 0].sum()) >= 2)
    m = build("A", aux_qubits, dict(readout="fired"), {})
    m.eval()
    b = Batch.from_data_list([g])
    bb = m.backbone
    with torch.no_grad():
        x = bb.node_encoder(b.x)
        for conv in bb.convs:
            x = torch.relu(conv(x, b.edge_index, b.edge_attr))
        f = b.x[:, 0] > 0.5
        manual = torch.cat([x[f].mean(0), x[f].max(0).values])
        got = bb(b.x, b.edge_index, b.edge_attr, b.batch)[0]
    assert torch.allclose(manual, got, atol=1e-5)


# ---------------------------------------------------------------- C6 sublattices
@pytest.mark.parametrize("d,sizes", [(3, (16, 8)), (5, (72, 48))])
def test_c6_labels_are_the_two_components(d, sizes):
    circ = build_circuit(d, P)
    g = build_dem_graph(circ)
    nd = circ.num_detectors
    lab = graph_variants.sublattice_labels(g.edge_index, g.edge_attr, nd)
    ei = g.edge_index
    inner = (ei[0] < nd) & (ei[1] < nd)
    A = coo_matrix((np.ones(inner.sum()), (ei[0][inner], ei[1][inner])), shape=(nd, nd))
    n, comp = connected_components(A, directed=False)
    assert n == 2
    # the labels are exactly one of the two components
    assert {tuple(sorted(set(comp[lab == 1]))), tuple(sorted(set(comp[lab == 0])))} == {(0,), (1,)}
    assert (int(lab.sum()), int((lab == 0).sum())) == sizes
    # every observable-flipping edge touches the label-carrying sublattice
    flips = g.edge_attr[:, 5] > 0.5
    det_end = np.where(ei[0] < nd, ei[0], ei[1])
    assert flips.any() and (lab[det_end[flips]] == 1).all()
    # memory-Z: the first and last detector rounds hold only Z-type detectors
    coords = circ.get_detector_coordinates()
    t = np.array([coords[i][2] for i in range(nd)])
    assert (lab[(t == t.min()) | (t == t.max())] == 1).all()


def test_c6_features(base):
    spec = graph_variants.build_spec(D, P, sublattice_types=True)
    ds = graph_variants.transform(base, D, P, sublattice_types=True)
    nd = spec.num_detectors
    for g, g0 in zip(ds[:20], base[:20]):
        assert g.x.shape == (nd + 1, 6) and g.edge_attr.shape[1] == 7
        assert torch.equal(g.x[:, :5], g0.x) and torch.equal(g.edge_attr[:, :6], g0.edge_attr)
        assert torch.equal(g.x[:nd, 5], torch.from_numpy(spec.sublattice).float())
        assert g.x[nd, 5] == 0
        assert torch.equal(g.y, g0.y) and torch.equal(g.zx_target, g0.zx_target)
    ea, ei = ds[0].edge_attr.numpy(), ds[0].edge_index.numpy()
    det_end = np.where(ei[0] < nd, ei[0], ei[1])
    expect = ea[:, 4] * spec.sublattice[det_end]
    assert np.array_equal(ea[:, 6], expect) and ea[:, 6].sum() > 0
    assert ((ea[:, 6] == 1) <= (ea[:, 4] == 1)).all()


# ---------------------------------------------------------------- C3 per-round boundary
def merged_dem(circ):
    """Independent copy of build_dem_graph's merge rule, from the decomposed DEM."""
    merged = {}
    for inst in circ.detector_error_model(decompose_errors=True).flattened():
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
            if not dets:
                continue
            obs = sum(1 for t in cc if t.is_logical_observable_id()) % 2
            k = frozenset(dets)
            pc = merged.get(k, [0.0, obs])[0]
            merged[k] = [pc + p - 2 * pc * p, obs]
    return merged


@pytest.mark.parametrize("d", [3, 5])
def test_c3_every_dem_edge_maps_one_to_one(d):
    circ = build_circuit(d, P)
    nd = circ.num_detectors
    spec = graph_variants.build_spec(d, P, boundary="per-round")
    coords = circ.get_detector_coordinates()
    t = np.array([coords[i][2] for i in range(nd)])
    merged = merged_dem(circ)
    pc = np.clip(np.array([v[0] for v in merged.values()]), 1e-12, 1 - 1e-12)
    w = -np.log(pc / (1 - pc))
    wn = ((w - w.mean()) / w.std()).astype(np.float32)

    ei = spec.edge_index.numpy()
    ea = spec.edge_attr.numpy()
    und = ei[:, 0::2]
    assert np.array_equal(ei[:, 1::2], und[::-1]), "edges not stored as u->v, v->u pairs"
    assert np.array_equal(ea[0::2], ea[1::2])
    assert und.shape[1] == len(merged), "edge count differs from the DEM"
    bnode = {r: spec.boundary_nodes[i] for i, r in enumerate(spec.boundary_rounds)}
    seen = set()
    for j, (key, (p_e, obs)) in enumerate(merged.items()):
        ds = sorted(key)
        expect = (ds[0], ds[1]) if len(ds) == 2 else (ds[0], bnode[float(t[ds[0]])])
        got = tuple(int(x) for x in und[:, j])
        assert got == expect, (key, got)
        assert ea[2 * j, 0] == wn[j], "raw probability (w_norm) differs"
        assert ea[2 * j, 5] == obs and ea[2 * j, 4] == (len(ds) == 1)
        seen.add(got)
    assert len(seen) == len(merged)                         # one-to-one
    # one boundary node per detector round that has boundary edges, only to that round
    assert spec.num_nodes == nd + len(spec.boundary_rounds)
    for r, b in bnode.items():
        nb = und[0][und[1] == b]
        assert len(nb) > 0 and (t[nb] == r).all()
    old_hub_degree = sum(1 for k in merged if len(k) == 1)
    assert sum(int((und[1] == b).sum()) for b in spec.boundary_nodes) == old_hub_degree
    if d == 5:
        assert spec.boundary_rounds == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_c3_dataset_and_mwpm_unchanged(base):
    ds = graph_variants.transform(base, D, P, boundary="per-round")
    spec = graph_variants.build_spec(D, P, boundary="per-round")
    nd = spec.num_detectors
    for g, g0 in zip(ds[:20], base[:20]):
        assert g.x.shape == (spec.num_nodes, 5)
        assert torch.equal(g.x[:nd], g0.x[:nd]) and (g.x[nd:, 0] == 0).all()
        assert (g.x[nd:, 4] == 1).all() and torch.equal(g.y, g0.y)
    a = qec_run.mwpm_on_test(D, P, base)
    b = qec_run.mwpm_on_test(D, P, ds)
    assert a[0] == b[0] and np.array_equal(a[1], b[1])


# ---------------------------------------------------------------- end to end
CLI = ["qec_run.py", "--d", "3", "--p", "0.01", "--shots", "1000", "--seeds", "0",
       "--epochs", "2", "--lambdas", "0.1", "--device", "cpu"]


@pytest.mark.parametrize("flags,arms", [
    (["--readout", "fired"], ["A"]),
    (["--head", "mlp"], ["A"]),
    (["--residual", "--norm", "layer"], ["A"]),
    (["--residual", "--norm", "layer", "--depth", "8"], ["A"]),
    (["--sublattice-types"], ["A"]),
    (["--boundary", "per-round"], ["A"]),
    (["--readout", "fired", "--head", "mlp", "--residual", "--norm", "layer",
      "--sublattice-types", "--boundary", "per-round"], ["A", "Raw", "ZX"]),
])
def test_cli(flags, arms, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", CLI + ["--arms", *arms, "--outdir", str(tmp_path), *flags,
                                            "--big-test-shots", "500"])
    qec_run.main()
    (f,) = Path(tmp_path).glob("run_*.json")
    r = json.loads(f.read_text())
    cfg = r["provenance"]["config"]
    for k in ("readout", "head", "residual", "norm", "depth", "sublattice_types", "boundary"):
        assert k in cfg
    for arm in arms:
        a = r["arms"][arm]
        assert a["model"]["readout"] == cfg["readout"] and a["graph"]["boundary"] == cfg["boundary"]
        zf = a["selected_per_seed"][0]["zero_fired_test"]
        assert zf["n"] > 0 and 0 <= zf["pred_flip"] <= zf["n"]
        assert "zero_fired" in r["big_test"]["arms"][arm]["0"]
