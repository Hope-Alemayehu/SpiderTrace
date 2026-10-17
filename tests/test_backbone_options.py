"""Backbone options --norm / --readout (gnn_models.GNNBackbone).

  * The defaults build the original backbone: same parameter count and the same
    state_dict keys, so seeded runs start from the same weights as before.
  * --norm layer --readout mean-max gives a working forward and backward pass
    for all three arms (A, Raw, ZX), with sane logits at initialisation.

Models are built the way run_arm builds them (qec_run._seed_all, then
qec_run.build_model), at d=5, p=0.003, on a small seeded batch.

    pytest tests/test_backbone_options.py -s     # -s prints the init statistics
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch.nn as nn  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

import gnn_models  # noqa: E402
import qec_run  # noqa: E402

D, P, SHOTS, SPLIT_SEED, MODEL_SEED = 5, 0.003, 2000, 12345, 0
NEW = dict(norm="layer", readout="mean-max")
N_PARAMS_A_DEFAULT = 78053      # 4 x GINEConv(96) + encoder + Linear(288, 1)


@pytest.fixture(scope="module")
def batch():
    train, _, _ = qec_run.make_datasets(D, P, SHOTS, SPLIT_SEED)
    return next(iter(DataLoader(train[:256], batch_size=256, shuffle=False)))


@pytest.fixture(scope="module")
def aux_qubits():
    return qec_run._data_qubits_for(D)


def build(kind, aux_qubits, **kw):
    qec_run._seed_all(MODEL_SEED)
    return qec_run.build_model(kind, d=D, aux_qubits=aux_qubits, **kw)


def n_params(m):
    return sum(t.numel() for t in m.parameters())


def test_defaults_build_the_original_backbone(aux_qubits):
    explicit = dict(norm="none", readout="add-mean-max")
    for kind in ("A", "Raw", "ZX"):
        m, e = build(kind, aux_qubits), build(kind, aux_qubits, **explicit)
        assert m.backbone.out_dim == gnn_models.READOUT_DIM == 288
        assert len(m.backbone.norms) == 0
        assert list(m.state_dict()) == list(e.state_dict())
        for k, v in m.state_dict().items():
            assert torch.equal(v, e.state_dict()[k]), k
        assert not any(isinstance(mod, nn.LayerNorm) for mod in m.modules())
    assert n_params(build("A", aux_qubits)) == N_PARAMS_A_DEFAULT


@pytest.mark.parametrize("kind", ["A", "Raw", "ZX"])
def test_layer_norm_mean_max_forward_backward(kind, batch, aux_qubits):
    m = build(kind, aux_qubits, **NEW)
    bb = m.backbone
    assert bb.out_dim == 2 * gnn_models.HIDDEN
    assert len(bb.norms) == len(bb.convs)
    assert m.flip_head.in_features == bb.out_dim
    if kind != "A":
        assert m.aux_head.in_features == bb.out_dim
    # LayerNorm adds 2 * 96 per layer; the flip head loses 96 inputs.
    if kind == "A":
        assert n_params(m) == N_PARAMS_A_DEFAULT + 4 * 2 * 96 - 96

    m.eval()
    with torch.no_grad():
        logit, aux = m(batch)
    B = batch.y.numel()
    assert logit.shape == (B,)
    if kind == "A":
        assert aux is None
    else:
        assert aux.shape == (B, len(aux_qubits), gnn_models.NUM_PAULI)
    bce = nn.functional.binary_cross_entropy_with_logits(logit, batch.y.float())
    print(f"\n[{kind}] init (eval): logit mean {logit.mean():.4f} std {logit.std():.4f}"
          f" | BCE {bce:.4f}")
    # The default backbone starts at logit ~ +147, BCE ~ 120 on this setting.
    assert torch.isfinite(logit).all()
    assert logit.abs().max() < 5 and bce < 2

    m.train()
    loss = gnn_models.compute_loss(m, batch, nn.BCEWithLogitsLoss(), lambda_aux=0.1)
    loss.backward()
    grads = [t.grad for t in m.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_bad_option_rejected(aux_qubits):
    with pytest.raises(ValueError, match="norm must be"):
        build("A", aux_qubits, norm="batch")
    with pytest.raises(ValueError, match="readout must be"):
        build("A", aux_qubits, readout="sum")
