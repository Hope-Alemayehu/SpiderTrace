"""--fresh-train: new training shots every epoch (fresh_data.FreshTrainSource).

  * The sampling is the draw qec_run makes for fresh shots (_fresh_samples).
  * Epoch k's batches equal what the current code builds from the same seed:
    Batch.from_data_list over qec_run._fresh_shots(d, p, n, base + k), in sample
    order, field by field, including the raw and ZX aux targets, for the default
    graph and both graph variants.
  * Aux class weights from fresh shots equal compute_aux_class_weights.
  * The seed schedule never overlaps the split, extra-train, big-test or eval-set
    seeds, and qec_run refuses a schedule that would.
  * A run records the schedule in provenance; with --fresh-train off nothing changes
    (the rest of the suite runs the default path, and tests/test_eval_outputs.py
    checks that saving models changes no result).

    pytest tests/test_fresh_train.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import Batch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import fresh_data  # noqa: E402
import graph_variants  # noqa: E402
import qec_run  # noqa: E402

D, P, SPLIT = 3, 0.01, 12345
N, BS = 1000, 256          # 3 full batches and a partial one
BASE = fresh_data.resolve_fresh_seed_base(SPLIT)
FIELDS = ("x", "edge_index", "edge_attr", "y", "batch", "ptr")


def template(variant=None):
    _, val, _ = qec_run.make_datasets(D, P, 1000, SPLIT)
    g = [val[0]]
    return (graph_variants.transform(g, D, P, **variant) if variant else g)[0]


def source(aux_key=None, variant=None, n=N):
    return fresh_data.FreshTrainSource(D, P, n, BASE, template(variant), BS,
                                       aux_key=aux_key, chunk=qec_run._FRESH_SHOT_CHUNK)


def reference_batches(epoch, variant=None, n=N):
    shots = qec_run._fresh_shots(D, P, n, BASE + epoch)
    if variant:
        shots = graph_variants.transform(shots, D, P, **variant)
    return [Batch.from_data_list(shots[a:a + BS]) for a in range(0, n, BS)]


def test_sampling_is_the_fresh_shots_draw():
    n = 23000                       # crosses chunk boundaries and truncates
    got = list(fresh_data.sample_chunks(D, P, n, 777, qec_run._FRESH_SHOT_CHUNK))
    ref = qec_run._fresh_samples(D, P, n, 777)
    for i in range(3):
        assert np.array_equal(np.concatenate([c[i] for c in got]), ref[i])


@pytest.mark.parametrize("aux_key", [None, "raw_target", "zx_target"])
def test_epoch_batches_equal_current_code(aux_key):
    src = source(aux_key)
    epoch = 3
    src.set_fresh_epoch(epoch)
    mine = list(src)
    ref = reference_batches(epoch)
    assert len(mine) == len(ref) == 4
    for b, r in zip(mine, ref):
        for k in FIELDS + ((aux_key,) if aux_key else ()):
            assert b[k].dtype == r[k].dtype, k
            assert torch.equal(b[k], r[k]), k


@pytest.mark.parametrize("variant", [dict(boundary="per-round"),
                                     dict(sublattice_types=True)])
def test_epoch_batches_equal_current_code_graph_variants(variant):
    src = source("zx_target", variant)
    src.set_fresh_epoch(2)
    for b, r in zip(src, reference_batches(2, variant)):
        for k in FIELDS + ("zx_target",):
            assert torch.equal(b[k], r[k]), k


def test_each_epoch_is_new_and_reproducible():
    src = source()
    fired = {e: src.epoch_arrays(e)[0].copy() for e in (1, 2)}
    assert not np.array_equal(fired[1], fired[2])
    assert np.array_equal(source().epoch_arrays(1)[0], fired[1])   # same seed, same shots


@pytest.mark.parametrize("aux_key", ["raw_target", "zx_target"])
def test_aux_class_weights_match_current_code(aux_key):
    q = qec_run._data_qubits_for(D)
    nq = qec_run._num_qubits_for(D)
    mine = qec_run.aux_class_weights_from_counts(source(aux_key).class_counts(1, q))
    ref = qec_run.compute_aux_class_weights(qec_run._fresh_shots(D, P, N, BASE + 1),
                                            aux_key, nq, q)
    assert mine == ref


def test_default_schedule_avoids_reserved_seeds():
    for split in (0, 1, SPLIT, 99999):
        sched = fresh_data.fresh_seed_schedule(
            fresh_data.resolve_fresh_seed_base(split), 10000)
        reserved = {split, split + 1, split + 2}       # split, extra-train, big test
        assert not set(sched) & reserved
        assert len(set(sched)) == len(sched)


ARGS = ["qec_run.py", "--d", str(D), "--p", str(P), "--shots", "1000", "--seeds", "0",
        "--arms", "A", "ZX", "--lambdas", "0.1", "--epochs", "2", "--device", "cpu",
        "--no-save-models"]


def run_main(mp, outdir, *extra):
    mp.setattr(sys, "argv", ARGS + ["--outdir", str(outdir), *extra])
    qec_run.main()
    (f,) = Path(outdir).glob("run_*.json")
    return json.loads(f.read_text())


@pytest.mark.parametrize("extra", [
    ["--fresh-train"],                                           # no --train-shots
    ["--train-shots", "600", "--fresh-train-seed", "5"],         # seed without flag
    ["--train-shots", "600", "--fresh-train", "--extra-train-seed", "7"],
    ["--train-shots", "600", "--fresh-train", "--fresh-train-seed", str(SPLIT - 1)],
    ["--train-shots", "600", "--fresh-train", "--fresh-train-seed", str(SPLIT),
     "--big-test-shots", "100"],                                 # epoch 1 = 12346
    ["--train-shots", "600", "--fresh-train", "--fresh-train-seed", "100",
     "--big-test-shots", "100", "--big-test-seed", "102"],       # epoch 2 = big test
])
def test_bad_fresh_options_rejected(extra, tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, *extra)


def test_schedule_overlapping_eval_set_rejected(tmp_path, monkeypatch):
    ev = tmp_path / "ev.npz"
    qec_run.write_eval_set(ev, D, P, 300, 99999)
    with pytest.raises(RuntimeError, match="fresh-train seed schedule"):
        run_main(monkeypatch, tmp_path / "out", "--train-shots", "600", "--fresh-train",
                 "--fresh-train-seed", "99997", "--eval-set", str(ev))


def test_run_records_schedule(tmp_path, monkeypatch):
    r = run_main(monkeypatch, tmp_path, "--train-shots", "600", "--fresh-train",
                 "--aux-data-qubits-only", "--aux-class-weight", "inv-freq")
    ft = r["provenance"]["fresh_train"]
    assert ft["seed_base"] == BASE and ft["seeds_per_epoch"] == [BASE + 1, BASE + 2]
    assert ft["shots_per_epoch"] == 600
    assert r["splits"]["fresh_train_per_epoch"] == 600
    for kind in ("A", "ZX"):
        for g in r["arms"][kind]["lambda_grid"]:
            assert g["train"]["n_train"] == 600 and g["train"]["epochs_run"] == 2
    q = qec_run._data_qubits_for(D)
    ref_w, _ = qec_run.compute_aux_class_weights(
        qec_run._fresh_shots(D, P, 600, BASE + 1), "zx_target",
        qec_run._num_qubits_for(D), q)
    assert r["arms"]["ZX"]["aux_class_weights"] == ref_w


def test_fresh_train_off_records_nothing(tmp_path, monkeypatch):
    r = run_main(monkeypatch, tmp_path, "--train-shots", "600")
    assert "fresh_train" not in r["provenance"]
    assert r["provenance"]["config"]["fresh_train"] is False
    assert "fresh_train_per_epoch" not in r["splits"]
