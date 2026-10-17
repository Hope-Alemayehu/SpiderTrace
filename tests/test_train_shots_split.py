"""--train-shots resizes ONLY the train split (qec_run.make_datasets).

Val and test must be identical with and without it, so every train size is
scored on the same test shots as a default run. Train sets must be nested
across sizes, so a larger train set only ever adds shots.

Every dataset is rebuilt from scratch (the dataset cache is emptied before each
call), so "identical" means equal values from independent builds, not the same
cached objects. Small config (d=3, 2,000 shots) so it runs in seconds on CPU:
the base split is 1,400 train / 300 val / 300 test.

    pytest tests/test_train_shots_split.py
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import qec_run  # noqa: E402

D, P, SHOTS, SPLIT_SEED = 3, 0.01, 2000, 12345
N_BASE_TRAIN = 1400
FIELDS = ("x", "edge_index", "edge_attr", "y", "raw_target", "zx_target")


def build(**kw):
    """make_datasets on an empty cache: an independent build every call."""
    qec_run._DATASET_CACHE.clear()
    return qec_run.make_datasets(D, P, SHOTS, SPLIT_SEED, **kw)


def assert_same(a, b):
    assert len(a) == len(b)
    for i, (g, h) in enumerate(zip(a, b)):
        for k in FIELDS:
            assert torch.equal(getattr(g, k), getattr(h, k)), f"shot {i} field {k}"


@pytest.fixture(scope="module")
def base():
    train, val, test = build()
    assert (len(train), len(val), len(test)) == (N_BASE_TRAIN, 300, 300)
    return train, val, test


@pytest.mark.parametrize("n", [500, N_BASE_TRAIN, 3000])
def test_val_test_identical_with_and_without_train_shots(base, n):
    _, val0, test0 = base
    train, val, test = build(train_shots=n)
    assert len(train) == n
    assert_same(val, val0)
    assert_same(test, test0)


def test_train_sets_nested(base, monkeypatch):
    train0 = base[0]
    # Small chunks so the extras span several sample() calls and a partial one.
    monkeypatch.setattr(qec_run, "_EXTRA_TRAIN_CHUNK", 256)
    small = build(train_shots=500)[0]
    mid = build(train_shots=2600)[0]  # 1,200 extra
    big = build(train_shots=3500)[0]  # 2,100 extra
    assert_same(small, train0[:500])
    assert_same(build(train_shots=N_BASE_TRAIN)[0], train0)
    assert_same(mid[:N_BASE_TRAIN], train0)
    assert_same(big[:len(mid)], mid)


def test_extra_seed_changes_only_the_extra_shots(base):
    train0 = base[0]
    a = build(train_shots=2000, extra_train_seed=1)[0]
    b = build(train_shots=2000, extra_train_seed=2)[0]
    assert_same(a[:N_BASE_TRAIN], train0)
    assert_same(b[:N_BASE_TRAIN], train0)
    assert any(not torch.equal(g.x, h.x)
               for g, h in zip(a[N_BASE_TRAIN:], b[N_BASE_TRAIN:]))


def test_extra_seed_equal_to_split_seed_rejected():
    with pytest.raises(ValueError, match="equals split_seed"):
        build(train_shots=3000, extra_train_seed=SPLIT_SEED)
