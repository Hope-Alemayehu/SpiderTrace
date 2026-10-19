"""--big-test-shots: a secondary test set of fresh shots (qec_run.main).

  * The primary results (7,500-shot split, here a small one) are identical with
    and without the big test set.
  * The big_test block is consistent: error counts match the wrong-shot lists,
    and the McNemar counts match the lists of both decoders.
  * Its shots come from their own seed, so they differ from the split pool and
    from the extra-train shots, and the seed checks reject collisions.

Small config (d=3, 1,000 shots, GNN-A, 3 epochs, 3,000 big-test shots) so it
runs in seconds on CPU.

    pytest tests/test_big_test.py
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

import qec_run  # noqa: E402

ARGS = ["qec_run.py", "--d", "3", "--p", "0.01", "--shots", "1000", "--seeds", "0",
        "--arms", "A", "--epochs", "3", "--device", "cpu"]
N_BIG = 3000
SPLIT_SEED = 12345


def run_main(mp, outdir, *extra):
    mp.setattr(sys, "argv", ARGS + ["--outdir", str(outdir), *extra])
    qec_run.main()
    (f,) = Path(outdir).glob("run_*.json")
    return json.loads(f.read_text())


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        plain = run_main(mp, tmp_path_factory.mktemp("plain"))
        big = run_main(mp, tmp_path_factory.mktemp("big"),
                       "--big-test-shots", str(N_BIG))
    return plain, big


def strip_volatile(obj):
    if isinstance(obj, dict):
        return {k: strip_volatile(v) for k, v in obj.items()
                if k not in ("provenance", "wall_time_s", "big_test")}
    if isinstance(obj, list):
        return [strip_volatile(v) for v in obj]
    return obj


def test_primary_results_unchanged(runs):
    plain, big = runs
    assert "big_test" not in plain
    assert strip_volatile(big) == strip_volatile(plain)


def test_big_test_block_is_consistent(runs):
    bt = runs[1]["big_test"]
    assert bt["n_test"] == N_BIG
    assert bt["seed"] == SPLIT_SEED + 2
    m, a = bt["mwpm"], bt["arms"]["A"]["0"]
    for rec in (m, a):
        assert rec["n_errors"] == len(rec["test_wrong_idx"])
        assert rec["test_ler"] == pytest.approx(rec["n_errors"] / N_BIG)
        assert all(0 <= i < N_BIG for i in rec["test_wrong_idx"])
    assert m["n_errors"] > 0       # p=0.01 at d=3: MWPM does make errors
    wa, wm = set(a["test_wrong_idx"]), set(m["test_wrong_idx"])
    mc = bt["mcnemar_a_vs_mwpm"]["0"]
    assert mc["discordant_a_right_b_wrong"] == len(wm - wa)
    assert mc["discordant_a_wrong_b_right"] == len(wa - wm)
    assert "mcnemar_zx_vs_raw" not in bt


def test_big_test_shots_are_fresh_and_deterministic():
    def fired(seed, n=300):
        return np.stack([g.x[:, 0].numpy() for g in qec_run._fresh_shots(3, 0.01, n, seed)])
    big = fired(SPLIT_SEED + 2)
    assert np.array_equal(big, fired(SPLIT_SEED + 2))            # reproducible
    base, _, _ = qec_run.make_datasets(3, 0.01, 1000, SPLIT_SEED)
    pool = np.stack([g.x[:, 0].numpy() for g in base[:300]])
    extra = fired(SPLIT_SEED + 1)                                # extra-train seed
    assert not np.array_equal(big, pool)
    assert not np.array_equal(big, extra)


@pytest.mark.parametrize("extra", [
    ["--big-test-shots", "100", "--big-test-seed", str(SPLIT_SEED)],
    ["--big-test-shots", "100", "--big-test-seed", str(SPLIT_SEED + 1)],
    ["--train-shots", "2000", "--extra-train-seed", "7",
     "--big-test-shots", "100", "--big-test-seed", "7"],
    ["--big-test-seed", "99"],
    ["--big-test-shots", "0"],
])
def test_bad_big_test_options_rejected(extra, tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, *extra)
