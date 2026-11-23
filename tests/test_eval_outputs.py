"""Correlated-matching baseline, stored eval sets (--eval-set) and saved models.

  * Defaults: saving models changes no result; the JSON gains only the
    correlated-matching blocks and the vs-corr McNemar keys.
  * Correlated matching: blocks present and consistent on a tiny d=3 run, and
    the decoder really uses correlations (fewer errors than MWPM at d=5, p=0.003).
  * --eval-set: an exported file rebuilds exactly the shots --big-test-shots
    samples, the run's big_test block matches, and a tampered or mismatched file
    fails loudly.
  * Saved models: one weights file per arm and seed, which reproduces the run's
    big-test predictions, and probabilities consistent with the JSON.

Small configs (d=3, p=0.01, 1,000 shots, 2 epochs, 2,000 big-test shots), CPU.

    pytest tests/test_eval_outputs.py
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

D, P, N_BIG, SEED = 3, 0.01, 2000, 12347
ARGS = ["qec_run.py", "--d", str(D), "--p", str(P), "--shots", "1000", "--seeds", "0", "1",
        "--arms", "A", "ZX", "--lambdas", "0.1", "--epochs", "2", "--device", "cpu"]
VOLATILE = {"provenance", "wall_time_s", "eval_time_s", "build_time_s", "decode_time_s"}


def run_main(mp, outdir, *extra):
    mp.setattr(sys, "argv", ARGS + ["--outdir", str(outdir), *extra])
    qec_run.main()
    (f,) = Path(outdir).glob("run_*.json")
    return f, json.loads(f.read_text())


def strip(obj, drop=()):
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k not in VOLATILE and k not in drop}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


@pytest.fixture(scope="module")
def eval_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("evalset") / "eval.npz"
    qec_run.write_eval_set(path, D, P, N_BIG, SEED, extra_meta={"note": "test"})
    return path


@pytest.fixture(scope="module")
def runs(tmp_path_factory, eval_file):
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        out["default"] = run_main(mp, tmp_path_factory.mktemp("default"),
                                  "--big-test-shots", str(N_BIG))
        out["nosave"] = run_main(mp, tmp_path_factory.mktemp("nosave"),
                                 "--big-test-shots", str(N_BIG), "--no-save-models")
        out["evalset"] = run_main(mp, tmp_path_factory.mktemp("evalset_run"),
                                  "--eval-set", str(eval_file))
    return out


# ---- defaults ----------------------------------------------------------------------
def test_saving_models_changes_no_result(runs):
    (_, a), (_, b) = runs["default"], runs["nosave"]
    assert strip(a) == strip(b)
    assert a["provenance"]["config"]["no_save_models"] is False
    assert a["provenance"]["config"]["eval_set"] is None


def test_no_save_models_writes_only_the_json(runs):
    f, _ = runs["nosave"]
    assert sorted(p.name for p in f.parent.iterdir()) == [f.name]


def test_json_gains_only_the_new_blocks(runs):
    _, r = runs["default"]
    new = {"correlated_matching", "mcnemar_a_vs_corr", "mcnemar_zx_vs_corr"}
    assert new <= set(r) and new <= set(r["big_test"])
    assert "mcnemar_raw_vs_corr" not in r          # only arms that were run
    for key in ("mwpm", "mcnemar_zx_vs_mwpm", "mcnemar_a_vs_mwpm", "mcnemar_zx_vs_raw"):
        assert (key in r) == (key != "mcnemar_zx_vs_raw")


# ---- correlated matching -----------------------------------------------------------
@pytest.mark.parametrize("where", ["primary", "big"])
def test_correlated_matching_block_is_consistent(runs, where):
    _, r = runs["default"]
    blk = r if where == "primary" else r["big_test"]
    n = r["mwpm"]["n_test"] if where == "primary" else r["big_test"]["n_test"]
    c, m = blk["correlated_matching"], blk["mwpm"]
    assert c["n_errors"] == len(c["test_wrong_idx"]) > 0
    assert c["test_ler"] == pytest.approx(c["n_errors"] / n)
    assert all(0 <= i < n for i in c["test_wrong_idx"])
    assert c["decode_time_s"] >= 0
    # Sane: a decoder of the same family, close to MWPM on the same shots.
    assert abs(c["n_errors"] - m["n_errors"]) <= 0.25 * m["n_errors"] + 5
    for arm in ("A", "ZX"):
        wa = set(r["arms"][arm]["selected_per_seed"][0]["test_wrong_idx"]) \
            if where == "primary" else set(blk["arms"][arm]["0"]["test_wrong_idx"])
        wc = set(c["test_wrong_idx"])
        mc = blk[f"mcnemar_{arm.lower()}_vs_corr"]["0"]
        assert mc["discordant_a_right_b_wrong"] == len(wc - wa)
        assert mc["discordant_a_wrong_b_right"] == len(wa - wc)
        assert 0.0 <= mc["p_value"] <= 1.0


def test_correlated_matching_uses_correlations():
    """At d=5, p=0.003 correlated matching beats plain MWPM clearly (about 25% fewer
    errors on 200k shots), so a set-up that silently fell back to plain matching
    would show up here."""
    ds = qec_run._fresh_shots(5, 0.003, 20000, 777)
    mwpm_ler, _, y1 = qec_run.mwpm_on_test(5, 0.003, ds)
    corr_ler, _, y2, _ = qec_run.correlated_matching_on_test(5, 0.003, ds)
    assert np.array_equal(y1, y2)
    assert corr_ler < 0.9 * mwpm_ler


# ---- --eval-set --------------------------------------------------------------------
def test_eval_set_rebuilds_the_same_shots(eval_file):
    loaded, meta, _, checks = qec_run.read_eval_set(eval_file, D, P)
    assert all(checks.values()), checks       # same machine: strict fingerprint too
    fresh = qec_run._fresh_shots(D, P, N_BIG, SEED)
    assert meta["n_shots"] == N_BIG and meta["seed"] == SEED and meta["note"] == "test"
    assert len(loaded) == len(fresh)
    for g, h in zip(loaded, fresh):
        for k in ("x", "edge_index", "edge_attr", "y", "raw_target", "zx_target"):
            assert torch.equal(g[k], h[k]), k


def test_eval_set_run_matches_sampled_big_test(runs, eval_file):
    (_, a), (_, e) = runs["default"], runs["evalset"]
    ba, be = a["big_test"], e["big_test"]
    assert be["eval_set"]["meta"]["seed"] == SEED == ba["seed"]
    assert be["eval_set"]["path"] == str(eval_file)
    assert strip(be, drop=("eval_set",)) == strip(ba)
    assert strip(e, drop=("big_test",)) == strip(a, drop=("big_test",))


def _rewrite(src, dst, **changes):
    with np.load(src, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays["meta"]))
    for k, v in changes.items():
        if k in meta:
            meta[k] = v
        else:
            arrays[k] = v
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(dst, **arrays)
    return dst


def test_eval_set_mwpm_mismatch_fails_loudly(eval_file, tmp_path):
    with np.load(eval_file) as z:
        n = int(json.loads(str(z["meta"]))["mwpm_n_errors"])
        wrong = z["mwpm_wrong_idx"]
    bad = _rewrite(eval_file, tmp_path / "count.npz", mwpm_n_errors=n + 1)
    with pytest.raises(RuntimeError, match="MWPM on the stored eval set"):
        qec_run.load_eval_set_arrays(bad, D, P)
    bad = _rewrite(eval_file, tmp_path / "idx.npz", mwpm_wrong_idx=wrong[::-1].copy())
    with pytest.raises(RuntimeError, match="MWPM on the stored eval set"):
        qec_run.load_eval_set_arrays(bad, D, P)


def test_eval_set_corrupted_bits_fail_loudly(eval_file, tmp_path):
    with np.load(eval_file) as z:
        dets = z["detectors"].copy()
    dets[0, 0] ^= 1
    bad = _rewrite(eval_file, tmp_path / "bits.npz", detectors=dets)
    with pytest.raises(RuntimeError, match="inconsistent"):
        qec_run.load_eval_set_arrays(bad, D, P)


def test_fingerprint_mismatch_only_warns_when_content_checks_pass(eval_file, tmp_path,
                                                                  capsys):
    """A different str(dem) fingerprint (platform rounding of the printed
    probabilities) is tolerated, with a warning, when the bits and MWPM match."""
    other = _rewrite(eval_file, tmp_path / "fp.npz", dem_sha256="0" * 64)
    with pytest.warns(UserWarning, match="strict DEM fingerprint"):
        dets, _, _, meta, _, checks = qec_run.load_eval_set_arrays(other, D, P)
    assert checks == {"format_d_p_shots": True, "dem_fingerprint": False,
                      "bits_consistent": True, "mwpm_reproduced": True}
    assert len(dets) == N_BIG
    out = capsys.readouterr().out
    assert "dem_fingerprint FAIL" in out and "mwpm_reproduced pass" in out


@pytest.mark.parametrize("break_what", ["mwpm", "bits"])
def test_fingerprint_mismatch_with_failed_content_check_still_fails(
        break_what, eval_file, tmp_path, recwarn):
    with np.load(eval_file) as z:
        n = int(json.loads(str(z["meta"]))["mwpm_n_errors"])
        dets = z["detectors"].copy()
    dets[0, 0] ^= 1
    change = {"mwpm_n_errors": n + 1} if break_what == "mwpm" else {"detectors": dets}
    bad = _rewrite(eval_file, tmp_path / "both.npz", dem_sha256="0" * 64, **change)
    with pytest.raises(RuntimeError, match="strict DEM fingerprint also differs"):
        qec_run.load_eval_set_arrays(bad, D, P)
    assert not [w for w in recwarn if "loading anyway" in str(w.message)]


CANONICAL = Path(ROOT) / "data" / "eval_d5_p0.003_seed12347_200k.npz"


@pytest.mark.skipif(not CANONICAL.exists(), reason="canonical eval set not present")
def test_canonical_eval_set_loads():
    """The committed big test set (Kaggle Linux, MWPM 664) passes the content checks
    on any platform; the strict fingerprint may differ off Linux (warning only)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dets, obs, errs, meta, sha, checks = qec_run.load_eval_set_arrays(CANONICAL, 5, 0.003)
    assert sha == "4b9504da61a4404c3a07bed770cb33c6568a8c014f45d20562e6d5fabc292772"
    assert checks["bits_consistent"] and checks["mwpm_reproduced"]
    assert (meta["n_shots"], meta["seed"], meta["mwpm_n_errors"]) == (200000, 12347, 664)


def test_eval_set_wrong_code_rejected(eval_file):
    with pytest.raises(RuntimeError, match="is for d=3"):
        qec_run.load_eval_set_arrays(eval_file, 5, P)


@pytest.mark.parametrize("extra", [["--big-test-shots", "100"], ["--big-test-seed", "7"]])
def test_eval_set_excludes_big_test_flags(extra, eval_file, tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, "--eval-set", str(eval_file), *extra)


def test_export_script_refuses_wrong_mwpm_count(tmp_path, monkeypatch):
    import export_eval_set
    out = tmp_path / "e.npz"
    monkeypatch.setattr(sys, "argv", ["export_eval_set.py", "--d", "3", "--p", "0.01",
                                      "--shots", "500", "--expect-mwpm-errors", "999999",
                                      "--out", str(out)])
    with pytest.raises(SystemExit):
        export_eval_set.main()
    assert list(tmp_path.iterdir()) == []


def test_export_script_writes_a_verified_file(tmp_path, monkeypatch):
    import export_eval_set
    out = tmp_path / "e.npz"
    monkeypatch.setattr(sys, "argv", ["export_eval_set.py", "--d", "3", "--p", "0.01",
                                      "--shots", "500", "--out", str(out)])
    export_eval_set.main()
    dets, obs, errs, meta, _, _ = qec_run.load_eval_set_arrays(out, D, P)
    assert meta["seed"] == SEED and len(dets) == 500
    assert meta["provenance"]["library_versions"]["stim"]
    assert "git_commit" in meta["provenance"]


# ---- saved models and probabilities ------------------------------------------------
def test_saved_weights_reproduce_big_test_predictions(runs):
    f, r = runs["default"]
    wdir = f.with_name(f.stem + "_weights")
    assert sorted(p.name for p in wdir.iterdir()) == [
        "A_seed0.pt", "A_seed1.pt", "ZX_seed0.pt", "ZX_seed1.pt"]
    big_ds = qec_run._fresh_shots(D, P, N_BIG, SEED)
    for kind in ("A", "ZX"):
        for seed in (0, 1):
            ck = torch.load(wdir / f"{kind}_seed{seed}.pt", weights_only=False)
            assert (ck["arm"], ck["seed"]) == (kind, seed)
            sel = r["arms"][kind]["selected_per_seed"][seed]
            assert ck["selected_lambda"] == sel["selected_lambda"]
            hp = {k: v for k, v in ck["model"].items() if k != "n_params"}
            model = qec_run.build_model(kind, d=D, aux_qubits=ck["aux_qubits"], **hp)
            model.load_state_dict(ck["state_dict"])
            pred, y = qec_run.predict_flips(model, big_ds, "cpu")
            wrong = np.flatnonzero(pred != y).tolist()
            assert wrong == r["big_test"]["arms"][kind][str(seed)]["test_wrong_idx"]


def test_saved_probabilities_match_the_json(runs):
    f, r = runs["default"]
    with np.load(f.with_name(f.stem + "_bigtest_probs.npz"), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        assert meta["run_json"] == f.name and meta["n_test"] == N_BIG
        y = z["y_true"]
        for kind in ("A", "ZX"):
            for seed in (0, 1):
                prob = z[f"{kind}_seed{seed}"]
                assert prob.dtype == np.float32 and prob.shape == (N_BIG,)
                assert np.all((prob >= 0) & (prob <= 1))
                wrong = np.flatnonzero((prob > 0.5).astype(int) != y).tolist()
                assert wrong == r["big_test"]["arms"][kind][str(seed)]["test_wrong_idx"]
