"""Training options --lr, --lr-schedule, --lr-min, --grad-clip, --patience.

  * Defaults: constant lr 1e-3, no scheduler, no clipping, no grad-norm records.
  * Cosine: the recorded per-epoch lr follows lr_min + (lr - lr_min)(1 + cos(pi e/T))/2.
  * Clipping: every optimizer step sees a global grad norm <= grad_clip, and the
    per-epoch grad-norm statistics are recorded.
  * Patience: with lr = 0 the weights never change, so val loss is constant and
    the run must stop at exactly epoch 1 + patience.
  * Resume: a cosine + clipped run crashed mid-training and resumed matches the
    uninterrupted run exactly (CPU), lr curve included. The scheduler state is
    in the checkpoint.
  * Bad values are rejected on the command line.

Small config (d=3, 1,000 shots, GNN-A) so it runs in seconds on CPU.

    pytest tests/test_training_options.py
"""
import json
import math
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import qec_run  # noqa: E402

D, P, SHOTS, SPLIT_SEED = 3, 0.01, 1000, 12345
ARGS = ["qec_run.py", "--d", str(D), "--p", str(P), "--shots", str(SHOTS),
        "--seeds", "0", "--arms", "A", "--epochs", "4", "--device", "cpu"]


def run_train(epochs, **opts):
    train_ds, val_ds, _ = qec_run.make_datasets(D, P, SHOTS, SPLIT_SEED)
    qec_run._seed_all(0)
    model = qec_run.build_model("A", d=D)
    return qec_run.train(model, train_ds, val_ds, lambda_aux=0.0, epochs=epochs,
                         seed=0, device="cpu", **opts)


def run_main(mp, outdir, *extra):
    mp.setattr(sys, "argv", ARGS + ["--outdir", str(outdir), *extra])
    qec_run.main()
    (f,) = Path(outdir).glob("run_*.json")
    return json.loads(f.read_text())


def test_defaults_constant_lr_no_clipping():
    _, info = run_train(3)
    assert info["optim"] == {"lr": 1e-3, "lr_schedule": "constant", "lr_min": 1e-5,
                             "grad_clip": None, "patience": 10}
    assert info["lr"] == [1e-3] * 3
    assert not any(k in info for k in ("grad_norm_mean", "grad_norm_max",
                                       "clipped_frac"))


def test_cosine_lr_curve():
    lr, lr_min, T = 1e-3, 1e-5, 6
    _, info = run_train(T, lr=lr, lr_schedule="cosine", lr_min=lr_min, patience=T)
    expected = [lr_min + (lr - lr_min) * (1 + math.cos(math.pi * e / T)) / 2
                for e in range(T)]
    assert info["epochs_run"] == T
    assert info["lr"] == pytest.approx(expected, rel=1e-9)
    assert info["lr"][0] == lr and info["lr"][-1] < 0.1 * lr


def test_grad_clip_bounds_every_step(monkeypatch):
    clip = 0.05
    seen = []
    real_step = torch.optim.Adam.step

    def recording_step(self, *a, **kw):
        grads = [p.grad for g in self.param_groups for p in g["params"]
                 if p.grad is not None]
        seen.append(float(torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(g) for g in grads]))))
        return real_step(self, *a, **kw)

    monkeypatch.setattr(torch.optim.Adam, "step", recording_step)
    _, info = run_train(2, grad_clip=clip)
    assert seen and max(seen) <= clip * (1 + 1e-5)
    # The clip was active: the pre-clip norms it recorded exceed it.
    assert len(info["grad_norm_max"]) == 2 and max(info["grad_norm_max"]) > clip
    assert 0 < max(info["clipped_frac"]) <= 1
    assert all(m <= x for m, x in zip(info["grad_norm_mean"], info["grad_norm_max"]))

    seen.clear()
    run_train(2)                       # no clipping: norms do exceed `clip`
    assert max(seen) > clip


@pytest.mark.parametrize("patience", [2, 4])
def test_patience_controls_early_stop(patience):
    _, info = run_train(20, lr=0.0, patience=patience)
    assert info["epochs_run"] == 1 + patience
    assert info["best_epoch"] == 1


@pytest.fixture(scope="module")
def resume_runs(tmp_path_factory):
    extra = ["--lr-schedule", "cosine", "--lr-min", "1e-5", "--grad-clip", "0.5",
             "--patience", "4"]
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        out["full"] = run_main(mp, tmp_path_factory.mktemp("full"), *extra)

        crash_dir = tmp_path_factory.mktemp("crash")
        real_save = qec_run.RunCheckpoint.save_progress

        class Crash(Exception):
            pass

        def crashing_save(self, key, state, elapsed_s):
            real_save(self, key, state, elapsed_s)
            if state["epoch"] == 2:
                raise Crash()

        mp.setattr(qec_run.RunCheckpoint, "save_progress", crashing_save)
        with pytest.raises(Crash):
            run_main(mp, crash_dir, *extra)
        mp.setattr(qec_run.RunCheckpoint, "save_progress", real_save)
        (ckpt,) = crash_dir.glob("checkpoint_*")
        out["state"] = torch.load(ckpt / "progress_A_seed0_lam0.0.pt",
                                  map_location="cpu", weights_only=False)
        out["resumed"] = run_main(mp, crash_dir, *extra, "--resume")
    return out


def strip(obj):
    volatile = {"provenance", "checkpoint", "wall_time_s", "resumed_after_epoch"}
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k not in volatile}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


def test_scheduler_state_is_checkpointed(resume_runs):
    st = resume_runs["state"]
    assert st["epoch"] == 2
    assert st["sched_state"] is not None and st["sched_state"]["last_epoch"] == 2


def test_resume_with_cosine_and_clipping_matches_uninterrupted(resume_runs):
    full, resumed = resume_runs["full"], resume_runs["resumed"]
    assert strip(resumed) == strip(full)
    t = resumed["arms"]["A"]["lambda_grid"][0]["train"]
    assert t["resumed_after_epoch"] == 2
    assert len(t["lr"]) == 4 and t["lr"][0] > t["lr"][-1]
    assert len(t["clipped_frac"]) == 4
    assert resumed["provenance"]["config"]["lr_schedule"] == "cosine"
    assert resumed["provenance"]["config"]["grad_clip"] == 0.5


@pytest.mark.parametrize("extra", [
    ["--lr", "0"],
    ["--grad-clip", "0"],
    ["--patience", "0"],
    ["--lr-schedule", "cosine", "--lr", "1e-3", "--lr-min", "1e-2"],
])
def test_bad_training_options_rejected(extra, tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, *extra)
