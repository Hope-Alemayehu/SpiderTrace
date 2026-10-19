"""Per-epoch checkpointing and --resume (qec_run.RunCheckpoint).

A run is "crashed" by making RunCheckpoint.save_progress raise right after it
writes its Nth checkpoint, i.e. after N completed epochs. The run is then resumed
with --resume and compared with an uninterrupted run of the same command. On CPU
the two must agree exactly: every LER, McNemar count, wrong-shot index and
per-epoch loss value. Only provenance, wall times and the resume record differ.

Config: d=3, 1,000 shots, arms A and ZX (lambda 0.1 and 0.5), 4 epochs, so the run
trains 3 models. The crash comes after 6 epochs: A has finished (4 epochs) and
ZX at lambda 0.1 is halfway (2 of 4).

    pytest tests/test_checkpoint_resume.py
"""
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import qec_run  # noqa: E402

ARGS = ["qec_run.py", "--d", "3", "--p", "0.01", "--shots", "1000", "--seeds", "0",
        "--lambdas", "0.1", "0.5", "--arms", "A", "ZX", "--epochs", "4",
        "--device", "cpu"]
CRASH_AFTER_EPOCHS = 6
# Allowed to differ between an uninterrupted run and a resumed one.
VOLATILE = {"provenance", "checkpoint", "wall_time_s", "resumed_after_epoch"}


class Crash(Exception):
    pass


def run_main(mp, outdir, *extra):
    mp.setattr(sys, "argv", ARGS + ["--outdir", str(outdir), *extra])
    qec_run.main()
    files = sorted(Path(outdir).glob("run_*.json"))
    assert len(files) == 1, files
    return json.loads(files[0].read_text())


def strip(obj):
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k not in VOLATILE}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


def checkpoint_dirs(outdir):
    return sorted(Path(outdir).glob("checkpoint_*"))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    full_dir = tmp_path_factory.mktemp("full")
    crash_dir = tmp_path_factory.mktemp("crash")
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        out["full"] = run_main(mp, full_dir)
        out["full_leftovers"] = checkpoint_dirs(full_dir)

        # Crash after CRASH_AFTER_EPOCHS saved epochs.
        real_save = qec_run.RunCheckpoint.save_progress
        calls = {"n": 0}

        def crashing_save(self, *a, **kw):
            real_save(self, *a, **kw)
            calls["n"] += 1
            if calls["n"] == CRASH_AFTER_EPOCHS:
                raise Crash()

        mp.setattr(qec_run.RunCheckpoint, "save_progress", crashing_save)
        with pytest.raises(Crash):
            run_main(mp, crash_dir)
        mp.setattr(qec_run.RunCheckpoint, "save_progress", real_save)

        (ckpt,) = checkpoint_dirs(crash_dir)
        out["partial"] = json.loads((ckpt / "partial.json").read_text())
        out["files_after_crash"] = sorted(p.name for p in ckpt.iterdir())

        # Rerunning without --resume must refuse rather than overwrite progress.
        with pytest.raises(RuntimeError, match="already exists"):
            run_main(mp, crash_dir)

        out["resumed"] = run_main(mp, crash_dir, "--resume")
        out["resumed_leftovers"] = checkpoint_dirs(crash_dir)
    return out


def test_finished_run_leaves_no_checkpoint(runs):
    assert runs["full_leftovers"] == []
    assert runs["resumed_leftovers"] == []
    assert "checkpoint" not in runs["full"]


def test_crash_loses_at_most_the_epoch_in_progress(runs):
    part = runs["partial"]
    assert list(part["done"]) == ["A_seed0_lam0.0"]
    assert part["done"]["A_seed0_lam0.0"]["train"]["epochs_run"] == 4
    prog = part["in_progress"]
    assert prog["model"] == "ZX_seed0_lam0.1"
    assert prog["epochs_done"] == CRASH_AFTER_EPOCHS - 4
    assert len(prog["val_loss"]) == CRASH_AFTER_EPOCHS - 4
    assert runs["files_after_crash"] == [
        "done_A_seed0_lam0.0.pt", "partial.json", "progress_ZX_seed0_lam0.1.pt"]


def test_resumed_run_matches_uninterrupted_run(runs):
    full, resumed = runs["full"], runs["resumed"]
    assert strip(resumed) == strip(full)
    # And the comparison is not vacuous: the curves and wrong-shot lists exist.
    zx = resumed["arms"]["ZX"]["lambda_grid"]
    assert [len(g["train"]["val_loss"]) for g in zx] == [4, 4]
    assert "test_wrong_idx" in resumed["arms"]["A"]["selected_per_seed"][0]


def test_resume_is_recorded(runs):
    rec = runs["resumed"]["checkpoint"]
    assert len(rec["resumes"]) == 1
    assert rec["provenance_at_start"]["config"]["resume"] is False
    grid = {g["lambda"]: g["train"] for g in runs["resumed"]["arms"]["ZX"]["lambda_grid"]}
    assert grid[0.1]["resumed_after_epoch"] == CRASH_AFTER_EPOCHS - 4
    assert "resumed_after_epoch" not in grid[0.5]   # trained after the resume


def test_resume_without_checkpoint_refused(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="no checkpoint"):
        run_main(monkeypatch, tmp_path, "--resume")


def test_checkpoint_tag_ignores_only_device_outdir_resume():
    base = {"d": 3, "p": 0.01, "epochs": 4, "device": "cpu", "outdir": "a",
            "resume": False}
    same = dict(base, device="cuda", outdir="b", resume=True)
    assert qec_run.checkpoint_tag(base) == qec_run.checkpoint_tag(same)
    assert qec_run.checkpoint_tag(base) != qec_run.checkpoint_tag(dict(base, epochs=5))
