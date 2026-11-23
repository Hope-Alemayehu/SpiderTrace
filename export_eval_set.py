#!/usr/bin/env python3
"""Sample the big test set once and store it, for qec_run.py --eval-set.

    python export_eval_set.py --d 5 --p 0.003 --shots 200000 \
        --expect-mwpm-errors 664 --out eval_d5_p0.003_seed12347_200k.npz

The shots are drawn exactly as qec_run.py --big-test-shots draws them: the same DEM,
a sampler seeded with --big-test-seed (default: split seed + 2 = 12347) and the same
sequence of sample() calls (qec_run._fresh_samples). Stim's seeded sampling is not
portable across platforms, so the canonical file is the one exported on Kaggle Linux,
whose big test MWPM decodes with 664 errors at d=5, p=0.003.

Needs stim, pymatching, numpy and scipy only: no torch, no GPU.

The file (format in qec_run.py, "Stored evaluation set") holds the detector bits,
the observable labels, the fired DEM errors (so the aux targets are rebuilt
exactly), MWPM's wrong shots, and a JSON header with the provenance: commit, dirty
flag, platform, stim / pymatching versions, seed, and the DEM fingerprint.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

import qec_run


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--shots", type=int, default=200000)
    ap.add_argument("--split-seed", type=int, default=12345,
                    help="only used to derive the default --big-test-seed")
    ap.add_argument("--big-test-seed", type=int, default=None,
                    help="sampler seed. Default: split seed + 2, as in qec_run.py.")
    ap.add_argument("--expect-mwpm-errors", type=int, default=None,
                    help="fail, and write nothing, unless plain MWPM makes exactly this "
                         "many errors on the sampled shots (664 for the canonical "
                         "d=5, p=0.003 big test).")
    ap.add_argument("--out", required=True, help="output .npz path")
    args = ap.parse_args()
    if args.shots < 1:
        ap.error("--shots must be >= 1")
    out = Path(args.out)
    if out.suffix != ".npz":
        ap.error("--out must end in .npz")
    if out.exists():
        ap.error(f"{out} exists; refusing to overwrite an eval set")

    seed = qec_run.resolve_big_test_seed(args.split_seed, args.big_test_seed)
    prov = qec_run.provenance(vars(args).copy())
    t0 = time.perf_counter()
    tmp = out.with_name(out.stem + ".partial.npz")
    meta = qec_run.write_eval_set(tmp, args.d, args.p, args.shots, seed,
                                  extra_meta={"provenance": prov})
    if args.expect_mwpm_errors is not None and meta["mwpm_n_errors"] != args.expect_mwpm_errors:
        tmp.unlink()
        sys.exit(f"MWPM made {meta['mwpm_n_errors']} errors on these shots, expected "
                 f"{args.expect_mwpm_errors}. These are not the canonical shots (stim's "
                 "seeded sampling differs between platforms). Nothing written.")
    tmp.rename(out)

    # Read it back through the checks qec_run --eval-set runs (DEM fingerprint,
    # detector/error consistency, MWPM error count), so an unusable file is never
    # left behind silently. Building the PyG graphs is the only step skipped here.
    dets, obs, errs, meta2, sha, checks = qec_run.load_eval_set_arrays(out, args.d, args.p)
    assert len(dets) == args.shots and meta2 == meta
    assert all(checks.values()), checks      # same machine: even the strict fingerprint
    print(json.dumps({k: v for k, v in meta.items() if k != "provenance"}, indent=1))
    print(f"commit {prov['git_commit']} (dirty: {prov['git_dirty']}), "
          f"platform {prov['platform']}, stim {prov['library_versions']['stim']}")
    print(f"Wrote {out}: {out.stat().st_size / 1e6:.2f} MB, sha256 {sha}, "
          f"{time.perf_counter() - t0:.1f} s. Verified by reading it back.")


if __name__ == "__main__":
    main()
