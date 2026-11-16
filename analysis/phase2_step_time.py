"""Relative cost of one training step per ablation config (CPU, d=5, batch 256).

    python analysis/phase2_step_time.py  ->  analysis/phase2_results/step_time.json

Times collation of 256 graphs (Batch.from_data_list) plus forward, BCE, backward
and an Adam step, median of 30 steps after 5 warm-up steps, per config. Only the
RATIO to C0 is meant for planning; absolute CPU times do not transfer to a GPU.
"""
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.dont_write_bytecode = True

import torch  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

import graph_variants  # noqa: E402
import qec_run  # noqa: E402

torch.set_num_threads(4)
CONFIGS = {
    "C0": ({}, {}),
    "C2 --readout fired": (dict(readout="fired"), {}),
    "C5 --head mlp": (dict(head="mlp"), {}),
    "C4 --residual --norm layer": (dict(residual=True, norm="layer"), {}),
    "C4b --depth 8 --residual --norm layer": (dict(residual=True, norm="layer", num_layers=8), {}),
    "C6 --sublattice-types": ({}, dict(sublattice_types=True)),
    "C3 --boundary per-round": ({}, dict(boundary="per-round")),
}
train, _, _ = qec_run.make_datasets(5, 0.003, 4000, 12345)
pool = train[:256]
out = {}
for name, (mk, gk) in CONFIGS.items():
    ds = graph_variants.transform(pool, 5, 0.003, **gk)
    ni, ne = graph_variants.feature_dims(gk.get("sublattice_types", False))
    qec_run._seed_all(0)
    m = qec_run.build_model("A", d=5, node_in_dim=ni, edge_in_dim=ne, **mk)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    m.train()
    times = []
    for i in range(35):
        t0 = time.perf_counter()
        b = Batch.from_data_list(ds)
        opt.zero_grad()
        loss = bce(m(b)[0], b.y.float())
        loss.backward()
        opt.step()
        if i >= 5:
            times.append(time.perf_counter() - t0)
    out[name] = {"median_step_s": statistics.median(times),
                 "n_params": sum(t.numel() for t in m.parameters())}
base = out["C0"]["median_step_s"]
for name, r in out.items():
    r["ratio_to_C0"] = r["median_step_s"] / base
    print(f"{name:40s} {1e3 * r['median_step_s']:7.1f} ms  x{r['ratio_to_C0']:.2f}  params {r['n_params']:,}")
(HERE / "phase2_results").mkdir(exist_ok=True)
(HERE / "phase2_results" / "step_time.json").write_text(json.dumps(out, indent=1))
