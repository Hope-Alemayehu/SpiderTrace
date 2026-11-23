"""Fresh training shots every epoch (qec_run.py --fresh-train).

Each epoch e (1-based) trains on n NEW shots drawn from stim with sampler seed
base + e. The draw is exactly the one qec_run._fresh_shots(d, p, n, base + e)
makes (same DEM, same seeded sampler, the same sequence of sample() calls in
chunks of qec_run._FRESH_SHOT_CHUNK), so the shots are the same; only the way
they become batches differs:

  * No per-shot torch_geometric Data objects. Every shot shares one decoding graph
    and only node column 0 (the fired bit of the detector nodes) changes, so a
    batch is the fixed template graph repeated B times with the fired bits written
    in. The result equals torch_geometric's Batch.from_data_list over the same
    shots in sample order (tests/test_fresh_train.py checks every field).
  * Aux targets (raw / ZX) are the product of the fired DEM errors' Pauli strings
    with phases ignored, i.e. an XOR of their symplectic (x, z) bit vectors. That
    is one sparse matrix product per chunk instead of a Python loop per shot.
  * Batches come in sample order, without shuffling: fresh shots are i.i.d.
    already, so shuffling would change nothing statistically, and sample order
    makes "the same batches as the current code builds" exactly checkable.

Stim's seeded sampling is reproducible on one platform but not across platforms
(see results/LOG.md, 2026-11-23), so a fresh-train run is reproducible from its
recorded seed schedule on the same kind of machine.
"""
import numpy as np

# Default seed schedule: base = FRESH_SEED_OFFSET + 1000 * split_seed, epoch e uses
# base + e. With split_seed 12345 that is 13,345,001 onwards, far from the split,
# extra-train, big-test and eval-set seeds (12345, 12346, 12347). qec_run still
# checks every schedule against those seeds and refuses any collision.
FRESH_SEED_OFFSET = 1_000_000


def resolve_fresh_seed_base(split_seed, fresh_train_seed=None):
    return (FRESH_SEED_OFFSET + 1000 * split_seed if fresh_train_seed is None
            else fresh_train_seed)


def fresh_seed_schedule(seed_base, epochs):
    """Sampler seed of each epoch, epochs 1..epochs."""
    return [seed_base + e for e in range(1, epochs + 1)]


def sample_chunks(d, p, n, seed, chunk):
    """Yield (dets, obs, errs) for n shots from a sampler seeded with `seed`, in
    sample() calls of `chunk` shots (the last chunk truncated). The same calls
    as qec_run._fresh_shots, so the same shots."""
    from qec_zx_dataset import build_circuit
    dem = build_circuit(d, p).detector_error_model(decompose_errors=False,
                                                   flatten_loops=True)
    sampler = dem.compile_sampler(seed=seed)
    have = 0
    while have < n:
        dets, obs, errs = sampler.sample(shots=chunk, return_errors=True,
                                         recorded_errors_to_replay=None)
        take = min(chunk, n - have)
        yield dets[:take], obs[:take], errs[:take]
        have += chunk


_PAULI_TABLE_CACHE = {}


def pauli_bit_tables(d, p):
    """Per DEM error, the symplectic bits of its raw and ZX Pauli strings:
    {"raw_target": (X, Z), "zx_target": (X, Z)}, each (num_errors, num_qubits)
    uint8. From qec_zx_dataset.build_fault_tables, the source of the targets."""
    key = (d, p)
    if key not in _PAULI_TABLE_CACHE:
        from qec_zx_dataset import build_circuit, build_fault_tables
        tables, _ = build_fault_tables(build_circuit(d, p))
        out = {}
        for name, strings in (("raw_target", tables.raw_pauli),
                              ("zx_target", tables.zx_pauli)):
            xz = [ps.to_numpy() for ps in strings]
            out[name] = (np.array([x for x, _ in xz], dtype=np.uint8),
                         np.array([z for _, z in xz], dtype=np.uint8))
        _PAULI_TABLE_CACHE[key] = (out, tables.num_qubits)
    return _PAULI_TABLE_CACHE[key]


def pauli_index_targets(errs, x_table, z_table):
    """Pauli class per shot and qubit (0=I, 1=X, 2=Y, 3=Z, stim's convention),
    of the product of the fired errors' Pauli strings. errs: (shots, num_errors)."""
    from scipy import sparse
    E = sparse.csr_matrix(errs.astype(np.int32))
    x = np.asarray(E @ x_table.astype(np.int32)) % 2
    z = np.asarray(E @ z_table.astype(np.int32)) % 2
    return (x + z * (3 - 2 * x)).astype(np.int8)    # (0,0)I (1,0)X (1,1)Y (0,1)Z


class FreshTrainSource:
    """Iterable of training batches; each pass is one epoch of fresh shots.

    train_model calls set_fresh_epoch(e) before epoch e, so a resumed run picks up
    the right part of the schedule. One epoch's arrays are kept in memory (for
    d=5 and 300k shots: about 36 MB of detector bits, plus 15 MB of targets for
    an aux arm); the large per-shot error matrix is only held one chunk at a time.

    template: one shot of the graph the arm trains on (after its graph variant),
    as a torch_geometric Data. Its node column 0 is replaced by each shot's fired
    bits; everything else is copied unchanged.
    """

    def __init__(self, d, p, n_per_epoch, seed_base, template, batch_size,
                 aux_key=None, chunk=10000):
        import torch
        from qec_zx_dataset import build_circuit
        self.d, self.p, self.n = d, p, int(n_per_epoch)
        self.seed_base, self.batch_size, self.chunk = seed_base, batch_size, chunk
        self.aux_key = aux_key
        self.num_detectors = build_circuit(d, p).num_detectors
        x = template.x.clone()
        nd = self.num_detectors
        if not (x.size(0) > nd and bool((x[:nd, 4] == 0).all())
                and bool((x[nd:, 4] == 1).all())):
            raise ValueError("template graph: detector nodes must come first, then "
                             "the boundary node(s)")
        x[:, 0] = 0.0
        self.x_t, self.ei_t, self.ea_t = x, template.edge_index, template.edge_attr
        self.n_nodes, self.n_edges = x.size(0), self.ei_t.size(1)
        if aux_key is not None:
            tables, self.num_qubits = pauli_bit_tables(d, p)
            self._xz = tables[aux_key]
        self._epoch, self._data, self._data_epoch = 1, None, None
        self._static = {}
        self._torch = torch

    def __len__(self):
        return self.n

    def seed_for(self, epoch):
        return self.seed_base + epoch

    def set_fresh_epoch(self, epoch):
        self._epoch = int(epoch)

    def epoch_arrays(self, epoch):
        """(fired uint8 (n, nd), y int64 (n,), targets int8 (n, num_qubits) or None)."""
        if self._data_epoch != epoch:
            fired, ys, tg = [], [], []
            for dets, obs, errs in sample_chunks(self.d, self.p, self.n,
                                                 self.seed_for(epoch), self.chunk):
                fired.append(dets.astype(np.uint8))
                ys.append(obs[:, 0].astype(np.int64))
                if self.aux_key is not None:
                    tg.append(pauli_index_targets(errs, *self._xz))
            self._data = (np.concatenate(fired), np.concatenate(ys),
                          np.concatenate(tg) if tg else None)
            self._data_epoch = epoch
        return self._data

    def _static_for(self, B):
        if B not in self._static:
            torch = self._torch
            offs = torch.arange(B, dtype=torch.long) * self.n_nodes
            self._static[B] = {
                "edge_index": self.ei_t.repeat(1, B) + offs.repeat_interleave(self.n_edges),
                "edge_attr": self.ea_t.repeat(B, 1),
                "batch": torch.arange(B, dtype=torch.long).repeat_interleave(self.n_nodes),
                "ptr": torch.arange(B + 1, dtype=torch.long) * self.n_nodes,
            }
        return self._static[B]

    def make_batch(self, fired, y, targets=None):
        """One batch (a torch_geometric Data with Batch's fields) from B shots."""
        torch = self._torch
        from torch_geometric.data import Data
        B = len(y)
        st = self._static_for(B)
        x = self.x_t.repeat(B, 1)
        x.view(B, self.n_nodes, -1)[:, :self.num_detectors, 0] = torch.from_numpy(
            np.ascontiguousarray(fired, dtype=np.float32))
        fields = dict(x=x, edge_index=st["edge_index"], edge_attr=st["edge_attr"],
                      y=torch.from_numpy(np.ascontiguousarray(y)),
                      batch=st["batch"], ptr=st["ptr"])
        if targets is not None:
            fields[self.aux_key] = torch.nn.functional.one_hot(
                torch.from_numpy(targets.astype(np.int64)), 4).float()
        return Data(**fields)

    def __iter__(self):
        fired, y, tg = self.epoch_arrays(self._epoch)
        bs = self.batch_size
        for a in range(0, self.n, bs):
            yield self.make_batch(fired[a:a + bs], y[a:a + bs],
                                  None if tg is None else tg[a:a + bs])

    def class_counts(self, epoch=1, qubit_idx=None):
        """Pauli class counts of the aux targets in one epoch's shots (for
        --aux-class-weight; the fixed-train path counts its training split)."""
        _, _, tg = self.epoch_arrays(epoch)
        if qubit_idx is not None:
            tg = tg[:, np.asarray(qubit_idx)]
        return np.bincount(tg.reshape(-1).astype(np.int64), minlength=4)
