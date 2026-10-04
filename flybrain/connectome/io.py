"""Readers for the MaleCNS v1.0 connectome as distributed by the garyb9/fly-drone
mirror (CC-BY-4.0, derived from Janelia FlyEM public release files).

Binary layout (little-endian), from crates/brain-core/src/core/format.rs of the mirror:

neurons.bin  16 B header: u32 magic "FLYN", u32 version, u32 count, u32 core_count
             24 B records: u64 body_id@0; f32 x,y,z@8,12,16; u16 group_id@20;
             u8 flags@22 (bit0 core, bit1 INHIBITORY, bit2 input, bit3 readout); u8 pad
graph.bin    32 B header: u32 magic "FLYG"@0, u32 version@4, u32 n_nodes@8,
             u64 edge_count@16, f32 w_norm@24
             then u32 offsets[n+1], u32 targets[E], i16 weights[E]  (CSR)
             row = presynaptic source; weight = synaptic contact count (edges >= 3
             contacts only); sign = source neuron's inhibitory flag (Dale's law).

IMPORTANT (see LIMITATIONS.md): "weights" are anatomical synaptic-contact counts standing
in for unmeasured physiological strengths. The inhibitory flag is a neurotransmitter
*prediction* (GABA/glutamate -> inhibitory; acetylcholine -> excitatory), not a measured
sign.
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

DATA_DIR = Path(os.environ.get("FLYBRAIN_DATA", Path(__file__).resolve().parents[2] / "data"))
NEURONS_MAGIC = 0x4E594C46
GRAPH_MAGIC = 0x4759_4C46

FLAG_CORE, FLAG_INHIBITORY, FLAG_INPUT, FLAG_READOUT = 1, 2, 4, 8


@dataclass
class Connectome:
    """Whole-CNS graph as CSR (rows = presynaptic) plus per-neuron annotations."""

    body_id: np.ndarray          # (N,) uint64
    pos: np.ndarray              # (N,3) float32, mirror scene coordinates
    group_id: np.ndarray         # (N,) uint16 superclass index into `groups`
    flags: np.ndarray            # (N,) uint8
    groups: list[str]
    cell_type: np.ndarray        # (N,) object str ('' if unknown)
    side: np.ndarray             # (N,) object str
    nt: np.ndarray               # (N,) object str, predicted neurotransmitter
    W: sp.csr_matrix             # (N,N) int32 contact counts, unsigned
    w_norm: float

    # ---- convenience -----------------------------------------------------------
    @property
    def n(self) -> int:
        return self.W.shape[0]

    @property
    def n_edges(self) -> int:
        return int(self.W.nnz)

    @property
    def inhibitory(self) -> np.ndarray:
        return (self.flags & FLAG_INHIBITORY) != 0

    @property
    def sign(self) -> np.ndarray:
        """+1 excitatory / -1 inhibitory per *source* neuron (Dale's law)."""
        return np.where(self.inhibitory, -1, 1).astype(np.int8)

    def signed(self) -> sp.csr_matrix:
        """Signed contact-count matrix: row i multiplied by sign of neuron i."""
        return sp.diags(self.sign.astype(np.float32)) @ self.W.astype(np.float32)

    def ids_of_type(self, pattern: str, regex: bool = False) -> np.ndarray:
        import re
        if regex:
            r = re.compile(pattern)
            return np.flatnonzero([bool(r.match(t)) for t in self.cell_type])
        return np.flatnonzero(self.cell_type == pattern)

    def subgraph(self, idx: np.ndarray) -> sp.csr_matrix:
        idx = np.asarray(idx)
        return self.W[idx][:, idx].tocsr()

    def type_counts(self) -> dict[str, int]:
        u, c = np.unique(self.cell_type, return_counts=True)
        return dict(zip(u.tolist(), c.tolist()))


def _read_neurons(path: Path):
    b = path.read_bytes()
    magic, version, count, core_count = struct.unpack_from("<IIII", b, 0)
    if magic != NEURONS_MAGIC:
        raise ValueError(f"bad neurons.bin magic {magic:#x}")
    rec = np.frombuffer(b, dtype=np.dtype([
        ("body_id", "<u8"), ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("group_id", "<u2"), ("flags", "u1"), ("pad", "u1")]), count=count, offset=16)
    pos = np.stack([rec["x"], rec["y"], rec["z"]], 1)
    return rec["body_id"].copy(), pos, rec["group_id"].copy(), rec["flags"].copy(), version, core_count


def _read_graph(path: Path):
    b = path.read_bytes()
    magic, version, n_nodes = struct.unpack_from("<III", b, 0)
    if magic != GRAPH_MAGIC:
        raise ValueError(f"bad graph.bin magic {magic:#x}")
    (edge_count,) = struct.unpack_from("<Q", b, 16)
    (w_norm,) = struct.unpack_from("<f", b, 24)
    o = 32
    offsets = np.frombuffer(b, dtype="<u4", count=n_nodes + 1, offset=o); o += 4 * (n_nodes + 1)
    targets = np.frombuffer(b, dtype="<u4", count=edge_count, offset=o); o += 4 * edge_count
    weights = np.frombuffer(b, dtype="<i2", count=edge_count, offset=o)
    W = sp.csr_matrix((weights.astype(np.int32), targets.astype(np.int32), offsets.astype(np.int64)),
                      shape=(n_nodes, n_nodes))
    return W, float(w_norm), version


def load_connectome(data_dir: Optional[Path] = None) -> Connectome:
    d = Path(data_dir or DATA_DIR) / "malecns"
    body_id, pos, group_id, flags, _, _ = _read_neurons(d / "neurons.bin")
    W, w_norm, _ = _read_graph(d / "graph.bin")
    if W.shape[0] != len(body_id):
        raise ValueError("neurons.bin / graph.bin node count mismatch")
    groups = json.loads((d / "groups.json").read_text())["groups"]
    cells = json.loads((d / "cells.json").read_text())
    if len(cells) != len(body_id):
        raise ValueError("cells.json length mismatch")
    cid = np.array([int(c["id"]) for c in cells], dtype=np.uint64)
    if not np.array_equal(cid, body_id):
        raise ValueError("cells.json order does not match neurons.bin")
    cell_type = np.array([c.get("type") or "" for c in cells], dtype=object)
    side = np.array([c.get("side") or "" for c in cells], dtype=object)
    nt = np.array([c.get("nt") or "" for c in cells], dtype=object)
    return Connectome(body_id, pos, group_id, flags, groups, cell_type, side, nt, W, w_norm)


def load_official_annotations(data_dir: Optional[Path] = None):
    """Official Janelia body-annotation table (feather). Used only to cross-check the
    mirror's cell-type labels; not required for experiments."""
    import pandas as pd
    d = Path(data_dir or DATA_DIR) / "malecns"
    return pd.read_feather(d / "body-annotations-male-cns-v1.0-minconf-0.5.feather")


def graph_summary(c: Connectome) -> dict:
    W = c.W
    outdeg = np.diff(W.indptr)
    indeg = np.bincount(W.indices, minlength=c.n)
    contacts = int(W.data.sum())
    return {
        "n_neurons": int(c.n),
        "n_edges": int(W.nnz),
        "n_contacts": contacts,
        "frac_inhibitory": float(c.inhibitory.mean()),
        "density": float(W.nnz / (c.n * (c.n - 1))),
        "outdeg_mean": float(outdeg.mean()),
        "outdeg_median": float(np.median(outdeg)),
        "outdeg_max": int(outdeg.max()),
        "indeg_mean": float(indeg.mean()),
        "indeg_max": int(indeg.max()),
        "contacts_per_edge_mean": float(W.data.mean()),
        "n_edges_zero_weight": int((W.data == 0).sum()),
        "n_cell_types": int(len(set(c.cell_type.tolist()) - {""})),
        "frac_typed": float((c.cell_type != "").mean()),
        "w_norm": c.w_norm,
    }
